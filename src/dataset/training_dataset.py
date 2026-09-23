from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train, validation, and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class TrainingDatasetBuilder:
    """Build training examples for the recommendation model."""

    REQUIRED_COLUMNS = {
        "customer_id",
        "article_id",
    }

    LABEL_COLUMN = "label"

    def __init__(self, interactions: pd.DataFrame):
        self.interactions = interactions.copy()

    def validate_columns(self) -> None:
        """Validate the minimum columns required to build examples."""
        missing_columns = self.REQUIRED_COLUMNS.difference(
            self.interactions.columns
        )

        if missing_columns:
            raise ValueError(
                "Interactions dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def build_positive_examples(self) -> pd.DataFrame:
        """Create positive examples from observed customer-item interactions."""
        self.validate_columns()

        positives = self.interactions[
            ["customer_id", "article_id"]
        ].drop_duplicates()

        positives = positives.copy()
        positives[self.LABEL_COLUMN] = 1

        return positives.reset_index(drop=True)

    def sample_negative_examples(
        self,
        articles: pd.DataFrame,
        negative_ratio: int = 1,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generate negative customer-item examples.

        Negative examples are sampled from articles that the customer
        has not interacted with.
        """
        self.validate_columns()

        if "article_id" not in articles.columns:
            raise ValueError(
                "Articles dataset must contain 'article_id'."
            )

        if negative_ratio < 1:
            raise ValueError(
                "negative_ratio must be greater than or equal to 1."
            )

        customers = (
            self.interactions["customer_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        article_ids = (
            articles["article_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not customers or not article_ids:
            return pd.DataFrame(
                columns=["customer_id", "article_id", self.LABEL_COLUMN]
            )

        interacted = (
            self.interactions.groupby("customer_id")["article_id"]
            .apply(set)
            .to_dict()
        )

        rows: list[dict] = []

        for customer_id in customers:
            seen_articles = interacted.get(customer_id, set())

            candidates = [
                article_id
                for article_id in article_ids
                if article_id not in seen_articles
            ]

            if not candidates:
                continue

            target_count = min(
                len(candidates),
                negative_ratio
                * max(1, len(seen_articles)),
            )

            sampled = pd.Series(candidates).sample(
                n=target_count,
                random_state=random_state,
            )

            for article_id in sampled.tolist():
                rows.append(
                    {
                        "customer_id": customer_id,
                        "article_id": article_id,
                        self.LABEL_COLUMN: 0,
                    }
                )

        return pd.DataFrame(
            rows,
            columns=["customer_id", "article_id", self.LABEL_COLUMN],
        ).reset_index(drop=True)

    def build_training_examples(
        self,
        articles: pd.DataFrame,
        negative_ratio: int = 1,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Combine positive and negative recommendation examples."""
        positives = self.build_positive_examples()

        negatives = self.sample_negative_examples(
            articles=articles,
            negative_ratio=negative_ratio,
            random_state=random_state,
        )

        result = pd.concat(
            [positives, negatives],
            ignore_index=True,
        )

        return result.drop_duplicates(
            subset=["customer_id", "article_id"],
            keep="first",
        ).reset_index(drop=True)

    def split_by_customer(
        self,
        dataset: pd.DataFrame,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> DatasetSplit:
        """
        Split examples by customer.

        A customer appears in only one split, preventing customer
        information from leaking between train, validation, and test.
        """
        if not 0 <= validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1."
            )

        if not 0 <= test_fraction < 1:
            raise ValueError(
                "test_fraction must be between 0 and 1."
            )

        if validation_fraction + test_fraction >= 1:
            raise ValueError(
                "validation_fraction + test_fraction must be less than 1."
            )

        if "customer_id" not in dataset.columns:
            raise ValueError(
                "Dataset must contain 'customer_id'."
            )

        customers = (
            dataset["customer_id"]
            .dropna()
            .drop_duplicates()
            .sample(
                frac=1,
                random_state=random_state,
            )
            .tolist()
        )

        total_customers = len(customers)

        test_count = int(total_customers * test_fraction)
        validation_count = int(
            total_customers * validation_fraction
        )

        test_customers = set(
            customers[:test_count]
        )

        validation_customers = set(
            customers[
                test_count : test_count + validation_count
            ]
        )

        test = dataset[
            dataset["customer_id"].isin(test_customers)
        ].copy()

        validation = dataset[
            dataset["customer_id"].isin(validation_customers)
        ].copy()

        train = dataset[
            ~dataset["customer_id"].isin(
                test_customers | validation_customers
            )
        ].copy()

        return DatasetSplit(
            train=train.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            test=test.reset_index(drop=True),
        )

    def build_dataset(
        self,
        articles: pd.DataFrame,
        negative_ratio: int = 1,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> DatasetSplit:
        """Build and split the complete recommendation training dataset."""
        examples = self.build_training_examples(
            articles=articles,
            negative_ratio=negative_ratio,
            random_state=random_state,
        )

        return self.split_by_customer(
            dataset=examples,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_state=random_state,
        )
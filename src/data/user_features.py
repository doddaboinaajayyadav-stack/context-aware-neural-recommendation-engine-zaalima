from __future__ import annotations

import pandas as pd


class UserInteractionFeatureBuilder:
    """Build user-level behavioral features from transaction data."""

    REQUIRED_COLUMNS = {
        "customer_id",
        "article_id",
        "t_dat",
    }

    def __init__(self, transactions: pd.DataFrame):
        self.transactions = transactions.copy()

    def validate_columns(self) -> None:
        """Validate the transaction columns required for feature generation."""
        missing_columns = self.REQUIRED_COLUMNS.difference(
            self.transactions.columns
        )

        if missing_columns:
            raise ValueError(
                "Transactions dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def prepare_dates(self) -> pd.DataFrame:
        """Convert transaction dates into pandas datetime values."""
        self.transactions["t_dat"] = pd.to_datetime(
            self.transactions["t_dat"],
            errors="coerce",
        )

        if self.transactions["t_dat"].isna().any():
            raise ValueError(
                "Transactions dataset contains invalid transaction dates."
            )

        return self.transactions

    def build_interaction_count(self) -> pd.Series:
        """Count total interactions for each customer."""
        return self.transactions.groupby(
            "customer_id"
        ).size().rename("interaction_count")

    def build_unique_article_count(self) -> pd.Series:
        """Count distinct articles interacted with by each customer."""
        return self.transactions.groupby(
            "customer_id"
        )["article_id"].nunique().rename("unique_articles")

    def build_first_interaction(self) -> pd.Series:
        """Find each customer's first recorded interaction date."""
        return self.transactions.groupby(
            "customer_id"
        )["t_dat"].min().rename("first_interaction")

    def build_last_interaction(self) -> pd.Series:
        """Find each customer's most recent interaction date."""
        return self.transactions.groupby(
            "customer_id"
        )["t_dat"].max().rename("last_interaction")

    def build_recency(
        self,
        reference_date: str | pd.Timestamp,
    ) -> pd.Series:
        """Calculate days since each customer's latest interaction."""
        reference = pd.Timestamp(reference_date)

        last_interaction = self.transactions.groupby(
            "customer_id"
        )["t_dat"].max()

        recency = (
            reference - last_interaction
        ).dt.days.rename("recency_days")

        return recency

    def build_features(
        self,
        reference_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Build the complete user-level interaction feature table."""
        self.validate_columns()
        self.prepare_dates()

        features = pd.concat(
            [
                self.build_interaction_count(),
                self.build_unique_article_count(),
                self.build_first_interaction(),
                self.build_last_interaction(),
                self.build_recency(reference_date),
            ],
            axis=1,
        ).reset_index()

        return features
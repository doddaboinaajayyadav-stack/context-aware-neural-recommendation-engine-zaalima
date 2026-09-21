from __future__ import annotations

import pandas as pd


class FeatureBuilder:
    """Integrate user, item, and contextual features."""

    REQUIRED_TRANSACTION_COLUMNS = {
        "customer_id",
        "article_id",
        "t_dat",
    }

    def __init__(
        self,
        transactions: pd.DataFrame,
        user_features: pd.DataFrame | None = None,
        item_features: pd.DataFrame | None = None,
        context_features: pd.DataFrame | None = None,
    ):
        self.transactions = transactions.copy()
        self.user_features = (
            user_features.copy()
            if user_features is not None
            else None
        )
        self.item_features = (
            item_features.copy()
            if item_features is not None
            else None
        )
        self.context_features = (
            context_features.copy()
            if context_features is not None
            else None
        )

    def validate_transactions(self) -> None:
        """Validate required transaction columns."""
        missing = self.REQUIRED_TRANSACTION_COLUMNS.difference(
            self.transactions.columns
        )

        if missing:
            raise ValueError(
                "Transactions dataset is missing required columns: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _validate_key(
        dataframe: pd.DataFrame,
        key: str,
        name: str,
    ) -> None:
        if key not in dataframe.columns:
            raise ValueError(
                f"{name} must contain key column '{key}'"
            )

    def merge_user_features(self) -> pd.DataFrame:
        """Merge customer-level features."""
        self.validate_transactions()

        if self.user_features is None:
            return self.transactions.copy()

        self._validate_key(
            self.user_features,
            "customer_id",
            "User features",
        )

        return self.transactions.merge(
            self.user_features,
            on="customer_id",
            how="left",
            suffixes=("", "_user"),
        )

    def merge_item_features(self) -> pd.DataFrame:
        """Merge article-level features."""
        dataframe = self.merge_user_features()

        if self.item_features is None:
            return dataframe

        self._validate_key(
            self.item_features,
            "article_id",
            "Item features",
        )

        return dataframe.merge(
            self.item_features,
            on="article_id",
            how="left",
            suffixes=("", "_item"),
        )

    def merge_context_features(self) -> pd.DataFrame:
        """Merge contextual features."""
        dataframe = self.merge_item_features()

        if self.context_features is None:
            return dataframe

        if "article_id" not in self.context_features.columns:
            raise ValueError(
                "Context features must contain key column "
                "'article_id'"
            )

        return dataframe.merge(
            self.context_features,
            on="article_id",
            how="left",
            suffixes=("", "_context"),
        )

    def build_features(self) -> pd.DataFrame:
        """Build the complete integrated feature table."""
        return self.merge_context_features()
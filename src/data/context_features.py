from __future__ import annotations

import pandas as pd


class ContextFeatureBuilder:
    """Build contextual and popularity-based features from transactions."""

    REQUIRED_COLUMNS = {
        "article_id",
        "t_dat",
    }

    OPTIONAL_COLUMNS = {
        "customer_id",
        "price",
    }

    def __init__(self, transactions: pd.DataFrame):
        self.transactions = transactions.copy()

    def validate_columns(self) -> None:
        """Validate the minimum transaction columns required."""
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
        self.validate_columns()

        converted_dates = pd.to_datetime(
            self.transactions["t_dat"],
            errors="coerce",
        )

        if converted_dates.isna().any():
            raise ValueError(
                "Transactions dataset contains invalid transaction dates."
            )

        self.transactions["t_dat"] = converted_dates

        return self.transactions

    def build_daily_transaction_count(self) -> pd.DataFrame:
        """Calculate the number of transactions recorded on each date."""
        self.prepare_dates()

        result = (
            self.transactions
            .groupby("t_dat")
            .size()
            .reset_index(name="transaction_count")
            .sort_values("t_dat")
            .reset_index(drop=True)
        )

        return result

    def build_article_popularity(self) -> pd.DataFrame:
        """Calculate overall purchase frequency for each article."""
        self.prepare_dates()

        result = (
            self.transactions
            .groupby("article_id")
            .size()
            .reset_index(name="purchase_count")
            .sort_values(
                ["purchase_count", "article_id"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

        return result

    def build_article_popularity_over_time(self) -> pd.DataFrame:
        """Calculate article purchase frequency for each transaction date."""
        self.prepare_dates()

        result = (
            self.transactions
            .groupby(["article_id", "t_dat"])
            .size()
            .reset_index(name="purchase_count")
            .sort_values(
                ["t_dat", "article_id"]
            )
            .reset_index(drop=True)
        )

        return result

    def build_price_features(self) -> pd.DataFrame:
        """Calculate average observed price for each article."""
        self.validate_columns()

        if "price" not in self.transactions.columns:
            return pd.DataFrame(
                columns=["article_id", "average_price"]
            )

        prices = self.transactions.copy()

        prices["price"] = pd.to_numeric(
            prices["price"],
            errors="coerce",
        )

        result = (
            prices
            .groupby("article_id")["price"]
            .mean()
            .reset_index(name="average_price")
        )

        return result

    def build_features(self) -> pd.DataFrame:
        """Build the complete contextual feature table."""
        self.prepare_dates()

        popularity = self.build_article_popularity()
        prices = self.build_price_features()

        result = popularity.merge(
            prices,
            on="article_id",
            how="left",
        )

        return result
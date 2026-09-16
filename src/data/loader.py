from pathlib import Path

import pandas as pd

from src.data.dataset_config import (
    ARTICLES_FILE,
    CUSTOMERS_FILE,
    TRANSACTIONS_FILE,
)


class DatasetLoader:
    """Load and validate the H&M recommendation dataset."""

    REQUIRED_COLUMNS = {
        "customers": {"customer_id"},
        "articles": {"article_id"},
        "transactions": {
            "t_dat",
            "customer_id",
            "article_id",
        },
    }

    def __init__(
        self,
        customers_path: Path = CUSTOMERS_FILE,
        articles_path: Path = ARTICLES_FILE,
        transactions_path: Path = TRANSACTIONS_FILE,
    ):
        self.customers_path = Path(customers_path)
        self.articles_path = Path(articles_path)
        self.transactions_path = Path(transactions_path)

    def _check_file_exists(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    def _validate_columns(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        required = self.REQUIRED_COLUMNS[dataset_name]
        missing = required.difference(dataframe.columns)

        if missing:
            raise ValueError(
                f"{dataset_name} dataset is missing required columns: "
                f"{sorted(missing)}"
            )

    def load_customers(self) -> pd.DataFrame:
        self._check_file_exists(self.customers_path)

        dataframe = pd.read_csv(self.customers_path)
        self._validate_columns(dataframe, "customers")

        return dataframe

    def load_articles(self) -> pd.DataFrame:
        self._check_file_exists(self.articles_path)

        dataframe = pd.read_csv(self.articles_path)
        self._validate_columns(dataframe, "articles")

        return dataframe

    def load_transactions(self) -> pd.DataFrame:
        self._check_file_exists(self.transactions_path)

        dataframe = pd.read_csv(self.transactions_path)
        self._validate_columns(dataframe, "transactions")

        return dataframe

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {
            "customers": self.load_customers(),
            "articles": self.load_articles(),
            "transactions": self.load_transactions(),
        }
import pandas as pd
import pytest

from src.data.loader import DatasetLoader


def test_loader_validates_customers_columns(tmp_path):
    customers_file = tmp_path / "customers.csv"
    articles_file = tmp_path / "articles.csv"
    transactions_file = tmp_path / "transactions_train.csv"

    pd.DataFrame(
        {
            "customer_id": ["customer_1"],
        }
    ).to_csv(customers_file, index=False)

    pd.DataFrame(
        {
            "article_id": [1001],
        }
    ).to_csv(articles_file, index=False)

    pd.DataFrame(
        {
            "t_dat": ["2026-01-01"],
            "customer_id": ["customer_1"],
            "article_id": [1001],
        }
    ).to_csv(transactions_file, index=False)

    loader = DatasetLoader(
        customers_path=customers_file,
        articles_path=articles_file,
        transactions_path=transactions_file,
    )

    customers = loader.load_customers()

    assert len(customers) == 1
    assert "customer_id" in customers.columns


def test_loader_rejects_missing_required_columns(tmp_path):
    customers_file = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "wrong_column": ["value"],
        }
    ).to_csv(customers_file, index=False)

    loader = DatasetLoader(customers_path=customers_file)

    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_customers()


def test_loader_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    loader = DatasetLoader(customers_path=missing_file)

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        loader.load_customers()
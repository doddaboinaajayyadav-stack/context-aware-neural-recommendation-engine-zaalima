from __future__ import annotations

import pandas as pd
import pytest

from src.data.context_features import ContextFeatureBuilder


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C002", "C002"],
            "article_id": ["1001", "1002", "1001", "1003"],
            "t_dat": [
                "2026-01-01",
                "2026-01-10",
                "2026-01-05",
                "2026-01-20",
            ],
            "price": [10.0, 20.0, 10.0, 30.0],
        }
    )


def test_validate_columns(transactions):
    builder = ContextFeatureBuilder(transactions)

    builder.validate_columns()


def test_validate_columns_rejects_missing_date(transactions):
    transactions = transactions.drop(columns=["t_dat"])

    builder = ContextFeatureBuilder(transactions)

    with pytest.raises(ValueError):
        builder.validate_columns()


def test_prepare_dates(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.prepare_dates()

    assert pd.api.types.is_datetime64_any_dtype(result["t_dat"])


def test_prepare_dates_rejects_invalid_dates(transactions):
    transactions.loc[0, "t_dat"] = "invalid-date"

    builder = ContextFeatureBuilder(transactions)

    with pytest.raises(ValueError):
        builder.prepare_dates()


def test_build_daily_transaction_count(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.build_daily_transaction_count()

    assert "transaction_count" in result.columns
    assert len(result) == 4


def test_build_article_popularity(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.build_article_popularity()

    assert "article_id" in result.columns
    assert "purchase_count" in result.columns

    popularity = result.set_index("article_id")

    assert popularity.loc["1001", "purchase_count"] == 2


def test_build_article_popularity_over_time(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.build_article_popularity_over_time()

    assert "article_id" in result.columns
    assert "t_dat" in result.columns
    assert "purchase_count" in result.columns


def test_build_price_features(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.build_price_features()

    assert "article_id" in result.columns
    assert "average_price" in result.columns


def test_build_features(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.build_features()

    assert not result.empty
    assert "article_id" in result.columns
    assert "purchase_count" in result.columns
    assert "average_price" in result.columns


def test_build_features_preserves_articles(transactions):
    builder = ContextFeatureBuilder(transactions)

    result = builder.build_features()

    assert set(result["article_id"]) == {"1001", "1002", "1003"}
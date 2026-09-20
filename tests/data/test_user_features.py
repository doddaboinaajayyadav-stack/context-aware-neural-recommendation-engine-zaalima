import pandas as pd
import pytest

from src.data.user_features import UserInteractionFeatureBuilder


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C001",
                "C002",
                "C002",
            ],
            "article_id": [
                1001,
                1002,
                1001,
                1003,
                1004,
            ],
            "t_dat": [
                "2025-01-01",
                "2025-01-10",
                "2025-01-20",
                "2025-02-01",
                "2025-02-15",
            ],
        }
    )


def test_validate_columns(transactions):
    builder = UserInteractionFeatureBuilder(transactions)

    builder.validate_columns()


def test_validate_columns_rejects_missing_column():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "article_id": [1001],
        }
    )

    builder = UserInteractionFeatureBuilder(dataframe)

    with pytest.raises(ValueError):
        builder.validate_columns()


def test_prepare_dates(transactions):
    builder = UserInteractionFeatureBuilder(transactions)

    result = builder.prepare_dates()

    assert pd.api.types.is_datetime64_any_dtype(
        result["t_dat"]
    )


def test_prepare_dates_rejects_invalid_dates():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "article_id": [1001],
            "t_dat": ["not-a-date"],
        }
    )

    builder = UserInteractionFeatureBuilder(dataframe)

    with pytest.raises(ValueError):
        builder.prepare_dates()


def test_build_interaction_count(transactions):
    builder = UserInteractionFeatureBuilder(transactions)
    builder.prepare_dates()

    result = builder.build_interaction_count()

    assert result["C001"] == 3
    assert result["C002"] == 2


def test_build_unique_article_count(transactions):
    builder = UserInteractionFeatureBuilder(transactions)
    builder.prepare_dates()

    result = builder.build_unique_article_count()

    assert result["C001"] == 2
    assert result["C002"] == 2


def test_build_first_interaction(transactions):
    builder = UserInteractionFeatureBuilder(transactions)
    builder.prepare_dates()

    result = builder.build_first_interaction()

    assert result["C001"] == pd.Timestamp("2025-01-01")
    assert result["C002"] == pd.Timestamp("2025-02-01")


def test_build_last_interaction(transactions):
    builder = UserInteractionFeatureBuilder(transactions)
    builder.prepare_dates()

    result = builder.build_last_interaction()

    assert result["C001"] == pd.Timestamp("2025-01-20")
    assert result["C002"] == pd.Timestamp("2025-02-15")


def test_build_recency(transactions):
    builder = UserInteractionFeatureBuilder(transactions)
    builder.prepare_dates()

    result = builder.build_recency("2025-03-01")

    assert result["C001"] == 40
    assert result["C002"] == 14


def test_build_features(transactions):
    builder = UserInteractionFeatureBuilder(transactions)

    result = builder.build_features("2025-03-01")

    assert list(result.columns) == [
        "customer_id",
        "interaction_count",
        "unique_articles",
        "first_interaction",
        "last_interaction",
        "recency_days",
    ]

    assert len(result) == 2

    customer_one = result[
        result["customer_id"] == "C001"
    ].iloc[0]

    assert customer_one["interaction_count"] == 3
    assert customer_one["unique_articles"] == 2
    assert customer_one["recency_days"] == 40
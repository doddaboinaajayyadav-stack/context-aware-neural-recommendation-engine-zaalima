import pandas as pd
import pytest

from src.features.feature_builder import FeatureBuilder


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C001"],
            "article_id": [1001, 1002, 1003],
            "t_dat": [
                "2020-09-01",
                "2020-09-02",
                "2020-09-03",
            ],
        }
    )


@pytest.fixture
def user_features():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "interaction_count": [5, 3],
            "unique_article_count": [3, 2],
        }
    )


@pytest.fixture
def item_features():
    return pd.DataFrame(
        {
            "article_id": [1001, 1002, 1003],
            "product_group_name": [
                "Garment Upper body",
                "Shoes",
                "Garment Lower body",
            ],
        }
    )


@pytest.fixture
def context_features():
    return pd.DataFrame(
        {
            "article_id": [1001, 1002, 1003],
            "transaction_year": [2020, 2020, 2020],
            "transaction_month": [9, 9, 9],
        }
    )


def test_validate_transactions(transactions):
    builder = FeatureBuilder(transactions)

    builder.validate_transactions()


def test_validate_transactions_rejects_missing_customer_id():
    dataframe = pd.DataFrame(
        {
            "article_id": [1001],
            "t_dat": ["2020-09-01"],
        }
    )

    builder = FeatureBuilder(dataframe)

    with pytest.raises(ValueError):
        builder.validate_transactions()


def test_merge_user_features(
    transactions,
    user_features,
):
    builder = FeatureBuilder(
        transactions,
        user_features=user_features,
    )

    result = builder.merge_user_features()

    assert "interaction_count" in result.columns
    assert "unique_article_count" in result.columns
    assert len(result) == len(transactions)


def test_merge_user_features_rejects_missing_key(
    transactions,
):
    invalid_user_features = pd.DataFrame(
        {
            "user_id": ["C001"],
            "interaction_count": [5],
        }
    )

    builder = FeatureBuilder(
        transactions,
        user_features=invalid_user_features,
    )

    with pytest.raises(ValueError):
        builder.merge_user_features()


def test_merge_item_features(
    transactions,
    item_features,
):
    builder = FeatureBuilder(
        transactions,
        item_features=item_features,
    )

    result = builder.merge_item_features()

    assert "product_group_name" in result.columns
    assert len(result) == len(transactions)


def test_merge_context_features(
    transactions,
    context_features,
):
    builder = FeatureBuilder(
        transactions,
        context_features=context_features,
    )

    result = builder.merge_context_features()

    assert "transaction_year" in result.columns
    assert "transaction_month" in result.columns
    assert len(result) == len(transactions)


def test_build_features_integrates_all_sources(
    transactions,
    user_features,
    item_features,
    context_features,
):
    builder = FeatureBuilder(
        transactions,
        user_features=user_features,
        item_features=item_features,
        context_features=context_features,
    )

    result = builder.build_features()

    assert "customer_id" in result.columns
    assert "article_id" in result.columns
    assert "interaction_count" in result.columns
    assert "product_group_name" in result.columns
    assert "transaction_year" in result.columns
    assert len(result) == len(transactions)


def test_build_features_preserves_transaction_rows(
    transactions,
    user_features,
    item_features,
    context_features,
):
    builder = FeatureBuilder(
        transactions,
        user_features=user_features,
        item_features=item_features,
        context_features=context_features,
    )

    result = builder.build_features()

    assert len(result) == 3
    assert result["customer_id"].tolist() == [
        "C001",
        "C002",
        "C001",
    ]


def test_optional_feature_sources_are_supported(transactions):
    builder = FeatureBuilder(transactions)

    result = builder.build_features()

    assert list(result.columns) == [
        "customer_id",
        "article_id",
        "t_dat",
    ]
import pandas as pd
import pytest

from src.features.feature_encoder import FeatureEncoder


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C001"],
            "product_group": [
                "Shoes",
                "Garment",
                "Shoes",
            ],
            "interaction_count": [10, 20, 30],
            "unique_articles": [2, 4, 6],
        }
    )


def test_validate_columns(dataframe):
    encoder = FeatureEncoder(dataframe)

    encoder.validate_columns(
        ["customer_id", "product_group"]
    )


def test_validate_columns_rejects_missing_column(dataframe):
    encoder = FeatureEncoder(dataframe)

    with pytest.raises(ValueError):
        encoder.validate_columns(
            ["customer_id", "missing_column"]
        )


def test_encode_categorical_returns_integer_values(
    dataframe,
):
    encoder = FeatureEncoder(dataframe)

    result = encoder.encode_categorical(
        ["product_group"]
    )

    assert result["product_group"].dtype == "int64"


def test_encode_categorical_preserves_row_count(
    dataframe,
):
    encoder = FeatureEncoder(dataframe)

    result = encoder.encode_categorical(
        ["product_group"]
    )

    assert len(result) == len(dataframe)


def test_encode_categorical_creates_mapping(dataframe):
    encoder = FeatureEncoder(dataframe)

    encoder.encode_categorical(
        ["product_group"]
    )

    assert "product_group" in encoder.category_mappings
    assert "Shoes" in encoder.category_mappings[
        "product_group"
    ]


def test_encode_categorical_handles_missing_values():
    dataframe = pd.DataFrame(
        {
            "category": [
                "Shoes",
                None,
                "Garment",
            ]
        }
    )

    encoder = FeatureEncoder(dataframe)

    result = encoder.encode_categorical(
        ["category"]
    )

    assert result["category"].isna().sum() == 0


def test_normalize_numeric(dataframe):
    encoder = FeatureEncoder(dataframe)

    result = encoder.normalize_numeric(
        ["interaction_count"]
    )

    assert result["interaction_count"].min() == 0.0
    assert result["interaction_count"].max() == 1.0


def test_normalize_numeric_preserves_rows(dataframe):
    encoder = FeatureEncoder(dataframe)

    result = encoder.normalize_numeric(
        ["interaction_count"]
    )

    assert len(result) == len(dataframe)


def test_normalize_constant_column():
    dataframe = pd.DataFrame(
        {
            "score": [5, 5, 5]
        }
    )

    encoder = FeatureEncoder(dataframe)

    result = encoder.normalize_numeric(
        ["score"]
    )

    assert result["score"].tolist() == [
        0.0,
        0.0,
        0.0,
    ]


def test_build_model_features(
    dataframe,
):
    encoder = FeatureEncoder(dataframe)

    result = encoder.build_model_features(
        categorical_columns=[
            "product_group"
        ],
        numeric_columns=[
            "interaction_count",
            "unique_articles",
        ],
    )

    assert result["product_group"].dtype == "int64"
    assert result["interaction_count"].min() == 0.0
    assert result["interaction_count"].max() == 1.0
    assert result["unique_articles"].min() == 0.0
    assert result["unique_articles"].max() == 1.0
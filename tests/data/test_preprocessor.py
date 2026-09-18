import pandas as pd
import pytest

from src.data.preprocessor import DatasetPreprocessor


def test_validate_required_columns():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "article_id": [1001, 1002],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    preprocessor.validate_required_columns(
        {"customer_id", "article_id"}
    )


def test_validate_required_columns_rejects_missing_column():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    with pytest.raises(ValueError):
        preprocessor.validate_required_columns(
            {"customer_id", "article_id"}
        )


def test_remove_duplicates():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C002"],
            "article_id": [1001, 1001, 1002],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    result = preprocessor.remove_duplicates()

    assert len(result) == 2


def test_fill_missing_values():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", None, "C003"],
            "price": [10.0, None, 30.0],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    result = preprocessor.fill_missing_values(
        {
            "customer_id": "UNKNOWN",
            "price": 0.0,
        }
    )

    assert result["customer_id"].isna().sum() == 0
    assert result["price"].isna().sum() == 0
    assert result.loc[1, "customer_id"] == "UNKNOWN"
    assert result.loc[1, "price"] == 0.0


def test_normalize_column_types():
    dataframe = pd.DataFrame(
        {
            "article_id": ["1001", "1002"],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    result = preprocessor.normalize_column_types(
        {"article_id": "int64"}
    )

    assert str(result["article_id"].dtype) == "int64"


def test_normalize_column_types_rejects_unknown_column():
    dataframe = pd.DataFrame(
        {
            "article_id": [1001, 1002],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    with pytest.raises(ValueError):
        preprocessor.normalize_column_types(
            {"unknown_column": "int64"}
        )


def test_filter_invalid_rows():
    dataframe = pd.DataFrame(
        {
            "price": [5.0, 10.0, 20.0],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    result = preprocessor.filter_invalid_rows(
        "price",
        10.0,
    )

    assert result["price"].tolist() == [10.0, 20.0]


def test_filter_invalid_rows_rejects_unknown_column():
    dataframe = pd.DataFrame(
        {
            "price": [5.0, 10.0],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    with pytest.raises(ValueError):
        preprocessor.filter_invalid_rows(
            "unknown_column",
            10.0,
        )


def test_preprocess_pipeline():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002"],
            "article_id": ["1001", "1002", "1002"],
            "price": [10.0, None, 20.0],
        }
    )

    preprocessor = DatasetPreprocessor(dataframe)

    result = preprocessor.preprocess(
        required_columns={
            "customer_id",
            "article_id",
            "price",
        },
        fill_values={
            "price": 0.0,
        },
        column_types={
            "article_id": "int64",
        },
    )

    assert len(result) == 3
    assert result["price"].isna().sum() == 0
    assert str(result["article_id"].dtype) == "int64"
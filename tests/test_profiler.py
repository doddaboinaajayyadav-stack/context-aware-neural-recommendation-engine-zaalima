import pandas as pd
import pytest

from src.data.profiler import DatasetProfiler


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002", "C003"],
            "article_id": [1001, 1002, 1002, 1003],
            "price": [10.5, 20.0, 20.0, 30.5],
        }
    )


def test_profile_shape(sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    result = profiler.profile_shape()

    assert result == {
        "rows": 4,
        "columns": 3,
    }


def test_profile_dtypes(sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    result = profiler.profile_dtypes()

    assert set(result.keys()) == {
        "customer_id",
        "article_id",
        "price",
    }


def test_profile_missing_values():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", None, "C003"],
            "price": [10.0, 20.0, None],
        }
    )

    profiler = DatasetProfiler(
        dataframe,
        "test_dataset",
    )

    result = profiler.profile_missing_values()

    assert result["customer_id"] == 1
    assert result["price"] == 1


def test_profile_duplicates(sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    result = profiler.profile_duplicates()

    assert result == 1


def test_profile_unique_values(sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    assert profiler.profile_unique_values("customer_id") == 3
    assert profiler.profile_unique_values("article_id") == 3


def test_profile_unique_values_invalid_column(sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    with pytest.raises(ValueError):
        profiler.profile_unique_values("unknown_column")


def test_generate_report(sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    report = profiler.generate_report()

    assert report["dataset_name"] == "test_dataset"
    assert report["shape"]["rows"] == 4
    assert report["shape"]["columns"] == 3
    assert report["duplicate_rows"] == 1


def test_save_report(tmp_path, sample_dataframe):
    profiler = DatasetProfiler(
        sample_dataframe,
        "test_dataset",
    )

    output_path = tmp_path / "profile.txt"

    profiler.save_report(output_path)

    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    assert "Dataset: test_dataset" in content
    assert "Rows: 4" in content
    assert "Columns: 3" in content
    assert "Duplicate Rows: 1" in content
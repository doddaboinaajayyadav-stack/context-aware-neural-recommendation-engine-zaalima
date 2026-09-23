import pandas as pd
import pytest

from src.dataset.training_dataset import (
    DatasetSplit,
    TrainingDatasetBuilder,
)


@pytest.fixture
def interactions():
    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C002",
                "C003",
            ],
            "article_id": [
                "A001",
                "A002",
                "A002",
                "A003",
            ],
        }
    )


@pytest.fixture
def articles():
    return pd.DataFrame(
        {
            "article_id": [
                "A001",
                "A002",
                "A003",
                "A004",
                "A005",
            ]
        }
    )


def test_validate_columns(interactions):
    builder = TrainingDatasetBuilder(interactions)

    builder.validate_columns()


def test_validate_columns_rejects_missing_customer_id():
    dataframe = pd.DataFrame(
        {
            "article_id": ["A001"],
        }
    )

    builder = TrainingDatasetBuilder(dataframe)

    with pytest.raises(ValueError):
        builder.validate_columns()


def test_validate_columns_rejects_missing_article_id():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001"],
        }
    )

    builder = TrainingDatasetBuilder(dataframe)

    with pytest.raises(ValueError):
        builder.validate_columns()


def test_build_positive_examples(interactions):
    builder = TrainingDatasetBuilder(interactions)

    result = builder.build_positive_examples()

    assert len(result) == 4
    assert set(result["label"]) == {1}
    assert set(result.columns) == {
        "customer_id",
        "article_id",
        "label",
    }


def test_build_positive_examples_removes_duplicates():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C001"],
            "article_id": ["A001", "A001"],
        }
    )

    builder = TrainingDatasetBuilder(dataframe)

    result = builder.build_positive_examples()

    assert len(result) == 1


def test_sample_negative_examples(interactions, articles):
    builder = TrainingDatasetBuilder(interactions)

    result = builder.sample_negative_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    assert not result.empty
    assert set(result["label"]) == {0}


def test_negative_examples_are_unseen(interactions, articles):
    builder = TrainingDatasetBuilder(interactions)

    result = builder.sample_negative_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    seen = set(
        zip(
            interactions["customer_id"],
            interactions["article_id"],
        )
    )

    generated = set(
        zip(
            result["customer_id"],
            result["article_id"],
        )
    )

    assert seen.isdisjoint(generated)


def test_sample_negative_examples_rejects_invalid_ratio(
    interactions,
    articles,
):
    builder = TrainingDatasetBuilder(interactions)

    with pytest.raises(ValueError):
        builder.sample_negative_examples(
            articles=articles,
            negative_ratio=0,
        )


def test_build_training_examples(interactions, articles):
    builder = TrainingDatasetBuilder(interactions)

    result = builder.build_training_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    assert not result.empty
    assert set(result["label"]) == {0, 1}


def test_build_training_examples_has_unique_pairs(
    interactions,
    articles,
):
    builder = TrainingDatasetBuilder(interactions)

    result = builder.build_training_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    pairs = result[
        ["customer_id", "article_id"]
    ]

    assert not pairs.duplicated().any()


def test_split_by_customer(interactions, articles):
    builder = TrainingDatasetBuilder(interactions)

    dataset = builder.build_training_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    result = builder.split_by_customer(
        dataset,
        validation_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )

    assert isinstance(result, DatasetSplit)


def test_split_has_no_customer_leakage(
    interactions,
    articles,
):
    builder = TrainingDatasetBuilder(interactions)

    dataset = builder.build_training_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    result = builder.split_by_customer(
        dataset,
        validation_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )

    train_customers = set(result.train["customer_id"])
    validation_customers = set(result.validation["customer_id"])
    test_customers = set(result.test["customer_id"])

    assert train_customers.isdisjoint(validation_customers)
    assert train_customers.isdisjoint(test_customers)
    assert validation_customers.isdisjoint(test_customers)


def test_split_preserves_all_rows(
    interactions,
    articles,
):
    builder = TrainingDatasetBuilder(interactions)

    dataset = builder.build_training_examples(
        articles=articles,
        negative_ratio=1,
        random_state=42,
    )

    result = builder.split_by_customer(
        dataset,
        validation_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )

    total_rows = (
        len(result.train)
        + len(result.validation)
        + len(result.test)
    )

    assert total_rows == len(dataset)


def test_split_rejects_invalid_fractions(
    interactions,
    articles,
):
    builder = TrainingDatasetBuilder(interactions)

    dataset = builder.build_training_examples(
        articles=articles,
        random_state=42,
    )

    with pytest.raises(ValueError):
        builder.split_by_customer(
            dataset,
            validation_fraction=0.8,
            test_fraction=0.3,
        )


def test_build_dataset_returns_three_splits(
    interactions,
    articles,
):
    builder = TrainingDatasetBuilder(interactions)

    result = builder.build_dataset(
        articles=articles,
        random_state=42,
    )

    assert isinstance(result, DatasetSplit)
    assert isinstance(result.train, pd.DataFrame)
    assert isinstance(result.validation, pd.DataFrame)
    assert isinstance(result.test, pd.DataFrame)
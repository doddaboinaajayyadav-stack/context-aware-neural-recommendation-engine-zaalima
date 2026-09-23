import pandas as pd
import pytest
import torch

from src.dataset.data_loader import RecommendationDataLoaderFactory


@pytest.fixture
def recommendation_data():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C3", "C4"],
            "article_id": [101, 102, 103, 104, 105],
            "user_interactions": [5, 2, 8, 1, 4],
            "unique_articles": [3, 2, 5, 1, 3],
            "item_category": [1, 2, 1, 3, 2],
            "recency_days": [2.0, 10.0, 1.0, 20.0, 5.0],
            "label": [1, 0, 1, 0, 1],
        }
    )


def test_factory_creates_dataloader(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    loader = factory.build()

    assert loader is not None


def test_dataloader_batch_size(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    loader = factory.build()
    batch = next(iter(loader))

    assert batch["features"].shape == (2, 4)
    assert batch["label"].shape == (2,)


def test_dataloader_returns_tensors(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    batch = next(iter(factory.build()))

    assert isinstance(batch["features"], torch.Tensor)
    assert isinstance(batch["label"], torch.Tensor)


def test_dataloader_feature_dtype(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    batch = next(iter(factory.build()))

    assert batch["features"].dtype == torch.float32


def test_dataloader_label_dtype(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    batch = next(iter(factory.build()))

    assert batch["label"].dtype == torch.float32


def test_dataloader_handles_last_partial_batch(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
        shuffle=False,
    )

    batches = list(factory.build())

    assert len(batches) == 3
    assert batches[-1]["features"].shape == (1, 4)
    assert batches[-1]["label"].shape == (1,)


def test_dataloader_length_returns_batch_count(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    assert len(factory) == 3


def test_dataloader_without_shuffle(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
        shuffle=False,
    )

    batches = list(factory.build())

    labels = torch.cat([batch["label"] for batch in batches])

    assert labels.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_factory_exposes_feature_columns(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        batch_size=2,
    )

    assert factory.feature_columns == [
        "user_interactions",
        "unique_articles",
        "item_category",
        "recency_days",
    ]


def test_factory_rejects_invalid_batch_size(recommendation_data):
    with pytest.raises(ValueError, match="batch_size"):
        RecommendationDataLoaderFactory(
            recommendation_data,
            batch_size=0,
        )


def test_factory_rejects_negative_workers(recommendation_data):
    with pytest.raises(ValueError, match="num_workers"):
        RecommendationDataLoaderFactory(
            recommendation_data,
            batch_size=2,
            num_workers=-1,
        )


def test_custom_feature_columns(recommendation_data):
    factory = RecommendationDataLoaderFactory(
        recommendation_data,
        feature_columns=[
            "user_interactions",
            "recency_days",
        ],
        batch_size=2,
    )

    batch = next(iter(factory.build()))

    assert batch["features"].shape == (2, 2)
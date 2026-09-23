import pandas as pd
import pytest
import torch

from src.dataset.pytorch_dataset import RecommendationDataset


@pytest.fixture
def recommendation_data():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C3"],
            "article_id": [101, 102, 103, 104],
            "user_interactions": [5, 2, 8, 1],
            "unique_articles": [3, 2, 5, 1],
            "item_category": [1, 2, 1, 3],
            "recency_days": [2.0, 10.0, 1.0, 20.0],
            "label": [1, 0, 1, 0],
        }
    )


def test_dataset_length(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    assert len(dataset) == 4


def test_dataset_uses_expected_features(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    assert dataset.feature_columns == [
        "user_interactions",
        "unique_articles",
        "item_category",
        "recency_days",
    ]


def test_getitem_returns_dictionary(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    sample = dataset[0]

    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"features", "label"}


def test_getitem_features_are_tensor(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    sample = dataset[0]

    assert isinstance(sample["features"], torch.Tensor)
    assert sample["features"].dtype == torch.float32


def test_getitem_label_is_tensor(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    sample = dataset[0]

    assert isinstance(sample["label"], torch.Tensor)
    assert sample["label"].dtype == torch.float32


def test_getitem_feature_shape(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    sample = dataset[0]

    assert sample["features"].shape == (4,)


def test_dataset_rejects_missing_customer_id(recommendation_data):
    data = recommendation_data.drop(columns=["customer_id"])

    with pytest.raises(ValueError, match="customer_id"):
        RecommendationDataset(data)


def test_dataset_rejects_missing_article_id(recommendation_data):
    data = recommendation_data.drop(columns=["article_id"])

    with pytest.raises(ValueError, match="article_id"):
        RecommendationDataset(data)


def test_dataset_rejects_missing_label(recommendation_data):
    data = recommendation_data.drop(columns=["label"])

    with pytest.raises(ValueError, match="label"):
        RecommendationDataset(data)


def test_dataset_rejects_non_numeric_features(recommendation_data):
    data = recommendation_data.copy()
    data["item_category"] = ["dress", "shirt", "jeans", "jacket"]

    with pytest.raises(TypeError, match="numeric"):
        RecommendationDataset(data)


def test_dataset_rejects_invalid_labels(recommendation_data):
    data = recommendation_data.copy()
    data.loc[0, "label"] = 2

    with pytest.raises(ValueError, match="binary"):
        RecommendationDataset(data)


def test_dataset_rejects_empty_feature_list(recommendation_data):
    with pytest.raises(ValueError, match="At least one feature"):
        RecommendationDataset(
            recommendation_data,
            feature_columns=[],
        )


def test_dataset_rejects_unknown_feature(recommendation_data):
    with pytest.raises(ValueError, match="missing"):
        RecommendationDataset(
            recommendation_data,
            feature_columns=["user_interactions", "unknown_feature"],
        )


def test_get_feature_matrix(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    matrix = dataset.get_feature_matrix()

    assert isinstance(matrix, torch.Tensor)
    assert matrix.shape == (4, 4)
    assert matrix.dtype == torch.float32


def test_get_labels(recommendation_data):
    dataset = RecommendationDataset(recommendation_data)

    labels = dataset.get_labels()

    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (4,)
    assert labels.tolist() == [1.0, 0.0, 1.0, 0.0]
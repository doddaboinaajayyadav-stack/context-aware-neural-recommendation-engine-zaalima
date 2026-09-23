import pytest
import torch

from src.model.two_tower import (
    ItemTower,
    TwoTowerModel,
    UserTower,
)


def test_user_tower_output_shape():
    tower = UserTower(
        input_dim=10,
        hidden_dim=16,
        embedding_dim=8,
    )

    features = torch.randn(4, 10)
    embeddings = tower(features)

    assert embeddings.shape == (4, 8)


def test_item_tower_output_shape():
    tower = ItemTower(
        input_dim=12,
        hidden_dim=16,
        embedding_dim=8,
    )

    features = torch.randn(5, 12)
    embeddings = tower(features)

    assert embeddings.shape == (5, 8)


def test_user_tower_rejects_invalid_input_dim():
    with pytest.raises(ValueError):
        UserTower(input_dim=0)


def test_item_tower_rejects_invalid_input_dim():
    with pytest.raises(ValueError):
        ItemTower(input_dim=0)


def test_two_tower_model_initialization():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        hidden_dim=16,
        embedding_dim=8,
    )

    assert isinstance(model.user_tower, UserTower)
    assert isinstance(model.item_tower, ItemTower)


def test_encode_user_shape():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    user_features = torch.randn(4, 10)

    embeddings = model.encode_user(user_features)

    assert embeddings.shape == (4, 8)


def test_encode_item_shape():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    item_features = torch.randn(6, 12)

    embeddings = model.encode_item(item_features)

    assert embeddings.shape == (6, 8)


def test_embeddings_are_normalized():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    user_features = torch.randn(4, 10)

    embeddings = model.encode_user(user_features)

    norms = torch.linalg.vector_norm(embeddings, dim=1)

    assert torch.allclose(
        norms,
        torch.ones(4),
        atol=1e-6,
    )


def test_score_output_shape():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    user_features = torch.randn(4, 10)
    item_features = torch.randn(4, 12)

    scores = model.score(
        user_features,
        item_features,
    )

    assert scores.shape == (4,)


def test_forward_matches_score():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    user_features = torch.randn(4, 10)
    item_features = torch.randn(4, 12)

    forward_scores = model(
        user_features,
        item_features,
    )

    score_scores = model.score(
        user_features,
        item_features,
    )

    assert torch.allclose(
        forward_scores,
        score_scores,
    )


def test_two_tower_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        TwoTowerModel(
            user_input_dim=0,
            item_input_dim=12,
        )

    with pytest.raises(ValueError):
        TwoTowerModel(
            user_input_dim=10,
            item_input_dim=0,
        )


def test_two_tower_supports_different_batch_sizes_for_embeddings():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    user_features = torch.randn(3, 10)
    item_features = torch.randn(7, 12)

    user_embeddings = model.encode_user(user_features)
    item_embeddings = model.encode_item(item_features)

    assert user_embeddings.shape == (3, 8)
    assert item_embeddings.shape == (7, 8)


def test_score_values_are_between_minus_one_and_one():
    model = TwoTowerModel(
        user_input_dim=10,
        item_input_dim=12,
        embedding_dim=8,
    )

    user_features = torch.randn(5, 10)
    item_features = torch.randn(5, 12)

    scores = model.score(
        user_features,
        item_features,
    )

    assert torch.all(scores >= -1.0)
    assert torch.all(scores <= 1.0)
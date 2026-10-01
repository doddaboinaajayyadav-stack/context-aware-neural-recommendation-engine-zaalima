import pytest
import torch

from src.model.two_tower import TwoTowerModel
from src.retrieval.candidate_retriever import CandidateRetriever
from src.ranking.ranker import CandidateRanker, RankingConfig
from src.pipeline.recommendation_pipeline import (
    RecommendationPipeline,
    RecommendationResult,
)


@pytest.fixture
def pipeline():
    model = TwoTowerModel(
        user_input_dim=4,
        item_input_dim=4,
        embedding_dim=8,
    )

    item_embeddings = torch.randn(6, 8)
    item_ids = [f"item_{i}" for i in range(6)]

    retriever = CandidateRetriever(
        item_embeddings=item_embeddings,
        item_ids=item_ids,
    )

    ranker = CandidateRanker(
        RankingConfig(top_k=3)
    )

    return RecommendationPipeline(
        model=model,
        retriever=retriever,
        ranker=ranker,
    )


def test_pipeline_initialization(pipeline):
    assert pipeline.model is not None
    assert pipeline.retriever is not None
    assert pipeline.ranker is not None


def test_recommend_returns_result(pipeline):
    user_features = torch.randn(1, 4)

    result = pipeline.recommend(
        user_features=user_features,
        top_k=3,
    )

    assert isinstance(result, RecommendationResult)


def test_recommend_returns_top_k_items(pipeline):
    user_features = torch.randn(1, 4)

    result = pipeline.recommend(
        user_features=user_features,
        top_k=3,
    )

    assert len(result.item_ids) == 3
    assert len(result.scores) == 3


def test_recommend_returns_unique_items(pipeline):
    user_features = torch.randn(1, 4)

    result = pipeline.recommend(
        user_features=user_features,
        top_k=3,
    )

    assert len(result.item_ids) == len(set(result.item_ids))


def test_recommend_scores_are_finite(pipeline):
    user_features = torch.randn(1, 4)

    result = pipeline.recommend(
        user_features=user_features,
        top_k=3,
    )

    assert all(torch.isfinite(torch.tensor(result.scores)))


def test_recommend_rejects_non_tensor(pipeline):
    with pytest.raises(TypeError):
        pipeline.recommend(
            user_features=[[1, 2, 3, 4]],
            top_k=3,
        )


def test_recommend_rejects_wrong_dimensions(pipeline):
    with pytest.raises(ValueError):
        pipeline.recommend(
            user_features=torch.randn(4),
            top_k=3,
        )


def test_recommend_rejects_multiple_users(pipeline):
    with pytest.raises(ValueError):
        pipeline.recommend(
            user_features=torch.randn(2, 4),
            top_k=3,
        )


def test_recommend_rejects_invalid_top_k(pipeline):
    user_features = torch.randn(1, 4)

    with pytest.raises(ValueError):
        pipeline.recommend(
            user_features=user_features,
            top_k=0,
        )


def test_recommend_caps_at_available_candidates(pipeline):
    user_features = torch.randn(1, 4)

    result = pipeline.recommend(
        user_features=user_features,
        top_k=100,
    )

    assert len(result.item_ids) == 3


def test_recommend_does_not_require_grad(pipeline):
    user_features = torch.randn(1, 4, requires_grad=True)

    result = pipeline.recommend(
        user_features=user_features,
        top_k=3,
    )

    assert all(isinstance(score, float) for score in result.scores)
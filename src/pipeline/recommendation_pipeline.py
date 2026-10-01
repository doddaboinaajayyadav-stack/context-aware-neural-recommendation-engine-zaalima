from __future__ import annotations

from dataclasses import dataclass

import torch

from src.model.two_tower import TwoTowerModel
from src.retrieval.candidate_retriever import CandidateRetriever
from src.ranking.ranker import CandidateRanker


@dataclass(frozen=True)
class RecommendationResult:
    item_ids: list[str]
    scores: list[float]


class RecommendationPipeline:
    """End-to-end recommendation pipeline.

    User features
        ↓
    User Tower
        ↓
    User Embedding
        ↓
    Candidate Retrieval
        ↓
    Candidate Ranking
        ↓
    Top-K Recommendations
    """

    def __init__(
        self,
        model: TwoTowerModel,
        retriever: CandidateRetriever,
        ranker: CandidateRanker,
    ) -> None:
        if model is None:
            raise ValueError("model must not be None")

        if retriever is None:
            raise ValueError("retriever must not be None")

        if ranker is None:
            raise ValueError("ranker must not be None")

        self.model = model
        self.retriever = retriever
        self.ranker = ranker

    @torch.no_grad()
    def recommend(
        self,
        user_features: torch.Tensor,
        top_k: int = 10,
    ) -> RecommendationResult:
        """Generate Top-K recommendations for one user."""

        if not isinstance(user_features, torch.Tensor):
            raise TypeError("user_features must be a torch.Tensor")

        if user_features.ndim != 2:
            raise ValueError("user_features must be a 2D tensor")

        if user_features.shape[0] != 1:
            raise ValueError(
                "user_features must contain exactly one user"
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.model.eval()

        user_embedding = self.model.encode_user(user_features)

        # The retriever expects a single 1D user embedding.
        user_embedding = user_embedding.squeeze(0)

        retrieved = self.retriever.retrieve(
            user_embedding=user_embedding,
            top_k=top_k,
        )

        if not retrieved:
            return RecommendationResult(
                item_ids=[],
                scores=[],
            )

        candidate_ids = [item_id for item_id, _ in retrieved]
        candidate_scores = [score for _, score in retrieved]

        ranked = self.ranker.rank(
            candidate_ids=candidate_ids,
            scores=candidate_scores,
        )

        return RecommendationResult(
            item_ids=[item_id for item_id, _ in ranked],
            scores=[float(score) for _, score in ranked],
        )
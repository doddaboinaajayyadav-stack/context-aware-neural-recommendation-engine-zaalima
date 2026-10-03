from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class EvaluationResult:
    """Evaluation metrics for a recommendation list."""

    precision: float
    recall: float
    hit_rate: float
    ndcg: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "hit_rate": self.hit_rate,
            "ndcg": self.ndcg,
        }


class RecommendationEvaluator:
    """Evaluate recommendation results using ranking metrics."""

    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        self.k = k

    def evaluate(
        self,
        recommendations: Sequence,
        relevant_items: Sequence,
    ) -> EvaluationResult:
        """Calculate all ranking metrics for one recommendation list."""
        return EvaluationResult(
            precision=precision_at_k(
                recommendations,
                relevant_items,
                self.k,
            ),
            recall=recall_at_k(
                recommendations,
                relevant_items,
                self.k,
            ),
            hit_rate=hit_rate_at_k(
                recommendations,
                relevant_items,
                self.k,
            ),
            ndcg=ndcg_at_k(
                recommendations,
                relevant_items,
                self.k,
            ),
        )

    def evaluate_batch(
        self,
        recommendation_lists: Sequence[Sequence],
        relevant_item_lists: Sequence[Sequence],
    ) -> EvaluationResult:
        """Calculate average ranking metrics across multiple users."""
        if len(recommendation_lists) != len(relevant_item_lists):
            raise ValueError(
                "recommendation_lists and relevant_item_lists "
                "must have the same length."
            )

        if not recommendation_lists:
            raise ValueError("Evaluation data cannot be empty.")

        results = [
            self.evaluate(recommendations, relevant_items)
            for recommendations, relevant_items in zip(
                recommendation_lists,
                relevant_item_lists,
            )
        ]

        count = len(results)

        return EvaluationResult(
            precision=sum(result.precision for result in results) / count,
            recall=sum(result.recall for result in results) / count,
            hit_rate=sum(result.hit_rate for result in results) / count,
            ndcg=sum(result.ndcg for result in results) / count,
        )
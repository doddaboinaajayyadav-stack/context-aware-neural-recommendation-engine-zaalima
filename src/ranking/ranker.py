from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for candidate ranking."""

    top_k: int = 10

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


class CandidateRanker:
    """
    Rank retrieved recommendation candidates using relevance scores.

    The ranker is intentionally model-agnostic: it accepts candidate IDs
    and their predicted relevance scores, then returns candidates ordered
    from highest to lowest score.
    """

    def __init__(self, config: RankingConfig | None = None) -> None:
        self.config = config or RankingConfig()

    @staticmethod
    def validate_inputs(
        candidate_ids: Sequence[str],
        scores: Sequence[float],
    ) -> None:
        """Validate candidate IDs and ranking scores."""

        if len(candidate_ids) != len(scores):
            raise ValueError(
                "candidate_ids and scores must have the same length"
            )

        if len(candidate_ids) == 0:
            raise ValueError("candidate_ids must not be empty")

        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be unique")

        numeric_scores = np.asarray(scores, dtype=float)

        if not np.all(np.isfinite(numeric_scores)):
            raise ValueError("scores must contain only finite values")

    @staticmethod
    def normalize_scores(scores: Sequence[float]) -> np.ndarray:
        """
        Normalize scores to the [0, 1] range using min-max normalization.
        """

        values = np.asarray(scores, dtype=float)

        if values.ndim != 1:
            raise ValueError("scores must be one-dimensional")

        if len(values) == 0:
            raise ValueError("scores must not be empty")

        if not np.all(np.isfinite(values)):
            raise ValueError("scores must contain only finite values")

        minimum = values.min()
        maximum = values.max()

        if np.isclose(maximum, minimum):
            return np.ones_like(values)

        return (values - minimum) / (maximum - minimum)

    def rank(
        self,
        candidate_ids: Sequence[str],
        scores: Sequence[float],
    ) -> list[tuple[str, float]]:
        """
        Return candidates sorted by descending relevance score.

        Ties preserve the original candidate order.
        """

        self.validate_inputs(candidate_ids, scores)

        ranked = list(zip(candidate_ids, map(float, scores)))

        ranked.sort(key=lambda item: item[1], reverse=True)

        return ranked[: self.config.top_k]

    def rank_with_normalized_scores(
        self,
        candidate_ids: Sequence[str],
        scores: Sequence[float],
    ) -> list[tuple[str, float]]:
        """Normalize scores and return the ranked Top-K candidates."""

        self.validate_inputs(candidate_ids, scores)

        normalized = self.normalize_scores(scores)

        return self.rank(candidate_ids, normalized.tolist())

    def top_k_ids(
        self,
        candidate_ids: Sequence[str],
        scores: Sequence[float],
    ) -> list[str]:
        """Return only the IDs of the highest-ranked candidates."""

        ranked = self.rank(candidate_ids, scores)

        return [candidate_id for candidate_id, _ in ranked]
"""Recommendation evaluation metrics."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validate_recommendations(
    recommendations: Sequence,
    relevant_items: Iterable,
) -> set:
    if recommendations is None:
        raise ValueError("recommendations must not be None")

    if relevant_items is None:
        raise ValueError("relevant_items must not be None")

    return set(relevant_items)


def precision_at_k(
    recommendations: Sequence,
    relevant_items: Iterable,
    k: int,
) -> float:
    """Calculate Precision@K.

    Precision@K = relevant recommended items in top K / K.
    """
    _validate_k(k)
    relevant = _validate_recommendations(recommendations, relevant_items)

    if not recommendations:
        return 0.0

    top_k = recommendations[:k]
    hits = sum(item in relevant for item in top_k)

    return hits / len(top_k)


def recall_at_k(
    recommendations: Sequence,
    relevant_items: Iterable,
    k: int,
) -> float:
    """Calculate Recall@K.

    Recall@K = relevant recommended items in top K / total relevant items.
    """
    _validate_k(k)
    relevant = _validate_recommendations(recommendations, relevant_items)

    if not relevant or not recommendations:
        return 0.0

    top_k = recommendations[:k]
    hits = sum(item in relevant for item in top_k)

    return hits / len(relevant)


def hit_rate_at_k(
    recommendations: Sequence,
    relevant_items: Iterable,
    k: int,
) -> float:
    """Calculate Hit Rate@K for a single recommendation list.

    Returns 1.0 if at least one relevant item appears in the top K,
    otherwise 0.0.
    """
    _validate_k(k)
    relevant = _validate_recommendations(recommendations, relevant_items)

    if not relevant or not recommendations:
        return 0.0

    return float(any(item in relevant for item in recommendations[:k]))


def ndcg_at_k(
    recommendations: Sequence,
    relevant_items: Iterable,
    k: int,
) -> float:
    """Calculate binary-relevance NDCG@K.

    A recommendation receives relevance 1 when it belongs to
    relevant_items and 0 otherwise.
    """
    _validate_k(k)
    relevant = _validate_recommendations(recommendations, relevant_items)

    if not relevant or not recommendations:
        return 0.0

    top_k = recommendations[:k]

    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
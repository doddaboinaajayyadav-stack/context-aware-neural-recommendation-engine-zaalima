import math

import pytest

from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    recommendations = [1, 2, 3, 4, 5]
    relevant = {2, 4}

    assert precision_at_k(recommendations, relevant, 5) == pytest.approx(0.4)


def test_precision_at_k_uses_top_k():
    recommendations = [1, 2, 3, 4, 5]
    relevant = {4, 5}

    assert precision_at_k(recommendations, relevant, 3) == pytest.approx(0.0)


def test_recall_at_k():
    recommendations = [1, 2, 3, 4, 5]
    relevant = {2, 4}

    assert recall_at_k(recommendations, relevant, 5) == pytest.approx(1.0)


def test_recall_at_k_partial():
    recommendations = [2, 1, 3, 4, 5]
    relevant = {2, 4}

    assert recall_at_k(recommendations, relevant, 3) == pytest.approx(0.5)


def test_hit_rate_at_k_when_hit_exists():
    recommendations = [1, 2, 3, 4]
    relevant = {3}

    assert hit_rate_at_k(recommendations, relevant, 3) == 1.0


def test_hit_rate_at_k_when_no_hit_exists():
    recommendations = [1, 2, 3, 4]
    relevant = {4}

    assert hit_rate_at_k(recommendations, relevant, 3) == 0.0


def test_ndcg_at_k_perfect_ranking():
    recommendations = [2, 4, 1, 3]
    relevant = {2, 4}

    assert ndcg_at_k(recommendations, relevant, 4) == pytest.approx(1.0)


def test_ndcg_at_k_rewards_better_ranking():
    relevant = {2, 4}

    good = [2, 4, 1, 3]
    bad = [1, 3, 2, 4]

    assert ndcg_at_k(good, relevant, 4) > ndcg_at_k(bad, relevant, 4)


def test_empty_recommendations():
    assert precision_at_k([], {1, 2}, 5) == 0.0
    assert recall_at_k([], {1, 2}, 5) == 0.0
    assert hit_rate_at_k([], {1, 2}, 5) == 0.0
    assert ndcg_at_k([], {1, 2}, 5) == 0.0


def test_empty_relevant_items():
    recommendations = [1, 2, 3]

    assert precision_at_k(recommendations, set(), 3) == 0.0
    assert recall_at_k(recommendations, set(), 3) == 0.0
    assert hit_rate_at_k(recommendations, set(), 3) == 0.0
    assert ndcg_at_k(recommendations, set(), 3) == 0.0


@pytest.mark.parametrize("k", [0, -1, -5])
def test_metrics_reject_invalid_k(k):
    recommendations = [1, 2, 3]
    relevant = {2}

    with pytest.raises(ValueError):
        precision_at_k(recommendations, relevant, k)

    with pytest.raises(ValueError):
        recall_at_k(recommendations, relevant, k)

    with pytest.raises(ValueError):
        hit_rate_at_k(recommendations, relevant, k)

    with pytest.raises(ValueError):
        ndcg_at_k(recommendations, relevant, k)


def test_metrics_accept_k_larger_than_recommendations():
    recommendations = [1, 2, 3]
    relevant = {2}

    assert precision_at_k(recommendations, relevant, 10) == pytest.approx(1 / 3)
    assert recall_at_k(recommendations, relevant, 10) == pytest.approx(1.0)


def test_ndcg_manual_calculation():
    recommendations = [1, 2, 3]
    relevant = {2}

    expected = (1 / math.log2(3)) / 1.0

    assert ndcg_at_k(recommendations, relevant, 3) == pytest.approx(expected)
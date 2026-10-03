from __future__ import annotations

import pytest

from src.evaluation.evaluator import (
    EvaluationResult,
    RecommendationEvaluator,
)


def test_evaluator_default_k():
    evaluator = RecommendationEvaluator()

    assert evaluator.k == 10


def test_evaluator_custom_k():
    evaluator = RecommendationEvaluator(k=5)

    assert evaluator.k == 5


@pytest.mark.parametrize("k", [0, -1, -5])
def test_evaluator_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        RecommendationEvaluator(k=k)


def test_evaluate_returns_evaluation_result():
    evaluator = RecommendationEvaluator(k=3)

    result = evaluator.evaluate(
        recommendations=["A", "B", "C"],
        relevant_items=["A", "C"],
    )

    assert isinstance(result, EvaluationResult)


def test_evaluate_precision():
    evaluator = RecommendationEvaluator(k=3)

    result = evaluator.evaluate(
        recommendations=["A", "B", "C"],
        relevant_items=["A", "C"],
    )

    assert result.precision == pytest.approx(2 / 3)


def test_evaluate_recall():
    evaluator = RecommendationEvaluator(k=3)

    result = evaluator.evaluate(
        recommendations=["A", "B", "C"],
        relevant_items=["A", "C"],
    )

    assert result.recall == pytest.approx(1.0)


def test_evaluate_hit_rate():
    evaluator = RecommendationEvaluator(k=3)

    result = evaluator.evaluate(
        recommendations=["A", "B", "C"],
        relevant_items=["A", "C"],
    )

    assert result.hit_rate == pytest.approx(1.0)


def test_evaluate_ndcg():
    evaluator = RecommendationEvaluator(k=3)

    result = evaluator.evaluate(
        recommendations=["A", "B", "C"],
        relevant_items=["A", "C"],
    )

    assert 0.0 <= result.ndcg <= 1.0


def test_result_as_dict():
    evaluator = RecommendationEvaluator(k=3)

    result = evaluator.evaluate(
        recommendations=["A", "B", "C"],
        relevant_items=["A", "C"],
    )

    metrics = result.as_dict()

    assert set(metrics.keys()) == {
        "precision",
        "recall",
        "hit_rate",
        "ndcg",
    }


def test_evaluate_batch():
    evaluator = RecommendationEvaluator(k=3)

    recommendations = [
        ["A", "B", "C"],
        ["D", "E", "F"],
    ]

    relevant_items = [
        ["A"],
        ["F"],
    ]

    result = evaluator.evaluate_batch(
        recommendations,
        relevant_items,
    )

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.ndcg <= 1.0


def test_evaluate_batch_rejects_mismatched_lengths():
    evaluator = RecommendationEvaluator(k=3)

    with pytest.raises(ValueError):
        evaluator.evaluate_batch(
            [["A", "B"]],
            [["A"], ["B"]],
        )


def test_evaluate_batch_rejects_empty_data():
    evaluator = RecommendationEvaluator(k=3)

    with pytest.raises(ValueError):
        evaluator.evaluate_batch([], [])
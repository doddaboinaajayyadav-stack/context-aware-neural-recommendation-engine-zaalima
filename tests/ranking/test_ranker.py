import numpy as np
import pytest

from src.ranking.ranker import CandidateRanker, RankingConfig


def test_ranking_config_defaults():
    config = RankingConfig()

    assert config.top_k == 10


def test_ranking_config_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        RankingConfig(top_k=0)


def test_ranker_initialization():
    ranker = CandidateRanker()

    assert ranker.config.top_k == 10


def test_ranker_sorts_descending():
    ranker = CandidateRanker()

    ids = ["A", "B", "C"]
    scores = [0.2, 0.9, 0.5]

    result = ranker.rank(ids, scores)

    assert result == [
        ("B", 0.9),
        ("C", 0.5),
        ("A", 0.2),
    ]


def test_ranker_returns_top_k():
    ranker = CandidateRanker(RankingConfig(top_k=2))

    ids = ["A", "B", "C", "D"]
    scores = [0.2, 0.9, 0.5, 0.8]

    result = ranker.rank(ids, scores)

    assert result == [
        ("B", 0.9),
        ("D", 0.8),
    ]


def test_top_k_ids_returns_only_ids():
    ranker = CandidateRanker(RankingConfig(top_k=2))

    ids = ["A", "B", "C"]
    scores = [0.2, 0.9, 0.5]

    assert ranker.top_k_ids(ids, scores) == ["B", "C"]


def test_normalize_scores():
    ranker = CandidateRanker()

    result = ranker.normalize_scores([10.0, 20.0, 30.0])

    np.testing.assert_allclose(
        result,
        np.array([0.0, 0.5, 1.0]),
    )


def test_normalize_constant_scores():
    ranker = CandidateRanker()

    result = ranker.normalize_scores([5.0, 5.0, 5.0])

    np.testing.assert_allclose(
        result,
        np.ones(3),
    )


def test_rank_with_normalized_scores():
    ranker = CandidateRanker()

    ids = ["A", "B", "C"]
    scores = [10.0, 30.0, 20.0]

    result = ranker.rank_with_normalized_scores(ids, scores)

    assert result == [
        ("B", 1.0),
        ("C", 0.5),
        ("A", 0.0),
    ]


def test_rejects_mismatched_lengths():
    ranker = CandidateRanker()

    with pytest.raises(ValueError):
        ranker.rank(["A", "B"], [0.5])


def test_rejects_empty_candidates():
    ranker = CandidateRanker()

    with pytest.raises(ValueError):
        ranker.rank([], [])


def test_rejects_duplicate_candidate_ids():
    ranker = CandidateRanker()

    with pytest.raises(ValueError):
        ranker.rank(["A", "A"], [0.5, 0.8])


def test_rejects_nan_scores():
    ranker = CandidateRanker()

    with pytest.raises(ValueError):
        ranker.rank(["A", "B"], [0.5, np.nan])


def test_rejects_infinite_scores():
    ranker = CandidateRanker()

    with pytest.raises(ValueError):
        ranker.rank(["A", "B"], [0.5, np.inf])


def test_ties_preserve_original_order():
    ranker = CandidateRanker()

    ids = ["A", "B", "C"]
    scores = [0.8, 0.8, 0.5]

    result = ranker.rank(ids, scores)

    assert result == [
        ("A", 0.8),
        ("B", 0.8),
        ("C", 0.5),
    ]
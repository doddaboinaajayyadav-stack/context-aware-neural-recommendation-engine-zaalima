import pytest
import torch

from src.retrieval.candidate_retriever import CandidateRetriever


def make_retriever():
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    item_ids = ["item_a", "item_b", "item_c", "item_d"]

    return CandidateRetriever(embeddings, item_ids)


def test_retriever_initialization():
    retriever = make_retriever()

    assert len(retriever.item_ids) == 4
    assert retriever.item_embeddings.shape == (4, 3)


def test_retrieve_returns_top_k():
    retriever = make_retriever()

    user_embedding = torch.tensor([1.0, 0.0, 0.0])

    results = retriever.retrieve(user_embedding, top_k=2)

    assert len(results) == 2
    assert results[0][0] == "item_a"


def test_results_are_sorted_by_similarity():
    retriever = make_retriever()

    user_embedding = torch.tensor([1.0, 0.0, 0.0])

    results = retriever.retrieve(user_embedding, top_k=4)

    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_cannot_be_zero():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(torch.tensor([1.0, 0.0, 0.0]), top_k=0)


def test_top_k_cannot_be_negative():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(torch.tensor([1.0, 0.0, 0.0]), top_k=-1)


def test_top_k_larger_than_catalog_is_capped():
    retriever = make_retriever()

    results = retriever.retrieve(
        torch.tensor([1.0, 0.0, 0.0]),
        top_k=100,
    )

    assert len(results) == 4


def test_invalid_user_embedding_dimension():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(torch.tensor([1.0, 0.0]))


def test_invalid_item_embedding_dimension():
    with pytest.raises(ValueError):
        CandidateRetriever(
            torch.tensor([1.0, 2.0, 3.0]),
            ["item_a"],
        )


def test_mismatched_item_ids():
    embeddings = torch.randn(3, 4)

    with pytest.raises(ValueError):
        CandidateRetriever(
            embeddings,
            ["item_a", "item_b"],
        )


def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError):
        CandidateRetriever(
            torch.empty((0, 4)),
            [],
        )


def test_integer_embeddings_are_rejected():
    with pytest.raises(ValueError):
        CandidateRetriever(
            torch.tensor([[1, 2], [3, 4]]),
            ["item_a", "item_b"],
        )


def test_similarity_scores_are_in_valid_range():
    retriever = make_retriever()

    results = retriever.retrieve(
        torch.tensor([1.0, 0.0, 0.0]),
        top_k=4,
    )

    assert all(-1.0 <= score <= 1.0 for _, score in results)


def test_item_ids_are_returned_with_scores():
    retriever = make_retriever()

    results = retriever.retrieve(
        torch.tensor([0.0, 1.0, 0.0]),
        top_k=2,
    )

    returned_ids = [item_id for item_id, _ in results]

    assert "item_b" in returned_ids
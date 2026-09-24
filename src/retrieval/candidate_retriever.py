from __future__ import annotations

from typing import Sequence

import torch


class CandidateRetriever:
    """Retrieve top-k item candidates using embedding similarity."""

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: Sequence[str],
    ) -> None:
        if item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a 2D tensor")

        if len(item_ids) != item_embeddings.shape[0]:
            raise ValueError(
                "item_ids length must match the number of item embeddings"
            )

        if item_embeddings.shape[0] == 0:
            raise ValueError("item_embeddings must contain at least one item")

        if not torch.is_floating_point(item_embeddings):
            raise ValueError("item_embeddings must be a floating-point tensor")

        self.item_embeddings = torch.nn.functional.normalize(
            item_embeddings, p=2, dim=1
        )
        self.item_ids = list(item_ids)

    def retrieve(
        self,
        user_embedding: torch.Tensor,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-k items ranked by cosine similarity."""

        if user_embedding.ndim != 1:
            raise ValueError("user_embedding must be a 1D tensor")

        if user_embedding.shape[0] != self.item_embeddings.shape[1]:
            raise ValueError(
                "user_embedding dimension must match item embedding dimension"
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        top_k = min(top_k, len(self.item_ids))

        normalized_user = torch.nn.functional.normalize(
            user_embedding.unsqueeze(0), p=2, dim=1
        )

        scores = torch.matmul(
            normalized_user,
            self.item_embeddings.T,
        ).squeeze(0)

        values, indices = torch.topk(scores, k=top_k)

        return [
            (self.item_ids[index.item()], score.item())
            for score, index in zip(values, indices)
        ]
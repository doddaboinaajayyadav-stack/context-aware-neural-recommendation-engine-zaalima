from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Neural network that converts user features into embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate user embeddings."""
        return self.network(features)


class ItemTower(nn.Module):
    """Neural network that converts item features into embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate item embeddings."""
        return self.network(features)


class TwoTowerModel(nn.Module):
    """Two-tower recommendation model for user-item matching."""

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if user_input_dim <= 0:
            raise ValueError("user_input_dim must be positive")

        if item_input_dim <= 0:
            raise ValueError("item_input_dim must be positive")

        self.user_tower = UserTower(
            input_dim=user_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

        self.item_tower = ItemTower(
            input_dim=item_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Generate normalized user embeddings."""
        embeddings = self.user_tower(user_features)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Generate normalized item embeddings."""
        embeddings = self.item_tower(item_features)
        return F.normalize(embeddings, p=2, dim=-1)

    def score(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate user-item similarity scores."""

        user_embeddings = self.encode_user(user_features)
        item_embeddings = self.encode_item(item_features)

        return torch.sum(
            user_embeddings * item_embeddings,
            dim=-1,
        )

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Generate recommendation scores."""
        return self.score(user_features, item_features)
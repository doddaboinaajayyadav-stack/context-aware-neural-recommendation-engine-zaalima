from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Configuration for two-tower model training."""

    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")


class TwoTowerTrainer:
    """Train a two-tower recommendation model."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.loss_function = nn.BCEWithLogitsLoss()

        self.history: list[float] = []

    def compute_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary recommendation loss."""
        scores = scores.float()
        labels = labels.float()

        return self.loss_function(scores, labels)

    def train_epoch(
        self,
        data_loader: DataLoader,
    ) -> float:
        """Train the model for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for batch in data_loader:
            user_features = batch["user_features"].to(self.device)
            item_features = batch["item_features"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            scores = self.model(
                user_features,
                item_features,
            )

            loss = self.compute_loss(scores, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        if total_batches == 0:
            raise ValueError("DataLoader must contain at least one batch.")

        return total_loss / total_batches

    def fit(
        self,
        data_loader: DataLoader,
    ) -> list[float]:
        """Train the model for the configured number of epochs."""
        self.history = []

        for _ in range(self.config.epochs):
            epoch_loss = self.train_epoch(data_loader)
            self.history.append(epoch_loss)

        return self.history

    def predict_scores(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Generate recommendation scores without updating model weights."""
        self.model.eval()

        user_features = user_features.to(self.device)
        item_features = item_features.to(self.device)

        with torch.no_grad():
            scores = self.model(
                user_features,
                item_features,
            )

        return scores

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "device": self.config.device,
            },
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state."""
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.history = list(
            checkpoint.get("history", [])
        )
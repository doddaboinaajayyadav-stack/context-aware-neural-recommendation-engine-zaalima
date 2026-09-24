from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.model.two_tower import TwoTowerModel
from src.training.trainer import (
    TrainingConfig,
    TwoTowerTrainer,
)


class DummyRecommendationDataset(Dataset):
    """Small dataset used for trainer tests."""

    def __init__(self) -> None:
        self.user_features = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        self.item_features = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            [1.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_features": self.user_features[index],
            "item_features": self.item_features[index],
            "label": self.labels[index],
        }


def create_model() -> TwoTowerModel:
    return TwoTowerModel(
        user_input_dim=4,
        item_input_dim=4,
        embedding_dim=8,
    )


def create_loader() -> DataLoader:
    dataset = DummyRecommendationDataset()

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.epochs == 5
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-5
    assert config.device == "cpu"


def test_training_config_rejects_invalid_epochs() -> None:
    try:
        TrainingConfig(epochs=0)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid epochs.")


def test_training_config_rejects_invalid_learning_rate() -> None:
    try:
        TrainingConfig(learning_rate=0)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid learning rate."
    )


def test_training_config_rejects_negative_weight_decay() -> None:
    try:
        TrainingConfig(weight_decay=-1)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for negative weight decay."
    )


def test_trainer_initialization() -> None:
    model = create_model()
    trainer = TwoTowerTrainer(model)

    assert trainer.model is model
    assert trainer.device.type == "cpu"
    assert trainer.history == []


def test_compute_loss_returns_scalar() -> None:
    trainer = TwoTowerTrainer(create_model())

    scores = torch.tensor(
        [1.0, -1.0],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    loss = trainer.compute_loss(scores, labels)

    assert loss.ndim == 0
    assert loss.item() >= 0


def test_train_epoch_returns_loss() -> None:
    trainer = TwoTowerTrainer(
        create_model(),
        TrainingConfig(epochs=1),
    )

    loss = trainer.train_epoch(create_loader())

    assert isinstance(loss, float)
    assert loss >= 0


def test_fit_returns_loss_history() -> None:
    trainer = TwoTowerTrainer(
        create_model(),
        TrainingConfig(epochs=3),
    )

    history = trainer.fit(create_loader())

    assert len(history) == 3
    assert all(isinstance(loss, float) for loss in history)
    assert all(loss >= 0 for loss in history)


def test_fit_updates_history() -> None:
    trainer = TwoTowerTrainer(
        create_model(),
        TrainingConfig(epochs=2),
    )

    trainer.fit(create_loader())

    assert len(trainer.history) == 2


def test_predict_scores_returns_tensor() -> None:
    trainer = TwoTowerTrainer(create_model())

    user_features = torch.randn(3, 4)
    item_features = torch.randn(3, 4)

    scores = trainer.predict_scores(
        user_features,
        item_features,
    )

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (3,)


def test_predict_scores_does_not_require_grad() -> None:
    trainer = TwoTowerTrainer(create_model())

    user_features = torch.randn(3, 4)
    item_features = torch.randn(3, 4)

    scores = trainer.predict_scores(
        user_features,
        item_features,
    )

    assert scores.requires_grad is False


def test_save_and_load_checkpoint(tmp_path) -> None:
    model = create_model()

    trainer = TwoTowerTrainer(
        model,
        TrainingConfig(epochs=1),
    )

    trainer.fit(create_loader())

    checkpoint_path = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(
        str(checkpoint_path)
    )

    assert checkpoint_path.exists()

    restored_model = create_model()

    restored_trainer = TwoTowerTrainer(
        restored_model,
        TrainingConfig(epochs=1),
    )

    restored_trainer.load_checkpoint(
        str(checkpoint_path)
    )

    assert len(restored_trainer.history) == 1


def test_train_epoch_rejects_empty_dataloader() -> None:
    class EmptyDataset(Dataset):
        def __len__(self) -> int:
            return 0

        def __getitem__(
            self,
            index: int,
        ) -> dict[str, torch.Tensor]:
            raise IndexError

    trainer = TwoTowerTrainer(create_model())

    loader = DataLoader(
        EmptyDataset(),
        batch_size=2,
    )

    try:
        trainer.train_epoch(loader)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for empty DataLoader."
    )
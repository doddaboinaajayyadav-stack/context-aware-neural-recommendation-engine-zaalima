from __future__ import annotations

from typing import Sequence

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset.pytorch_dataset import RecommendationDataset


class RecommendationDataLoaderFactory:
    """Create PyTorch DataLoaders for recommendation datasets."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str] | None = None,
        label_column: str = "label",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        self.dataset = RecommendationDataset(
            data=data,
            feature_columns=feature_columns,
            label_column=label_column,
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def build(self) -> DataLoader:
        """Build and return a PyTorch DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.build())

    @property
    def feature_columns(self) -> list[str]:
        """Return the feature columns used by the dataset."""
        return self.dataset.feature_columns
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation training examples."""

    REQUIRED_COLUMNS = {
        "customer_id",
        "article_id",
        "label",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str] | None = None,
        label_column: str = "label",
    ) -> None:
        self.data = data.reset_index(drop=True).copy()
        self.label_column = label_column

        self._validate_columns()

        if feature_columns is None:
            feature_columns = [
                column
                for column in self.data.columns
                if column not in {
                    "customer_id",
                    "article_id",
                    label_column,
                }
            ]

        self.feature_columns = list(feature_columns)

        if not self.feature_columns:
            raise ValueError("At least one feature column is required.")

        missing_features = [
            column
            for column in self.feature_columns
            if column not in self.data.columns
        ]

        if missing_features:
            raise ValueError(
                "Feature columns are missing from dataset: "
                f"{sorted(missing_features)}"
            )

        self._validate_numeric_features()
        self._validate_labels()

    def _validate_columns(self) -> None:
        """Validate required dataset columns."""
        missing_columns = self.REQUIRED_COLUMNS.difference(
            self.data.columns
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def _validate_numeric_features(self) -> None:
        """Ensure model input features are numeric."""
        non_numeric = [
            column
            for column in self.feature_columns
            if not pd.api.types.is_numeric_dtype(self.data[column])
        ]

        if non_numeric:
            raise TypeError(
                "Feature columns must be numeric: "
                f"{sorted(non_numeric)}"
            )

    def _validate_labels(self) -> None:
        """Validate binary recommendation labels."""
        labels = set(self.data[self.label_column].dropna().unique())

        if not labels.issubset({0, 1}):
            raise ValueError(
                "Labels must contain only binary values 0 and 1."
            )

    def __len__(self) -> int:
        """Return number of training examples."""
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one training example as PyTorch tensors."""
        row = self.data.iloc[index]

        features = torch.tensor(
            row[self.feature_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        label = torch.tensor(
            float(row[self.label_column]),
            dtype=torch.float32,
        )

        return {
            "features": features,
            "label": label,
        }

    def get_feature_matrix(self) -> torch.Tensor:
        """Return the complete feature matrix."""
        values = self.data[self.feature_columns].to_numpy(
            dtype=np.float32
        )

        return torch.tensor(values, dtype=torch.float32)

    def get_labels(self) -> torch.Tensor:
        """Return all labels as a PyTorch tensor."""
        values = self.data[self.label_column].to_numpy(
            dtype=np.float32
        )

        return torch.tensor(values, dtype=torch.float32)
from __future__ import annotations

import pandas as pd


class FeatureEncoder:
    """Encode categorical features for recommendation models."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
        self.category_mappings: dict[str, dict[str, int]] = {}

    def validate_columns(
        self,
        columns: list[str],
    ) -> None:
        """Validate that requested columns exist."""
        missing = [
            column
            for column in columns
            if column not in self.dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Columns not found: {missing}"
            )

    def encode_categorical(
        self,
        columns: list[str],
    ) -> pd.DataFrame:
        """Encode categorical columns using integer IDs."""
        self.validate_columns(columns)

        result = self.dataframe.copy()

        for column in columns:
            values = (
                result[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

            categories = sorted(values.unique())

            mapping = {
                value: index
                for index, value in enumerate(categories)
            }

            self.category_mappings[column] = mapping

            result[column] = values.map(mapping).astype("int64")

        return result

    def normalize_numeric(
        self,
        columns: list[str],
    ) -> pd.DataFrame:
        """Min-max normalize numeric features."""
        self.validate_columns(columns)

        result = self.dataframe.copy()

        for column in columns:
            numeric = pd.to_numeric(
                result[column],
                errors="raise",
            )

            minimum = numeric.min()
            maximum = numeric.max()

            if minimum == maximum:
                result[column] = 0.0
            else:
                result[column] = (
                    (numeric - minimum)
                    / (maximum - minimum)
                )

        return result

    def build_model_features(
        self,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build model-ready encoded features."""
        result = self.dataframe.copy()

        if categorical_columns:
            encoder = FeatureEncoder(result)
            result = encoder.encode_categorical(
                categorical_columns
            )
            self.category_mappings = (
                encoder.category_mappings
            )

        if numeric_columns:
            encoder = FeatureEncoder(result)
            result = encoder.normalize_numeric(
                numeric_columns
            )

        return result
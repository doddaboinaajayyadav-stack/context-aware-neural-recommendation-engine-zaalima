from __future__ import annotations

import pandas as pd


class DatasetPreprocessor:
    """Validate and preprocess recommendation dataset tables."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def validate_required_columns(
        self,
        required_columns: set[str],
    ) -> None:
        """Validate that all required columns are present."""
        missing_columns = required_columns.difference(
            self.dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows and return the cleaned dataframe."""
        self.dataframe = self.dataframe.drop_duplicates().reset_index(
            drop=True
        )

        return self.dataframe

    def fill_missing_values(
        self,
        fill_values: dict[str, object],
    ) -> pd.DataFrame:
        """Fill missing values using column-specific defaults."""
        self.dataframe = self.dataframe.fillna(fill_values)

        return self.dataframe

    def normalize_column_types(
        self,
        column_types: dict[str, str],
    ) -> pd.DataFrame:
        """Convert selected columns to the requested data types."""
        for column, dtype in column_types.items():
            if column not in self.dataframe.columns:
                raise ValueError(
                    f"Column '{column}' does not exist."
                )

            self.dataframe[column] = self.dataframe[column].astype(dtype)

        return self.dataframe

    def filter_invalid_rows(
        self,
        column: str,
        minimum_value: float | int,
    ) -> pd.DataFrame:
        """Remove rows where a numeric column is below a minimum value."""
        if column not in self.dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        self.dataframe = self.dataframe[
            self.dataframe[column] >= minimum_value
        ].reset_index(drop=True)

        return self.dataframe

    def get_dataframe(self) -> pd.DataFrame:
        """Return the current preprocessed dataframe."""
        return self.dataframe

    def preprocess(
        self,
        required_columns: set[str] | None = None,
        fill_values: dict[str, object] | None = None,
        column_types: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Run the standard preprocessing pipeline."""
        if required_columns:
            self.validate_required_columns(required_columns)

        if fill_values:
            self.fill_missing_values(fill_values)

        if column_types:
            self.normalize_column_types(column_types)

        self.remove_duplicates()

        return self.get_dataframe()
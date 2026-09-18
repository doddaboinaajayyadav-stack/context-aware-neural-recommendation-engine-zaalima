from pathlib import Path

import pandas as pd


class DatasetProfiler:
    """Generate data-quality and dataset statistics for recommendation data."""

    def __init__(self, dataframe: pd.DataFrame, dataset_name: str):
        self.dataframe = dataframe
        self.dataset_name = dataset_name

    def profile_shape(self) -> dict[str, int]:
        """Return the number of rows and columns."""
        rows, columns = self.dataframe.shape

        return {
            "rows": rows,
            "columns": columns,
        }

    def profile_dtypes(self) -> dict[str, str]:
        """Return column names and their data types."""
        return {
            column: str(dtype)
            for column, dtype in self.dataframe.dtypes.items()
        }

    def profile_missing_values(self) -> dict[str, int]:
        """Return the number of missing values for each column."""
        return self.dataframe.isna().sum().to_dict()

    def profile_duplicates(self) -> int:
        """Return the number of duplicate rows."""
        return int(self.dataframe.duplicated().sum())

    def profile_unique_values(self, column: str) -> int:
        """Return the number of unique values in a column."""
        if column not in self.dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist in {self.dataset_name}."
            )

        return int(self.dataframe[column].nunique())

    def profile_numeric_statistics(self) -> dict[str, dict[str, float]]:
        """Return basic statistics for numeric columns."""
        numeric_columns = self.dataframe.select_dtypes(
            include="number"
        ).columns

        statistics = {}

        for column in numeric_columns:
            statistics[column] = {
                "min": float(self.dataframe[column].min()),
                "max": float(self.dataframe[column].max()),
                "mean": float(self.dataframe[column].mean()),
            }

        return statistics

    def generate_report(self) -> dict:
        """Generate a complete profiling report."""
        report = {
            "dataset_name": self.dataset_name,
            "shape": self.profile_shape(),
            "dtypes": self.profile_dtypes(),
            "missing_values": self.profile_missing_values(),
            "duplicate_rows": self.profile_duplicates(),
        }

        return report

    def save_report(self, output_path: Path) -> None:
        """Save the profiling report as a text file."""
        report = self.generate_report()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            file.write(f"Dataset: {report['dataset_name']}\n\n")

            file.write("Shape:\n")
            file.write(f"  Rows: {report['shape']['rows']}\n")
            file.write(f"  Columns: {report['shape']['columns']}\n\n")

            file.write("Data Types:\n")
            for column, dtype in report["dtypes"].items():
                file.write(f"  {column}: {dtype}\n")

            file.write("\nMissing Values:\n")
            for column, count in report["missing_values"].items():
                file.write(f"  {column}: {count}\n")

            file.write(
                f"\nDuplicate Rows: {report['duplicate_rows']}\n"
            )
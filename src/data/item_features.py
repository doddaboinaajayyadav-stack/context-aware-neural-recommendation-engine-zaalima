from __future__ import annotations

import pandas as pd


class ItemMetadataFeatureBuilder:
    """Build item-level metadata features from article data."""

    REQUIRED_COLUMNS = {
        "article_id",
    }

    OPTIONAL_COLUMNS = {
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
    }

    def __init__(self, articles: pd.DataFrame):
        self.articles = articles.copy()

    def validate_columns(self) -> None:
        """Validate the minimum article columns required."""
        missing_columns = self.REQUIRED_COLUMNS.difference(
            self.articles.columns
        )

        if missing_columns:
            raise ValueError(
                "Articles dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def normalize_text_columns(self) -> pd.DataFrame:
        """Normalize available categorical text columns."""
        text_columns = [
            column
            for column in self.OPTIONAL_COLUMNS
            if column in self.articles.columns
        ]

        for column in text_columns:
            self.articles[column] = (
                self.articles[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

        return self.articles

    def build_category_features(self) -> pd.DataFrame:
        """Build normalized categorical metadata features."""
        self.validate_columns()
        self.normalize_text_columns()

        feature_columns = [
            column
            for column in [
                "article_id",
                "product_code",
                "prod_name",
                "product_type_no",
                "product_type_name",
                "product_group_name",
                "graphical_appearance_name",
                "colour_group_name",
                "department_name",
                "index_name",
                "index_group_name",
                "section_name",
                "garment_group_name",
            ]
            if column in self.articles.columns
        ]

        return self.articles[feature_columns].copy()

    def get_unique_category_counts(self) -> dict[str, int]:
        """Return unique-value counts for available categorical columns."""
        self.validate_columns()
        self.normalize_text_columns()

        counts = {}

        for column in self.OPTIONAL_COLUMNS:
            if column in self.articles.columns:
                counts[column] = int(
                    self.articles[column].nunique()
                )

        return counts

    def build_features(self) -> pd.DataFrame:
        """Build the complete item metadata feature table."""
        return self.build_category_features()
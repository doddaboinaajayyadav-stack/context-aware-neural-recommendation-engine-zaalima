import pandas as pd
import pytest

from src.data.item_features import ItemMetadataFeatureBuilder


@pytest.fixture
def articles():
    return pd.DataFrame(
        {
            "article_id": [1001, 1002, 1003],
            "product_code": ["P001", "P002", "P003"],
            "prod_name": [
                "Basic T-shirt",
                "Slim Jeans",
                None,
            ],
            "product_type_no": [252, 272, 252],
            "product_type_name": [
                "T-shirt",
                "Trousers",
                "T-shirt",
            ],
            "product_group_name": [
                "Garment Upper body",
                "Garment Lower body",
                "Garment Upper body",
            ],
            "colour_group_name": [
                "Black",
                "Blue",
                None,
            ],
            "department_name": [
                "Jersey",
                "Denim",
                "Jersey",
            ],
            "index_name": [
                "Ladieswear",
                "Ladieswear",
                "Ladieswear",
            ],
            "index_group_name": [
                "Ladieswear",
                "Ladieswear",
                "Ladieswear",
            ],
            "section_name": [
                "Womens Everyday Basics",
                "Womens Everyday Basics",
                "Womens Everyday Basics",
            ],
            "garment_group_name": [
                "Jersey Basic",
                "Trousers",
                "Jersey Basic",
            ],
        }
    )


def test_validate_columns(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    builder.validate_columns()


def test_validate_columns_rejects_missing_article_id():
    dataframe = pd.DataFrame(
        {
            "product_code": ["P001"],
            "product_type_name": ["T-shirt"],
        }
    )

    builder = ItemMetadataFeatureBuilder(dataframe)

    with pytest.raises(ValueError):
        builder.validate_columns()


def test_normalize_text_columns_fills_missing_values(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    result = builder.normalize_text_columns()

    assert result.loc[2, "prod_name"] == "Unknown"
    assert result.loc[2, "colour_group_name"] == "Unknown"


def test_normalize_text_columns_strips_whitespace():
    dataframe = pd.DataFrame(
        {
            "article_id": [1001],
            "product_type_name": ["  T-shirt  "],
        }
    )

    builder = ItemMetadataFeatureBuilder(dataframe)

    result = builder.normalize_text_columns()

    assert result.loc[0, "product_type_name"] == "T-shirt"


def test_build_category_features_contains_expected_columns(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    result = builder.build_category_features()

    expected_columns = [
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
    ]

    assert list(result.columns) == expected_columns


def test_build_category_features_preserves_row_count(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    result = builder.build_category_features()

    assert len(result) == 3


def test_build_category_features_normalizes_missing_metadata(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    result = builder.build_category_features()

    assert result.loc[2, "prod_name"] == "Unknown"
    assert result.loc[2, "colour_group_name"] == "Unknown"


def test_get_unique_category_counts(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    counts = builder.get_unique_category_counts()

    assert counts["product_type_name"] == 2
    assert counts["product_group_name"] == 2
    assert counts["colour_group_name"] == 3


def test_build_features_returns_item_feature_table(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    result = builder.build_features()

    assert "article_id" in result.columns
    assert "product_type_name" in result.columns
    assert "colour_group_name" in result.columns
    assert len(result) == 3


def test_build_features_preserves_article_ids(articles):
    builder = ItemMetadataFeatureBuilder(articles)

    result = builder.build_features()

    assert result["article_id"].tolist() == [1001, 1002, 1003]
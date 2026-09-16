from src.data.dataset_config import (
    ARTICLES_FILE,
    CUSTOMERS_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRANSACTIONS_FILE,
)


def test_raw_data_directory_is_defined():
    assert RAW_DATA_DIR.name == "raw"


def test_processed_data_directory_is_defined():
    assert PROCESSED_DATA_DIR.name == "processed"


def test_dataset_files_have_expected_names():
    assert CUSTOMERS_FILE.name == "customers.csv"
    assert ARTICLES_FILE.name == "articles.csv"
    assert TRANSACTIONS_FILE.name == "transactions_train.csv"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CUSTOMERS_FILE = RAW_DATA_DIR / "customers.csv"
ARTICLES_FILE = RAW_DATA_DIR / "articles.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions_train.csv"

DATASET_NAME = "H&M Personalized Fashion Recommendations"
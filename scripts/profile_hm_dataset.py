import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "hm_dataset_profile.json"

report = {
    "dataset": "H&M Personalized Fashion Recommendations",
    "files": {},
}

for name in ["articles", "customers"]:
    path = RAW / f"{name}.csv"
    df = pd.read_csv(path, low_memory=False)
    report["files"][name] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
            if v > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
    print(f"{name}: {len(df):,} rows, {len(df.columns)} columns")

path = RAW / "transactions_train.csv"
rows = 0
missing = {}
date_min = None
date_max = None
customers = set()
articles = set()
duplicate_rows_in_chunks = 0

for chunk in pd.read_csv(path, chunksize=200_000):
    rows += len(chunk)
    for col, count in chunk.isna().sum().items():
        missing[col] = missing.get(col, 0) + int(count)

    if "t_dat" in chunk:
        dates = pd.to_datetime(chunk["t_dat"], errors="coerce")
        low, high = dates.min(), dates.max()
        if pd.notna(low):
            date_min = min(date_min, str(low.date())) if date_min else str(low.date())
        if pd.notna(high):
            date_max = max(date_max, str(high.date())) if date_max else str(high.date())

    if "customer_id" in chunk:
        customers.update(chunk["customer_id"].dropna().unique())
    if "article_id" in chunk:
        articles.update(chunk["article_id"].dropna().astype(str).unique())

    duplicate_rows_in_chunks += int(chunk.duplicated().sum())
    print(f"Scanned {rows:,} transaction rows...", end="\r")

report["files"]["transactions_train"] = {
    "rows": rows,
    "columns": list(chunk.columns),
    "missing_values": {k: v for k, v in missing.items() if v},
    "date_min": date_min,
    "date_max": date_max,
    "unique_customers": len(customers),
    "unique_articles": len(articles),
    "duplicate_rows_within_chunks": duplicate_rows_in_chunks,
    "duplicate_note": "Duplicate count is within individual chunks, not a global duplicate count.",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nProfile saved to: {OUT}")
print(json.dumps(report["files"]["transactions_train"], indent=2))

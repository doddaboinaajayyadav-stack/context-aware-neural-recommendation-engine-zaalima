from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import RecommendationEvaluator

SEED = 42
MAX_ROWS = 100_000
MAX_CUSTOMERS = 1_500
MAX_ARTICLES = 2_000
KS = (5, 10)

RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
CHECKPOINT = ROOT / "models" / "hm_two_tower_sample.pt"


def main():
    import torch

    print("Loading model checkpoint mappings...")
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    user_to_idx = {
        value: idx for idx, value in enumerate(checkpoint["user_ids"])
    }
    item_to_idx = {
        value: idx for idx, value in enumerate(checkpoint["item_ids"])
    }

    print("Loading sampled H&M transactions...")
    transactions = pd.read_csv(
        RAW / "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        nrows=MAX_ROWS,
        dtype={"customer_id": "string", "article_id": "string"},
    )
    transactions = transactions.dropna(
        subset=["t_dat", "customer_id", "article_id"]
    )
    transactions["t_dat"] = pd.to_datetime(
        transactions["t_dat"], errors="coerce"
    )
    transactions = transactions.dropna(subset=["t_dat"])

    top_users = transactions["customer_id"].value_counts().head(
        MAX_CUSTOMERS
    ).index
    top_items = transactions["article_id"].value_counts().head(
        MAX_ARTICLES
    ).index
    transactions = transactions[
        transactions["customer_id"].isin(top_users)
        & transactions["article_id"].isin(top_items)
    ].copy()

    positives = transactions[
        ["customer_id", "article_id", "t_dat"]
    ].drop_duplicates(["customer_id", "article_id"])
    positives["user_idx"] = positives["customer_id"].map(user_to_idx)
    positives["item_idx"] = positives["article_id"].map(item_to_idx)
    positives = positives.dropna(subset=["user_idx", "item_idx"]).copy()
    positives["user_idx"] = positives["user_idx"].astype(int)
    positives["item_idx"] = positives["item_idx"].astype(int)

    positives = positives.sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    split_at = int(len(positives) * 0.8)
    train_positive = positives.iloc[:split_at].copy()
    test_positive = positives.iloc[split_at:].copy()

    print("Calculating popularity from training interactions only...")
    popularity = (
        train_positive["item_idx"].value_counts().to_dict()
    )
    catalog = list(range(len(item_to_idx)))
    global_ranked = sorted(
        catalog, key=lambda item: (-popularity.get(item, 0), item)
    )

    seen_by_user = (
        train_positive.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )

    results = {k: [] for k in KS}
    evaluated_users = 0
    skipped_users = 0

    print("Evaluating popularity recommendations...")
    for user_idx, group in test_positive.groupby("user_idx", sort=True):
        relevant = set(group["item_idx"].astype(int).tolist())
        seen = seen_by_user.get(int(user_idx), set())
        ranked = [item for item in global_ranked if item not in seen]

        if not relevant or not ranked:
            skipped_users += 1
            continue

        for k in KS:
            result = RecommendationEvaluator(k=k).evaluate(
                recommendations=ranked[:k],
                relevant_items=relevant,
            )
            results[k].append(result)

        evaluated_users += 1

    if evaluated_users == 0:
        raise RuntimeError("No eligible test users found.")

    metrics = {}
    for k, user_results in results.items():
        for name, attr in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("hit_rate", "hit_rate"),
            ("ndcg", "ndcg"),
        ):
            metrics[f"{name}_at_{k}"] = float(
                np.mean([getattr(result, attr) for result in user_results])
            )

    report = {
        "dataset": "H&M Personalized Fashion Recommendations",
        "method": "Popularity baseline",
        "evaluation_type": "same seeded random interaction split as Two-Tower evaluator",
        "seed": SEED,
        "sampled_transaction_row_limit": MAX_ROWS,
        "test_positive_interactions": int(len(test_positive)),
        "evaluated_users": evaluated_users,
        "skipped_users": skipped_users,
        "candidate_catalog_size": len(item_to_idx),
        "metrics": metrics,
        "limitations": [
            "Uses a sampled catalogue, not the full H&M catalogue.",
            "Popularity is calculated from training interactions only.",
            "Random interaction split, not a temporal evaluation.",
            "Metrics describe this sampled experiment only.",
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "hm_popularity_baseline.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPopularity baseline results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Evaluated users: {evaluated_users:,}")
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()

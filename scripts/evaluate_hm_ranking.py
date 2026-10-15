import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.two_tower import TwoTowerModel
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
    torch.set_num_threads(2)
    np.random.seed(SEED)

    print("Loading checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    user_ids = checkpoint["user_ids"]
    item_ids = checkpoint["item_ids"]
    user_to_idx = {value: idx for idx, value in enumerate(user_ids)}
    item_to_idx = {value: idx for idx, value in enumerate(item_ids)}

    model = TwoTowerModel(
        user_input_dim=checkpoint["user_input_dim"],
        item_input_dim=checkpoint["item_input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Reconstructing original sampled dataset...")
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

    # Exclude only items present in the training split.
    # Held-out positive items must remain eligible candidates.
    seen_by_user = (
        train_positive.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )

    item_matrix = torch.eye(len(item_ids), dtype=torch.float32)
    evaluator_results = {k: [] for k in KS}
    evaluated_users = 0
    skipped_users = 0
    user_groups = test_positive.groupby("user_idx", sort=True)

    print(f"Test positive interactions: {len(test_positive):,}")
    print("Scoring candidate articles for test users...")

    with torch.inference_mode():
        for user_idx, group in user_groups:
            relevant = set(group["item_idx"].astype(int).tolist())
            seen = seen_by_user.get(int(user_idx), set())
            candidates = [
                item_idx
                for item_idx in range(len(item_ids))
                if item_idx not in seen
            ]

            if not relevant or not candidates:
                skipped_users += 1
                continue

            user_vector = F.one_hot(
                torch.tensor([int(user_idx)]),
                num_classes=len(user_ids),
            ).float()
            user_features = user_vector.expand(len(candidates), -1)
            candidate_features = item_matrix[candidates]

            scores = model(user_features, candidate_features).reshape(-1)
            top_count = min(max(KS), len(candidates))
            top_positions = torch.topk(scores, k=top_count).indices.tolist()
            ranked_items = [candidates[position] for position in top_positions]

            for k in KS:
                result = RecommendationEvaluator(k=k).evaluate(
                    recommendations=ranked_items,
                    relevant_items=relevant,
                )
                evaluator_results[k].append(result)

            evaluated_users += 1

    if evaluated_users == 0:
        raise RuntimeError(
            "No eligible test users found. Check the checkpoint mappings "
            "and sampled transaction data."
        )

    metrics = {}
    for k, results in evaluator_results.items():
        metrics[f"precision_at_{k}"] = float(
            np.mean([result.precision for result in results])
        )
        metrics[f"recall_at_{k}"] = float(
            np.mean([result.recall for result in results])
        )
        metrics[f"hit_rate_at_{k}"] = float(
            np.mean([result.hit_rate for result in results])
        )
        metrics[f"ndcg_at_{k}"] = float(
            np.mean([result.ndcg for result in results])
        )

    report = {
        "dataset": "H&M Personalized Fashion Recommendations",
        "evaluation_type": "preliminary random interaction split",
        "seed": SEED,
        "sampled_transaction_row_limit": MAX_ROWS,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "test_positive_interactions": int(len(test_positive)),
        "evaluated_users": evaluated_users,
        "skipped_users": skipped_users,
        "candidate_catalog_size": len(item_ids),
        "metrics": metrics,
        "limitations": [
            "Uses a sampled catalogue, not the full H&M catalogue.",
            "Random interaction split, not a temporal evaluation.",
            "Only training interactions are excluded from candidates; "
            "held-out positives remain eligible.",
            "Metrics describe this sampled experiment only.",
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "hm_ranking_evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nRanking evaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"\nEvaluated users: {evaluated_users:,}")
    print(f"Report saved to: {output}")

if __name__ == "__main__":
    main()

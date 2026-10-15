from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.two_tower import TwoTowerModel

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MAX_ROWS = 100_000
MAX_CUSTOMERS = 1_500
MAX_ARTICLES = 2_000
EPOCHS = 3
BATCH_SIZE = 256
SEED = 42
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class PairDataset(Dataset):
    """Return integer IDs and labels; one-hot vectors are created per batch."""

    def __init__(self, pairs: pd.DataFrame):
        self.users = pairs["user_idx"].to_numpy(dtype=np.int64)
        self.items = pairs["item_idx"].to_numpy(dtype=np.int64)
        self.labels = pairs["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(self.users[index], dtype=torch.long),
            "item_idx": torch.tensor(self.items[index], dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def collate_pairs(batch):
    return {
        "user_idx": torch.stack([row["user_idx"] for row in batch]),
        "item_idx": torch.stack([row["item_idx"] for row in batch]),
        "label": torch.stack([row["label"] for row in batch]),
    }


def make_examples(positive_rows, seen_by_user, item_count, seed):
    """Pair positives with one randomly sampled unobserved article each."""
    rng = np.random.default_rng(seed)
    positive_examples = positive_rows[["user_idx", "item_idx"]].copy()
    positive_examples["label"] = 1.0
    negative_rows = []

    for user_idx, _item_idx in positive_rows[
        ["user_idx", "item_idx"]
    ].itertuples(index=False, name=None):
        user_idx = int(user_idx)
        seen = seen_by_user.get(user_idx, set())
        candidate = None

        for _ in range(50):
            proposed = int(rng.integers(0, item_count))
            if proposed not in seen:
                candidate = proposed
                break

        if candidate is not None:
            negative_rows.append(
                {"user_idx": user_idx, "item_idx": candidate, "label": 0.0}
            )

    if negative_rows:
        examples = pd.concat(
            [positive_examples, pd.DataFrame(negative_rows)],
            ignore_index=True,
        )
    else:
        examples = positive_examples

    return examples.sample(frac=1, random_state=seed).reset_index(drop=True)


def batch_features(batch, user_count, item_count):
    """Build one-hot feature matrices for only the current batch."""
    user_features = F.one_hot(
        batch["user_idx"], num_classes=user_count
    ).to(torch.float32)
    item_features = F.one_hot(
        batch["item_idx"], num_classes=item_count
    ).to(torch.float32)
    return user_features, item_features, batch["label"]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    transactions_path = RAW_DIR / "transactions_train.csv"
    articles_path = RAW_DIR / "articles.csv"
    for path in (transactions_path, articles_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Reading at most {MAX_ROWS:,} transaction rows...")
    transactions = pd.read_csv(
        transactions_path,
        usecols=["t_dat", "customer_id", "article_id"],
        nrows=MAX_ROWS,
        dtype={"customer_id": "string", "article_id": "string"},
    )
    articles = pd.read_csv(
        articles_path,
        usecols=["article_id"],
        dtype={"article_id": "string"},
    )

    transactions = transactions.dropna(
        subset=["t_dat", "customer_id", "article_id"]
    )
    transactions["t_dat"] = pd.to_datetime(
        transactions["t_dat"], errors="coerce"
    )
    transactions = transactions.dropna(subset=["t_dat"])
    articles = articles.dropna(subset=["article_id"]).drop_duplicates()

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

    if len(positives) < 100:
        raise RuntimeError(
            f"Only {len(positives)} positive pairs remain. "
            "Increase MAX_ROWS or check the CSV files."
        )

    user_ids = sorted(positives["customer_id"].unique().tolist())
    item_ids = sorted(positives["article_id"].unique().tolist())
    user_to_idx = {value: idx for idx, value in enumerate(user_ids)}
    item_to_idx = {value: idx for idx, value in enumerate(item_ids)}
    positives["user_idx"] = positives["customer_id"].map(user_to_idx)
    positives["item_idx"] = positives["article_id"].map(item_to_idx)
    positives = positives.reset_index(drop=True)

    # Random interaction split, not a temporal or cold-start split.
    positives = positives.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split_at = int(len(positives) * 0.8)
    train_positive = positives.iloc[:split_at].copy()
    test_positive = positives.iloc[split_at:].copy()

    # Build the training exclusion set from training interactions only.
    # Held-out interactions must not influence negative sampling.
    train_seen_by_user = (
        train_positive.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )

    train_examples = make_examples(
        train_positive, train_seen_by_user, len(item_ids), SEED
    )

    # Test negatives use training history as the exclusion set too.
    test_examples = make_examples(
        test_positive, train_seen_by_user, len(item_ids), SEED + 1
    )

    if train_examples["label"].nunique() < 2:
        raise RuntimeError("Training data must include positive and negative labels.")
    if test_examples["label"].nunique() < 2:
        raise RuntimeError("Test data must include positive and negative labels.")

    user_count, item_count = len(user_ids), len(item_ids)
    print(f"Transactions retained: {len(transactions):,}")
    print(f"Unique positive pairs: {len(positives):,}")
    print(f"Mapped customers: {user_count:,}")
    print(f"Mapped articles: {item_count:,}")
    print(f"Train examples: {len(train_examples):,}")
    print(f"Test examples: {len(test_examples):,}")

    train_loader = DataLoader(
        PairDataset(train_examples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pairs,
    )
    test_loader = DataLoader(
        PairDataset(test_examples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )

    model = TwoTowerModel(
        user_input_dim=user_count,
        item_input_dim=item_count,
        hidden_dim=32,
        embedding_dim=16,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_count = 0.0, 0

        for batch in train_loader:
            user_features, item_features, labels = batch_features(
                batch, user_count, item_count
            )
            optimizer.zero_grad()
            logits = model(user_features, item_features).reshape(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            count = len(labels)
            total_loss += loss.item() * count
            total_count += count

        mean_loss = total_loss / max(total_count, 1)
        history.append(mean_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} - train loss: {mean_loss:.4f}")

    model.eval()
    labels_list, scores_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            user_features, item_features, labels = batch_features(
                batch, user_count, item_count
            )
            logits = model(user_features, item_features).reshape(-1)
            labels_list.extend(labels.numpy().tolist())
            scores_list.extend(logits.numpy().tolist())

    y_true = np.asarray(labels_list, dtype=np.int32)
    y_score = np.asarray(scores_list, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(y_score, -30, 30)))
    y_pred = (probabilities >= 0.5).astype(np.int32)
    accuracy = float(np.mean(y_pred == y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    precision = tp / max(int(np.sum(y_pred == 1)), 1)
    recall = tp / max(int(np.sum(y_true == 1)), 1)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_score))
    except ImportError:
        auc = None

    checkpoint_path = MODEL_DIR / "hm_two_tower_sample.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_ids": user_ids,
            "item_ids": item_ids,
            "user_input_dim": user_count,
            "item_input_dim": item_count,
            "hidden_dim": 32,
            "embedding_dim": 16,
            "training_loss": history,
            "seed": SEED,
        },
        checkpoint_path,
    )

    report = {
        "dataset": "H&M Personalized Fashion Recommendations",
        "sampled_transaction_rows": int(len(transactions)),
        "unique_positive_pairs": int(len(positives)),
        "customers": user_count,
        "articles": item_count,
        "train_examples": int(len(train_examples)),
        "test_examples": int(len(test_examples)),
        "epochs": EPOCHS,
        "training_loss_by_epoch": history,
        "sampled_test_accuracy": accuracy,
        "sampled_test_precision": float(precision),
        "sampled_test_recall": float(recall),
        "sampled_test_roc_auc": auc,
        "evaluation_note": (
            "Preliminary binary metrics on random held-out interactions "
            "with sampled negatives; not full-catalog Recall@K or NDCG."
        ),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    report_path = REPORT_DIR / "hm_sample_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHeld-out sampled-pair evaluation:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {auc if auc is not None else 'unavailable (install scikit-learn)'}")
    print(f"\nCheckpoint saved: {checkpoint_path}")
    print(f"Report saved: {report_path}")
    print("This is a bounded initial experiment, not full-dataset training.")


if __name__ == "__main__":
    main()


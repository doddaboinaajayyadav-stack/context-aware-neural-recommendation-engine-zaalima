import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
from src.model.two_tower import TwoTowerModel

CHECKPOINT = ROOT / "models" / "hm_two_tower_sample.pt"
ARTICLES_CSV = ROOT / "data" / "raw" / "articles.csv"
OUTPUT_DIR = ROOT / "reports"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    user_ids = [str(x) for x in checkpoint["user_ids"]]
    item_ids = [str(x) for x in checkpoint["item_ids"]]

    model = TwoTowerModel(
        user_input_dim=checkpoint["user_input_dim"],
        item_input_dim=checkpoint["item_input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    customer_id = user_ids[0]
    user_idx = 0

    print("Checkpoint loaded successfully.")
    print("Sample customer:", customer_id)
    print("Known customers:", len(user_ids))
    print("Known model items:", len(item_ids))

    user_features = torch.zeros(
        (len(item_ids), len(user_ids)), dtype=torch.float32
    )
    user_features[:, user_idx] = 1.0
    item_features = torch.eye(len(item_ids), dtype=torch.float32)

    with torch.no_grad():
        scores = model(user_features, item_features).reshape(-1)

    columns = [
        "article_id", "prod_name", "product_type_name",
        "product_group_name", "colour_group_name",
        "department_name", "detail_desc",
    ]
    articles = pd.read_csv(
        ARTICLES_CSV,
        usecols=columns,
        dtype={"article_id": str},
    ).fillna("")

    article_lookup = articles.drop_duplicates("article_id").set_index("article_id")
    ranked_indices = torch.argsort(scores, descending=True).tolist()

    recommendations = []
    for idx in ranked_indices:
        article_id = item_ids[idx]
        if article_id not in article_lookup.index:
            continue

        product = article_lookup.loc[article_id]
        recommendations.append({
            "customer_id": customer_id,
            "article_id": article_id,
            "product_name": str(product["prod_name"]),
            "product_type": str(product["product_type_name"]),
            "product_group": str(product["product_group_name"]),
            "colour": str(product["colour_group_name"]),
            "department": str(product["department_name"]),
            "description": str(product["detail_desc"]),
            "score": round(float(scores[idx]), 6),
        })
        if len(recommendations) >= 10:
            break

    output_path = OUTPUT_DIR / "sample_recommendations.json"
    output_path.write_text(
        json.dumps(recommendations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTop 10 recommendations:")
    for rank, item in enumerate(recommendations, start=1):
        print(
            f'{rank:2}. {item["product_name"]} '
            f'| {item["product_type"]} '
            f'| Colour: {item["colour"]} '
            f'| Score: {item["score"]}'
        )

    print("\nRecommendation count:", len(recommendations))
    print("Saved to:", output_path)
    print("Scores are preliminary model similarity scores.")


if __name__ == "__main__":
    main()

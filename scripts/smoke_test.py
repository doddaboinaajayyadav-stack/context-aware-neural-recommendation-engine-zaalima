
import sys
from pathlib import Path

import torch

# Add the project root so Python can find the src package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.two_tower import TwoTowerModel
from src.retrieval.candidate_retriever import CandidateRetriever
from src.ranking.ranker import CandidateRanker, RankingConfig
from src.pipeline.recommendation_pipeline import RecommendationPipeline


def main():
    torch.manual_seed(42)

    # Create a model for the integration test.
    model = TwoTowerModel(
        user_input_dim=4,
        item_input_dim=5,
        hidden_dim=16,
        embedding_dim=8,
    )

    # Generate synthetic item features for testing.
    item_features = torch.randn(20, 5)
    item_ids = [f"DEMO_ITEM_{i:03d}" for i in range(20)]

    model.eval()
    with torch.no_grad():
        item_embeddings = model.encode_item(item_features)

    # Set up retrieval, ranking, and the recommendation pipeline.
    retriever = CandidateRetriever(item_embeddings, item_ids)
    ranker = CandidateRanker(RankingConfig(top_k=5))

    pipeline = RecommendationPipeline(
        model=model,
        retriever=retriever,
        ranker=ranker,
    )

    # Generate recommendations for one synthetic user.
    user_features = torch.randn(1, 4)
    result = pipeline.recommend(user_features, top_k=5)

    print("End-to-end smoke test: PASSED")
    print(f"Recommendations returned: {len(result.item_ids)}")

    for item_id, score in zip(result.item_ids, result.scores):
        print(f"{item_id}: {score:.4f}")

    print("\nIMPORTANT:")
    print("This test uses synthetic features and an untrained model.")
    print("The results do not represent real recommendation quality.")


if __name__ == "__main__":
    main()

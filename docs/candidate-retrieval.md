# Candidate Retrieval

## Overview

The candidate retrieval layer converts a user embedding into a ranked list of
candidate items using embedding similarity.

```text
User Embedding
      ↓
L2 Normalization
      ↓
Similarity with Item Embeddings
      ↓
Top-K Selection
      ↓
Ranked Candidates
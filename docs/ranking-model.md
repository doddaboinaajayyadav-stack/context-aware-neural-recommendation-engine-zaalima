# Ranking Model

## Overview

The ranking stage receives candidates produced by the retrieval layer and
orders them according to their predicted relevance scores.

```text
Retrieved Candidates
        ↓
Candidate Scores
        ↓
Score Validation
        ↓
Score Normalization
        ↓
Descending Ranking
        ↓
Top-K Recommendations
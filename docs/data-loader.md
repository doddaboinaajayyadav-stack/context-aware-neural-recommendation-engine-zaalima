# PyTorch DataLoader

## Overview

The recommendation engine uses PyTorch `DataLoader` to convert the
recommendation dataset into mini-batches suitable for model training.

## Pipeline

```text
Processed Features
       ↓
RecommendationDataset
       ↓
RecommendationDataLoaderFactory
       ↓
Mini-batches
       ↓
Recommendation Model
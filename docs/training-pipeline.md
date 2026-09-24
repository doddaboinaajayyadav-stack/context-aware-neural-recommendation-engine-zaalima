# Training Pipeline

## Overview

The training pipeline trains the Two-Tower recommendation model using
PyTorch.

The pipeline connects the prepared recommendation dataset and PyTorch
DataLoader to the Two-Tower model.

```text
Training Dataset
       ↓
PyTorch Dataset
       ↓
DataLoader
       ↓
Two-Tower Model
       ↓
Recommendation Scores
       ↓
Binary Loss
       ↓
Backpropagation
       ↓
Optimizer Update
       ↓
Trained Model


raining Step

For each batch:

User Features ──────→ User Tower ──────→ User Embedding
                                              │
                                              ↓
                                         Similarity
                                              ↑
                                              │
Item Features ──────→ Item Tower ──────→ Item Embedding
                                              │
                                              ↓
                                         Score / Logit
                                              │
                                              ↓
                                            Loss
                                              │
                                              ↓
                                       Backpropagation
                                              │
                                              ↓
                                       Optimizer Step

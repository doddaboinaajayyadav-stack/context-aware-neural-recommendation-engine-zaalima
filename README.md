# Context-Aware Neural Recommendation Engine

A deep learning recommendation system designed for e-commerce and content platforms.

The system goes beyond traditional collaborative filtering by combining:

- User metadata
- Historical interaction sequences
- Item metadata
- Contextual signals
- Neural embeddings
- Approximate Nearest Neighbor retrieval

## Project Objective

The goal is to build a personalized recommendation engine capable of retrieving relevant products for users based on both long-term preferences and short-term contextual behavior.

## Dataset

Primary Dataset:

**H&M Personalized Fashion Recommendations**

The dataset contains:

- Customer information
- Article/product information
- Historical transactions
- Product metadata
- Multi-year purchase interactions

## Planned Architecture

```text
H&M Dataset
      |
      v
Data Processing
      |
      v
Feature Engineering
      |
      +-------------------+
      |                   |
      v                   v
 User Features       Item Features
      |                   |
      +---------+---------+
                |
                v
        Two-Tower Network
        /               \
       /                 \
User Query Tower     Item Candidate Tower
       \                 /
        \               /
         v             v
          Embedding Space
                |
                v
        ANN Candidate Retrieval
                |
                v
          Redis Feature Store
                |
                v
             FastAPI
                |
                v
       Top-K Recommendations

             ^
             |
          Airflow
     ML Pipeline Scheduling
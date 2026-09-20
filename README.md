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


     ## Project Status

**Currently: Day 3 — Raw Dataset Loader**

Implemented the initial raw dataset loading layer with:

- Centralized dataset configuration
- Pandas CSV loading
- Required-column validation
- Missing-file validation
- Unit tests
- Data ingestion documentation

Test status:

**6 tests passed**

Next milestone:

**Dataset profiling and quality analysis.**

## Development Progress

### Day 4 — Dataset Profiling & Data Quality

Implemented a reusable dataset profiling component for the recommendation
engine.

#### Completed

- Added `DatasetProfiler` in `src/data/profiler.py`
- Added dataset shape profiling
- Added data-type inspection
- Added missing-value analysis
- Added duplicate-row detection
- Added unique-value analysis
- Added basic numeric statistics
- Added profiling report generation
- Added profiling report persistence
- Added automated profiler tests
- Added profiling documentation

#### Testing

```text
14 tests passed

### Day 6 — User Interaction Feature Engineering

- Implemented customer-level behavioral feature generation.
- Added interaction count and unique article count.
- Added first and last interaction dates.
- Added configurable recency calculation.
- Added transaction-date validation.
- Added 10 automated tests.
- **Test status: 33 total tests passing.**

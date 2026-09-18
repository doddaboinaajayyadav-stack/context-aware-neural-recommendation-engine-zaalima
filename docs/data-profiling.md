# Dataset Profiling & Data Quality

## Purpose

The dataset profiling stage provides an initial understanding of the
H&M Personalized Fashion Recommendations dataset before preprocessing
and feature engineering.

The profiler is designed to identify basic structural and data-quality
characteristics that will guide subsequent recommendation-engineering
steps.

## Profiling Components

The `DatasetProfiler` provides the following capabilities:

- Dataset row and column counts
- Column data types
- Missing-value counts
- Duplicate-row detection
- Unique-value counting
- Basic numeric statistics
- Profiling report generation
- Profiling report persistence

## Implementation

The profiling logic is implemented in:

```text
src/data/profiler.py
# Dataset Validation & Preprocessing

## Purpose

The preprocessing stage prepares recommendation data for subsequent
feature engineering and model development.

The preprocessing component is designed to validate required columns,
handle missing values, normalize selected data types, remove exact
duplicate rows, and filter invalid numeric records when required.

## Implementation

The preprocessing logic is implemented in:

```text
src/data/preprocessor.py
# User Interaction Feature Engineering

## Overview

Day 6 implements user-level behavioral feature engineering for the
Context-Aware Neural Recommendation Engine.

The feature builder transforms transaction-level records into
customer-level behavioral signals that can later be used by the
neural recommendation model.

## Features Generated

For each customer, the pipeline generates:

- `interaction_count`
  - Total number of recorded interactions.

- `unique_articles`
  - Number of distinct articles interacted with.

- `first_interaction`
  - Date of the customer's first recorded interaction.

- `last_interaction`
  - Date of the customer's most recent interaction.

- `recency_days`
  - Number of days since the customer's latest interaction relative
    to a configurable reference date.

## Processing Flow

```text
Transaction Data
       |
       v
Validate Required Columns
       |
       v
Convert Transaction Dates
       |
       v
Group Transactions by Customer
       |
       +----> Interaction Count
       |
       +----> Unique Article Count
       |
       +----> First Interaction
       |
       +----> Last Interaction
       |
       +----> Recency
       |
       v
User Feature Table
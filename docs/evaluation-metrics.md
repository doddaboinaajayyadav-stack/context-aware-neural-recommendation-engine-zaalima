# Recommendation Evaluation Metrics

## Overview

The recommendation engine evaluates recommendation quality using standard
top-K ranking metrics.

The evaluation layer measures how well the generated recommendations match
the relevant items for a user.

## Metrics

### Precision@K

Measures the proportion of recommended items in the top K that are relevant.

```text
Precision@K =
relevant items in top K / number of recommended items in top K
# Item Metadata Feature Engineering

## Overview

Day 7 implements item-level metadata feature engineering for the
Context-Aware Neural Recommendation Engine.

The feature builder transforms article metadata into a normalized
item feature table that can later be used as part of the candidate
representation in the recommendation system.

## Features Generated

The implementation supports the following article metadata:

- `article_id`
  - Unique identifier for each item.

- `product_code`
  - Product-level identifier.

- `prod_name`
  - Product name.

- `product_type_no`
  - Numeric product type identifier.

- `product_type_name`
  - Human-readable product type.

- `product_group_name`
  - Product group classification.

- `graphical_appearance_name`
  - Graphical appearance classification.

- `colour_group_name`
  - Colour classification.

- `department_name`
  - Department classification.

- `index_name`
  - Product index classification.

- `index_group_name`
  - Product index-group classification.

- `section_name`
  - Section classification.

- `garment_group_name`
  - Garment group classification.

## Processing Flow

```text
Article Metadata
       |
       v
Validate Required Columns
       |
       v
Normalize Available Metadata
       |
       +----> Product Information
       |
       +----> Product Type
       |
       +----> Product Group
       |
       +----> Colour
       |
       +----> Department
       |
       +----> Index
       |
       +----> Section
       |
       +----> Garment Group
       |
       v
Item Feature Table
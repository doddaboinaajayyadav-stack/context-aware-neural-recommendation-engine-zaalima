# Two-Tower Recommendation Model

## Overview

The recommendation engine uses a two-tower neural architecture.

```text
User Features
      |
      v
 User Tower
      |
      v
User Embedding
      |
      +-------- Similarity --------+
                                 |
Item Embedding                   v
      ^                    Recommendation
      |
 Item Tower
      ^
      |
Item Features

## User Tower

The user tower converts user-level behavioral features into a dense embedding.

```text
User Features
      ↓
Linear Layer
      ↓
ReLU
      ↓
Linear Layer
      ↓
User Embedding



One correction: **don't add an extra empty ` ``` ` before the `text` block**. The nested code block should be closed once after `User Embedding`.

So the final version is:

```markdown
## User Tower

The user tower converts user-level behavioral features into a dense embedding.

```text
User Features
      ↓
Linear Layer
      ↓
ReLU
      ↓
Linear Layer
      ↓
User Embedding



For our project, this belongs in the **Two-Tower / model architecture documentation**, not inside the Python code.
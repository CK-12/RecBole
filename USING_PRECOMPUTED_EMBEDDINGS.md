# Using Pre-computed Embeddings with Existing Context Recommender Models

This guide explains how to use pre-computed embeddings (e.g., from text encoders) with existing RecBole context recommender models like EulerNet, KD_DAGFM, DeepFM, etc.

## Solution Overview

We've created a new base class `ContextRecommenderWithEmbeddings` that extends `ContextRecommender` to support pre-computed embeddings. To enable this feature in any existing model, simply change the base class.

## Quick Start

### Step 1: Modify Your Model to Inherit from `ContextRecommenderWithEmbeddings`

For any context recommender model, change the import and base class:

**Before:**
```python
from recbole.model.abstract_recommender import ContextRecommender

class YourModel(ContextRecommender):
    def __init__(self, config, dataset):
        super(YourModel, self).__init__(config, dataset)
        # ... rest of your model
```

**After:**
```python
from recbole.model.abstract_recommender import ContextRecommenderWithEmbeddings

class YourModel(ContextRecommenderWithEmbeddings):
    def __init__(self, config, dataset):
        super(YourModel, self).__init__(config, dataset)
        # ... rest of your model (no other changes needed!)
```

That's it! Your model now supports pre-computed embeddings.

### Step 2: Configure Pre-computed Embedding Fields

In your config, specify which fields contain pre-computed embeddings:

```python
config_dict = {
    'model': 'YourModel',
    'embedding_size': 768,  # Must match your embedding dimension
    'precomputed_embedding_fields': ['text_embedding'],  # Field names with embeddings
    # ... other config parameters
}
```

### Step 3: Add Embeddings to Interactions

When creating interactions, add your pre-computed embeddings:

```python
import torch
from sentence_transformers import SentenceTransformer

# Initialize your embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text
text_list = ["Item description 1", "Item description 2"]
embeddings = embedder.encode(text_list)  # Shape: [batch_size, embedding_dim]
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

# Add to interaction
interaction['text_embedding'] = embeddings_tensor.to(device)
```

## Examples

### Example 1: EulerNet with Text Embeddings

**Modified eulernet.py:**
```python
from recbole.model.abstract_recommender import ContextRecommenderWithEmbeddings

class EulerNet(ContextRecommenderWithEmbeddings):  # Changed base class
    def __init__(self, config, dataset):
        super(EulerNet, self).__init__(config, dataset)
        # ... rest of EulerNet code unchanged
```

**Usage:**
```python
config_dict = {
    'model': 'EulerNet',
    'embedding_size': 768,
    'precomputed_embedding_fields': ['text_embedding'],
    # ... other EulerNet config
}

# In your training loop:
for batch in train_data:
    # Encode text descriptions
    text_embeddings = embedder.encode(item_descriptions)
    batch['text_embedding'] = torch.tensor(text_embeddings).to(device)

    # Train normally - embeddings are used directly!
    loss = model.calculate_loss(batch)
```

### Example 2: KD_DAGFM with Text Embeddings

**Modified kd_dagfm.py:**
```python
from recbole.model.abstract_recommender import ContextRecommenderWithEmbeddings

class KD_DAGFM(ContextRecommenderWithEmbeddings):  # Changed base class
    def __init__(self, config, dataset):
        super(KD_DAGFM, self).__init__(config, dataset)
        # ... rest of KD_DAGFM code unchanged
```

## How It Works

1. `ContextRecommenderWithEmbeddings` extends `ContextRecommender`
2. It overrides `embed_input_fields()` to check for pre-computed embeddings
3. If a field is in `precomputed_embedding_fields` and exists in the interaction:
   - It uses the embedding directly (no re-embedding)
   - Combines it with regular embeddings
4. All other functionality remains unchanged

## Double Tower Mode

For models using double tower architecture (e.g., DSSM), you need to specify which pre-computed embeddings belong to user vs item:

```python
config_dict = {
    'model': 'YourModel',
    'double_tower': True,
    'embedding_size': 768,
    'precomputed_embedding_fields_user': ['user_text_embedding'],  # User embeddings
    'precomputed_embedding_fields_item': ['item_text_embedding'],  # Item embeddings
    # ... other config
}
```

If you don't specify `precomputed_embedding_fields_user` and `precomputed_embedding_fields_item`,
all embeddings in `precomputed_embedding_fields` will be treated as item embeddings (common case).

## Important Notes

1. **Embedding Dimension**: The `embedding_size` in config must match your pre-computed embedding dimension.

2. **Field Names**: Fields in `precomputed_embedding_fields` should NOT be defined in your dataset as float fields. They should be added dynamically to interactions.

3. **Shape Requirements**:
   - Single embedding per sample: `[batch_size, embedding_dim]`
   - Multiple embeddings per sample: `[batch_size, num_fields, embedding_dim]`

4. **Device**: Make sure embeddings are on the same device as your model.

5. **Double Tower**: For double tower models, use `precomputed_embedding_fields_user` and `precomputed_embedding_fields_item` to specify which embeddings belong to which tower.

5. **No Code Changes Needed**: Once you change the base class, all existing model code (forward, calculate_loss, etc.) works without modification!

## Models That Work

Any model that inherits from `ContextRecommender` can use this feature by simply changing the base class:

- ✅ EulerNet
- ✅ KD_DAGFM
- ✅ DeepFM
- ✅ NFM
- ✅ WideDeep
- ✅ AutoInt
- ✅ DCN
- ✅ FFM
- ✅ And all other context recommender models!

## Files Created

- `recbole/model/abstract_recommender.py`: Added `ContextRecommenderWithEmbeddings` class
- `recbole/model/context_aware_recommender/eulernet_with_embeddings.py`: Example EulerNet with embedding support
- `recbole/model/context_aware_recommender/kd_dagfm_with_embeddings.py`: Example KD_DAGFM with embedding support

You can use these as templates or modify the original files directly.

# HARP Encoder: Step-by-Step Explanation

## Overview

The encoder is the first component of the HARP system. It takes preprocessed claim data (after `process.py` and `encodings.py`) and converts it into a unified, dense representation that can be used by downstream components (risk prediction and policy networks).

## Data Structure After Preprocessing

After running your preprocessing pipeline (`process.py` → `encode()` → `encodings.py`), your data contains:

### Categorical Features (Encoded as Integers)
- **CLM_DRG_CD**: DRG code (e.g., 123, -1 for missing, 1000 for OTH)
- **ADMTNG_ICD9_DGNS_CD**: Admitting diagnosis code
- **ICD9_DGNS_CD_1** through **ICD9_DGNS_CD_10**: Diagnosis codes
- **ICD9_PRCDR_CD_1** through **ICD9_PRCDR_CD_6**: Procedure codes

**Encoding Scheme** (from `encodings.py`):
- Missing/None/NONE/MISSING → `-1`
- V codes → `10000 + number` (for DGNS), `110000 + number` (for PRCDR)
- E codes → `20000 + number` (for DGNS), `120000 + number` (for PRCDR)
- OTH → `300000`
- Invalid → `400000`
- Regular numeric codes → as-is

### Numerical Features (Floats)
- **CLM_PMT_AMT**: Payment amount (e.g., 15000.0)
- **CLM_UTLZTN_DAY_CNT**: Utilization day count (e.g., 5.0)
- **NCH_PRMRY_PYR_CLM_PD_AMT**: Primary payer amount
- **NCH_BENE_IP_DDCTBL_AMT**: Deductible amount
- **NCH_BENE_PTA_COINSRNC_LBLTY_AM**: Coinsurance amount

### Labels (Not Used by Encoder)
- **claim_status**: 0 (rejected) or 1 (accepted)
- **reviewer**: 1, 2, or 3 (reviewer level)

---

## Encoder Processing Pipeline (Step by Step)

### STEP 1: Categorical Features → Embeddings

**Input:**
```python
categorical_features = {
    'CLM_DRG_CD': tensor([123, 456, 789]),      # Shape: (batch_size,)
    'ICD9_DGNS_CD_1': tensor([25000, 30000, -1])
}
```

**What Happens:**
- Each categorical feature has its own **Embedding layer** (like a lookup table)
- The embedding layer converts integer indices into dense vectors
- Example: `CLM_DRG_CD` index `123` → vector `[0.1, -0.3, 0.7, ...]` (32 dimensions)
- During training, similar codes learn similar embeddings

**Output:**
```python
[
    tensor([[0.1, -0.3, ...], [0.2, 0.1, ...]]),  # CLM_DRG_CD embeddings (batch, 32)
    tensor([[0.5, 0.2, ...], [0.3, -0.1, ...]])   # ICD9_DGNS_CD_1 embeddings (batch, 32)
]
```

**Why Embeddings?**
- Categorical codes are sparse (one-hot would be huge: 50000 dimensions for 50000 codes)
- Embeddings compress to dense vectors (e.g., 32 dimensions)
- Similar codes can learn similar representations
- Much more efficient and expressive than one-hot encoding

---

### STEP 2: Concatenate Categorical Embeddings

**Input:** List of embedding tensors from Step 1

**What Happens:**
- All categorical embeddings are concatenated along the feature dimension
- Example: If we have 3 features with dims [32, 32, 16], result is `(batch, 80)`

**Output:**
```python
tensor([[0.1, -0.3, ..., 0.5, 0.2, ...]])  # Shape: (batch_size, total_embedding_dim)
```

**Why Concatenate?**
- Combines all categorical information into a single vector
- Preserves information from all categorical features
- Ready to be combined with numerical features

---

### STEP 3: Numerical Features → Normalization + Projection

**Input:**
```python
numerical_features = tensor([[15000.0, 5.0, 12000.0, 500.0, 200.0]])  # Shape: (batch_size, 5)
```

**What Happens:**

**3a. Normalization (Z-score):**
- Formula: `(x - mean) / std`
- Centers data around 0, scales to unit variance
- Example: `[15000, 5, 12000]` → `[-0.5, 1.2, 0.3]` (if mean=16000, std=2000)
- Uses precomputed mean/std from training data

**3b. Linear Projection:**
- Projects normalized numerical features to dense representation
- Example: `(batch, 5)` → `(batch, 32)` via Linear layer
- This matches the dimensionality of categorical embeddings

**Output:**
```python
tensor([[0.2, -0.1, 0.5, ...]])  # Shape: (batch_size, numerical_proj_dim)
```

**Why Normalize?**
- Payment amounts (15000) and day counts (5) are on very different scales
- Neural networks learn better when features are on similar scales
- Normalization ensures all features contribute equally

**Why Project?**
- Numerical features (5 features) need to be combined with categorical embeddings (80+ dimensions)
- Projection converts them to a matching dimensionality
- Allows the network to learn how to combine categorical and numerical information

---

### STEP 4: Combine Categorical + Numerical

**Input:**
- Categorical embeddings from Step 2: `(batch, cat_dim)` e.g., `(batch, 80)`
- Numerical projections from Step 3: `(batch, num_dim)` e.g., `(batch, 32)`

**What Happens:**
- Concatenate along feature dimension
- Example: `(batch, 80) + (batch, 32) = (batch, 112)`

**Output:**
```python
tensor([[0.1, -0.3, ..., 0.2, -0.1, ...]])  # Shape: (batch, total_dim)
```

**Why Combine?**
- Creates a unified representation with both categorical and numerical information
- The feedforward network can now learn complex interactions between all features

---

### STEP 5: Feedforward Network

**Input:** Combined features from Step 4: `(batch, total_dim)` e.g., `(batch, 112)`

**What Happens (through each hidden layer):**
1. **Linear transformation**: `(batch, input_dim)` → `(batch, hidden_dim)`
2. **ReLU**: Non-linear activation (enables learning complex patterns)
3. **Dropout**: Randomly zeros some neurons (prevents overfitting)

**Architecture Example** (default: `[128, 64]`):
```
Layer 1: (batch, 112) → Linear -> ReLU -> Dropout -> (batch, 128)
Layer 2: (batch, 128) → Linear -> ReLU -> Dropout -> (batch, 64)
Final:   (batch, 64) → Linear -> (batch, output_dim=64)
```

**Output:**
```python
tensor([[0.3, -0.2, 0.1, ..., 0.5]])  # Shape: (batch_size, output_dim=64)
```

**Why Feedforward Network?**
- Learns complex non-linear patterns from the combined features
- Progressively reduces dimensionality (112 → 128 → 64)
- Creates a rich, compressed representation

**Why ReLU?**
- Non-linear activation enables learning complex patterns
- Simple and effective: max(0, x)

**Why Dropout?**
- Randomly sets some neurons to 0 during training
- Prevents overfitting (memorizing training data)
- Forces the network to learn robust features

---

### STEP 6: Use Encoded Representation

The encoded representation is now ready for:
- **f_networks.py**: Predict risk (BCE loss) and entropy (uncertainty)
- **Policy network**: Make routing decisions (which reviewer stage: 1, 2, or 3)

---

## Usage Examples

### Example 1: Automatic Configuration from Dataset (Recommended)

```python
import pandas as pd
from harp.models.encoder import create_encoder_from_dataset

# Load preprocessed data (after encode() in process.py)
df_train = pd.read_csv("harp/data/raw/harp_dataset_encoded/train.csv")

# Create encoder automatically - it analyzes the data and configures itself
encoder = create_encoder_from_dataset(df_train)

# The encoder automatically:
# 1. Identifies categorical vs numerical columns
# 2. Calculates vocabulary sizes for categorical features
# 3. Computes normalization statistics for numerical features
# 4. Sets appropriate embedding dimensions based on vocab sizes
```

### Example 2: Manual Configuration

```python
from harp.models.encoder import Encoder

# Define categorical features: (vocab_size, embedding_dim)
categorical_config = {
    'CLM_DRG_CD': (1000, 32),           # 1000 unique DRG codes, 32-dim embeddings
    'ICD9_DGNS_CD_1': (50000, 32),      # 50000 unique diagnosis codes, 32-dim embeddings
    'ICD9_PRCDR_CD_1': (10000, 16),     # 10000 unique procedure codes, 16-dim embeddings
}

# Define numerical features with normalization stats
numerical_config = {
    'num_features': 5,
    'normalization_stats': {
        'mean': [15000.0, 5.0, 12000.0, 500.0, 200.0],
        'std': [5000.0, 2.0, 4000.0, 200.0, 100.0]
    }
}

# Create encoder
    encoder = Encoder(
        categorical_config=categorical_config,
        numerical_config=numerical_config,
        hidden_dims=[128, 64],
        output_dim=64,
        dropout=0.1
    )
```

### Example 3: Forward Pass with Dataset

```python
from torch.utils.data import DataLoader
from harp.data.class_dataset import HARPDataset
from harp.models.encoder import create_encoder_from_dataset
import pandas as pd

# Load dataset
dataset = HARPDataset(dataset_dir="harp/data/raw/harp_dataset_encoded", mode="train")

# Create encoder from training data
df_train = pd.read_csv("harp/data/raw/harp_dataset_encoded/train.csv")
encoder = create_encoder_from_dataset(df_train)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Forward pass
for batch in dataloader:
    categorical_features = batch['categorical']  # Dict of tensors
    numerical_features = batch['numerical']       # Tensor or None
    
    # Forward through encoder
    encoded = encoder(categorical_features, numerical_features)
    # Output: tensor of shape (batch_size, 64) - encoded representation for each claim
    
    # Use encoded representation for downstream tasks
    # (e.g., risk prediction, policy network)
```

### Example 4: Manual Forward Pass

```python
import torch

# Prepare input data
batch_size = 32

# Categorical features (as integer indices)
categorical_features = {
    'CLM_DRG_CD': torch.randint(0, 1000, (batch_size,)),      # (32,)
    'ICD9_DGNS_CD_1': torch.randint(0, 50000, (batch_size,)), # (32,)
}

# Numerical features (as floats)
numerical_features = torch.randn(batch_size, 5)  # (32, 5)

# Forward pass
encoded = encoder(categorical_features, numerical_features)
# Output: tensor of shape (32, 64) - encoded representation for each claim
```

---

## Key Design Decisions

### Why Embeddings for Categorical Features?
- **Sparsity**: One-hot encoding would create huge vectors (50000 dimensions for 50000 codes)
- **Efficiency**: Embeddings compress to dense vectors (e.g., 32 dimensions)
- **Expressiveness**: Similar codes can learn similar representations
- **Learnability**: The network learns meaningful relationships between codes

### Why Normalize Numerical Features?
- **Scale Differences**: Payment amounts (15000) vs day counts (5) are on very different scales
- **Training Stability**: Neural networks learn better when features are on similar scales
- **Gradient Flow**: Prevents some features from dominating the gradient updates

### Why Feedforward Network?
- **Non-linearity**: Learns complex patterns that linear combinations cannot capture
- **Dimensionality Reduction**: Progressively compresses information (112 → 64)
- **Feature Learning**: Creates rich, compressed representations

### Why ReLU and Dropout?
- **ReLU**: Simple non-linear activation that enables learning complex patterns
- **Dropout**: Prevents overfitting by randomly zeroing neurons during training

---

## Summary

The encoder transforms raw preprocessed claim data into a unified, dense representation:

1. **Categorical codes** → **Embeddings** → **Concatenated**
2. **Numerical amounts** → **Normalized** → **Projected**
3. **Combined** → **Feedforward Network** → **Encoded Representation (64-dim)**

This encoded representation captures all the important information from the claim in a compact, learnable format that can be used by downstream components for risk prediction and routing decisions.


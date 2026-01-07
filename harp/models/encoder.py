"""
Encoder Network for HARP (Human-Aware Routing Policies)

This module implements the encoder component that processes input features x
and produces an encoded representation used by both the risk prediction network
(f_networks) and the policy network.

The encoder handles:
- Categorical features (IDs, codes) using embedding layers
- Numerical features (amounts, dates, counts) with normalization
- Feature combination into unified representation

Data Flow:
    Input Features (x)
        ├── Categorical Features (IDs, codes, NPIs)
        │   └──> Embedding Layers (convert indices to dense vectors)
        │       └──> Concatenate all embeddings
        │
        └── Numerical Features (amounts, dates, counts)
            └──> Normalize (mean/std standardization)
                └──> Linear Projection (match embedding dimension)
    
    Combined Features
        └──> Multi-layer Feedforward Network
            ├── Layer 1: Linear -> BatchNorm -> ReLU -> Dropout
            ├── Layer 2: Linear -> BatchNorm -> ReLU -> Dropout
            ├── ...
            └── Final Layer: Linear (no activation)
    
    Output: Encoded Representation (fixed-size vector)
        └──> Used by f_networks (risk prediction) and policy network (routing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class Encoder(nn.Module):
    """
    Encoder network that processes mixed-type input features.
    
    This encoder is the first component of the HARP system. It takes raw input features
    (both categorical like ICD9 codes and numerical like payment amounts) and converts
    them into a unified, dense representation that can be used by downstream components.
    
    Architecture Overview:
        1. Categorical features -> Embedding layers -> Concatenated embeddings
        2. Numerical features -> Normalization -> Linear projection
        3. Combined features -> Multi-layer feedforward network -> Encoded representation
    
    Args:
        categorical_config: Dict mapping feature names to (vocab_size, embedding_dim)
            Example: {'ICD9_DGNS_CD_1': (1000, 32), 'CLM_DRG_CD': (500, 16)}
            - vocab_size: Number of unique values this categorical feature can take
            - embedding_dim: Dimension of the embedding vector for this feature
        
        numerical_config: Dict with:
            - 'num_features': Number of numerical features (e.g., 5 for 5 different amounts)
            - 'normalization_stats': Optional dict with 'mean' and 'std' lists for normalization
                Example: {'mean': [1000.0, 5.0, ...], 'std': [500.0, 2.0, ...]}
        
        hidden_dims: List of hidden layer dimensions [default: [256, 128, 64]]
            Each number represents the width of one hidden layer. The network progressively
            reduces dimensionality: 256 -> 128 -> 64 -> output_dim
        
        output_dim: Dimension of final encoded representation [default: 64]
            This is the size of the vector that will be passed to f_networks and policy network
        
        dropout: Dropout probability [default: 0.1]
            Probability of randomly setting neurons to 0 during training (prevents overfitting)
    """
    
    def __init__(
        self,
        categorical_config: Dict[str, Tuple[int, int]],
        numerical_config: Dict,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        super(Encoder, self).__init__()
        
        # Store configuration for reference
        self.categorical_config = categorical_config
        self.numerical_config = numerical_config
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # ========================================================================
        # STEP 1: Build Embedding Layers for Categorical Features
        # ========================================================================
        # Embeddings convert categorical indices (like ICD9 codes) into dense vectors
        # that can capture semantic relationships. For example, similar diagnosis codes
        # will have similar embedding vectors after training.
        
        self.embeddings = nn.ModuleDict()  # Dictionary to store multiple embedding layers
        total_embedding_dim = 0  # Track total dimension after concatenating all embeddings
        
        # Create one embedding layer for each categorical feature
        for feature_name, (vocab_size, embedding_dim) in categorical_config.items():
            # vocab_size: How many unique values this feature can have
            # embedding_dim: Size of the vector representation for each value
            self.embeddings[feature_name] = nn.Embedding(
                num_embeddings=vocab_size,      # Number of unique categories
                embedding_dim=embedding_dim,     # Dimension of embedding vector
                padding_idx=0                    # Index 0 represents padding/missing values
            )
            # Accumulate total dimension: we'll concatenate all embeddings later
            total_embedding_dim += embedding_dim
        
        # ========================================================================
        # STEP 2: Build Linear Projection for Numerical Features
        # ========================================================================
        # Numerical features (like payment amounts) need to be:
        # 1. Normalized (scaled to similar ranges)
        # 2. Projected to a dimension that matches the embedding space
        
        num_numerical = numerical_config.get('num_features', 0)
        if num_numerical > 0:
            # Determine projection dimension: match roughly half of embedding space
            # This ensures numerical features have appropriate weight in the combined representation
            if total_embedding_dim > 0:
                # If we have embeddings, use half their dimension (but cap at 64)
                numerical_proj_dim = min(64, total_embedding_dim // 2)
            else:
                # If no embeddings, use a default dimension
                numerical_proj_dim = 32
            
            # Linear layer: projects num_numerical features -> numerical_proj_dim
            # This converts raw numerical values into a dense representation
            self.numerical_projection = nn.Linear(num_numerical, numerical_proj_dim)
            total_embedding_dim += numerical_proj_dim  # Add to total for feedforward input
        else:
            self.numerical_projection = None  # No numerical features to process
        
        # ========================================================================
        # STEP 3: Set Up Normalization Statistics for Numerical Features
        # ========================================================================
        # Normalization (z-score: (x - mean) / std) ensures numerical features are on
        # similar scales, which helps the network learn more effectively.
        
        # Register as buffers (not trainable parameters, but part of model state)
        self.register_buffer('numerical_mean', None)
        self.register_buffer('numerical_std', None)
        
        if 'normalization_stats' in numerical_config:
            # Convert normalization statistics to tensors
            mean = torch.tensor(numerical_config['normalization_stats']['mean'], dtype=torch.float32)
            std = torch.tensor(numerical_config['normalization_stats']['std'], dtype=torch.float32)
            
            # Avoid division by zero: clamp std to minimum value
            std = torch.clamp(std, min=1e-8)
            
            # Store for use during forward pass
            self.numerical_mean = mean
            self.numerical_std = std
        
        # ========================================================================
        # STEP 4: Build Feedforward Network
        # ========================================================================
        # The feedforward network takes the combined categorical + numerical features
        # and learns complex patterns to produce the final encoded representation.
        
        layers = []  # List to collect all layers
        input_dim = total_embedding_dim  # Start with combined feature dimension
        
        # Build each hidden layer
        for hidden_dim in hidden_dims:
            # Each hidden layer consists of:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),  # Linear transformation: input_dim -> hidden_dim
                nn.BatchNorm1d(hidden_dim),        # Normalize activations (speeds up training)
                nn.ReLU(),                         # Non-linear activation (enables learning complex patterns)
                nn.Dropout(dropout)                # Randomly zero some neurons (prevents overfitting)
            ])
            # Update input dimension for next layer
            input_dim = hidden_dim
        
        # Final output layer: no activation, just linear projection to output_dim
        # This produces the final encoded representation
        layers.append(nn.Linear(input_dim, output_dim))
        
        # Combine all layers into a sequential module
        self.feedforward = nn.Sequential(*layers)
        
    def forward(
        self,
        categorical_features: Dict[str, torch.Tensor],
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        This method processes input features through the following steps:
        1. Convert categorical indices to embeddings
        2. Normalize and project numerical features
        3. Concatenate all features
        4. Pass through feedforward network
        5. Return encoded representation
        
        Args:
            categorical_features: Dict mapping feature names to LongTensor indices
                Example: {
                    'ICD9_DGNS_CD_1': tensor([123, 456, 789]),  # Shape: (batch_size,)
                    'CLM_DRG_CD': tensor([10, 20, 30])
                }
                Each tensor contains integer indices representing categorical values
        
            numerical_features: FloatTensor of shape (batch_size, num_numerical_features)
                Example: tensor([[1000.0, 5.0, 20000.0],  # Claim 1: payment, days, amount
                                 [2000.0, 3.0, 15000.0]]) # Claim 2: payment, days, amount
        
        Returns:
            encoded: FloatTensor of shape (batch_size, output_dim)
                The final encoded representation that will be used by:
                - f_networks.py: to predict risk/entropy
                - policy network: to make routing decisions
        """
        # ========================================================================
        # STEP 1: Process Categorical Features -> Embeddings
        # ========================================================================
        # For each categorical feature, look up its embedding vector.
        # Embeddings convert sparse categorical indices into dense vectors that
        # can capture relationships (e.g., similar diagnosis codes have similar vectors).
        
        embedding_list = []  # Store embeddings for all categorical features
        
        # Process each categorical feature
        for feature_name, indices in categorical_features.items():
            # Check if we have an embedding layer for this feature
            if feature_name in self.embeddings:
                # Look up embedding: indices (batch_size,) -> embedding (batch_size, embedding_dim)
                # This is like a lookup table: index 123 -> vector [0.1, -0.3, 0.7, ...]
                embedding = self.embeddings[feature_name](indices)
                embedding_list.append(embedding)
        
        # ========================================================================
        # STEP 2: Concatenate All Categorical Embeddings
        # ========================================================================
        # Combine all categorical embeddings into one large vector.
        # Example: If we have 3 features with dims [32, 16, 8], result is (batch_size, 56)
        
        if embedding_list:
            # Concatenate along the last dimension (feature dimension)
            # Result: (batch_size, sum of all embedding_dims)
            categorical_encoded = torch.cat(embedding_list, dim=-1)
        else:
            # Edge case: no categorical features provided
            # Create an empty tensor with correct batch size
            if numerical_features is not None:
                batch_size = numerical_features.size(0)
                device = numerical_features.device
            else:
                # Get batch size from first categorical feature
                first_feature = list(categorical_features.keys())[0]
                batch_size = categorical_features[first_feature].size(0)
                device = next(self.parameters()).device
            
            categorical_encoded = torch.zeros(batch_size, 0, device=device)
        
        # ========================================================================
        # STEP 3: Process Numerical Features
        # ========================================================================
        # Numerical features need normalization (to similar scales) and projection
        # (to match the embedding dimension space).
        
        if numerical_features is not None and self.numerical_projection is not None:
            # --- Step 3a: Normalize numerical features ---
            # Normalization formula: (x - mean) / std
            # This ensures all numerical features are on similar scales (mean=0, std=1)
            # Example: payment amounts [1000, 2000, 3000] -> [-1.0, 0.0, 1.0]
            
            if self.numerical_mean is not None and self.numerical_std is not None:
                # Z-score normalization: center around 0, scale by standard deviation
                numerical_normalized = (numerical_features - self.numerical_mean) / self.numerical_std
            else:
                # No normalization stats provided, use raw features
                numerical_normalized = numerical_features
            
            # --- Step 3b: Project to embedding space ---
            # Linear transformation: (batch_size, num_numerical) -> (batch_size, numerical_proj_dim)
            # This converts raw numerical values into a dense representation that
            # can be combined with categorical embeddings
            numerical_encoded = self.numerical_projection(numerical_normalized)
            
            # ========================================================================
            # STEP 4: Combine Categorical and Numerical Features
            # ========================================================================
            # Concatenate categorical embeddings and numerical projections
            # Result: (batch_size, total_embedding_dim)
            # Example: categorical (batch, 56) + numerical (batch, 32) = (batch, 88)
            combined = torch.cat([categorical_encoded, numerical_encoded], dim=-1)
        else:
            # No numerical features, use only categorical
            combined = categorical_encoded
        
        # ========================================================================
        # STEP 5: Pass Through Feedforward Network
        # ========================================================================
        # The feedforward network learns complex patterns from the combined features
        # and produces the final encoded representation.
        # 
        # Flow through feedforward:
        #   combined (batch, total_dim) 
        #   -> Linear -> BatchNorm -> ReLU -> Dropout (batch, 256)
        #   -> Linear -> BatchNorm -> ReLU -> Dropout (batch, 128)
        #   -> Linear -> BatchNorm -> ReLU -> Dropout (batch, 64)
        #   -> Linear (batch, output_dim)
        
        encoded = self.feedforward(combined)
        
        # ========================================================================
        # STEP 6: Return Encoded Representation
        # ========================================================================
        # The encoded representation is now ready to be used by:
        # - f_networks.py: to predict risk (BCE loss) and entropy (uncertainty)
        # - policy network: to make routing decisions (which reviewer stage)
        
        return encoded
    
    def get_output_dim(self) -> int:
        """
        Return the output dimension of the encoder.
        
        This is useful when building downstream components (like f_networks or
        policy network) that need to know the size of the encoded representation.
        
        Returns:
            output_dim: The dimension of the encoded representation vector
                Example: 64 means each input produces a 64-dimensional vector
        """
        return self.output_dim


def create_encoder_from_config(config: Dict) -> Encoder:
    """
    Factory function to create an encoder from a configuration dictionary.
    
    This is a convenience function that allows you to create an encoder by passing
    a single configuration dictionary instead of individual parameters.
    
    Usage Example:
        config = {
            'categorical_config': {'ICD9_CD': (1000, 32)},
            'numerical_config': {'num_features': 5},
            'hidden_dims': [256, 128],
            'output_dim': 64,
            'dropout': 0.1
        }
        encoder = create_encoder_from_config(config)
    
    Args:
        config: Dictionary containing encoder configuration:
            - categorical_config: Dict[str, Tuple[int, int]]
                Required. Maps feature names to (vocab_size, embedding_dim)
            
            - numerical_config: Dict with:
                - 'num_features': int (required)
                - 'normalization_stats': Dict with 'mean' and 'std' lists (optional)
            
            - hidden_dims: List[int] (optional, default: [256, 128, 64])
                Hidden layer dimensions for feedforward network
            
            - output_dim: int (optional, default: 64)
                Dimension of final encoded representation
            
            - dropout: float (optional, default: 0.1)
                Dropout probability for regularization
    
    Returns:
        Encoder instance ready to use
    """
    return Encoder(
        categorical_config=config['categorical_config'],
        numerical_config=config['numerical_config'],
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),  # Default if not provided
        output_dim=config.get('output_dim', 64),                # Default if not provided
        dropout=config.get('dropout', 0.1)                       # Default if not provided
    )


def get_default_encoder_config(
    categorical_vocab_sizes: Dict[str, int],
    num_numerical_features: int,
    normalization_stats: Optional[Dict[str, List[float]]] = None
) -> Dict:
    """
    Generate a default encoder configuration based on feature information.
    
    This helper function automatically creates a reasonable encoder configuration
    based on your data. It:
    1. Automatically determines embedding dimensions based on vocabulary sizes
    2. Sets up numerical feature configuration
    3. Uses sensible defaults for network architecture
    
    Usage Example:
        vocab_sizes = {
            'ICD9_DGNS_CD_1': 5000,   # 5000 unique diagnosis codes
            'CLM_DRG_CD': 500,         # 500 unique DRG codes
            'AT_PHYSN_NPI': 10000      # 10000 unique physician IDs
        }
        num_numerical = 5  # 5 numerical features (amounts, dates, etc.)
        stats = {
            'mean': [1000.0, 5.0, 20000.0, 0.0, 0.0],
            'std': [500.0, 2.0, 10000.0, 1.0, 1.0]
        }
        config = get_default_encoder_config(vocab_sizes, num_numerical, stats)
        encoder = create_encoder_from_config(config)
    
    Args:
        categorical_vocab_sizes: Dict mapping categorical feature names to vocab sizes
            Example: {
                'ICD9_DGNS_CD_1': 5000,    # 5000 unique diagnosis codes
                'CLM_DRG_CD': 500,          # 500 unique DRG codes
                'AT_PHYSN_NPI': 10000       # 10000 unique physician NPIs
            }
            The function will automatically choose appropriate embedding dimensions
            based on these vocabulary sizes.
        
        num_numerical_features: Number of numerical features
            Example: 5 (for payment amount, utilization days, deductible, etc.)
        
        normalization_stats: Optional dict with 'mean' and 'std' lists for normalization
            Example: {
                'mean': [1000.0, 5.0, 20000.0, 0.0, 0.0],  # Mean of each numerical feature
                'std': [500.0, 2.0, 10000.0, 1.0, 1.0]      # Std dev of each numerical feature
            }
            If provided, numerical features will be normalized using these statistics.
            If None, numerical features will be used as-is (not recommended).
    
    Returns:
        Configuration dictionary that can be passed to create_encoder_from_config()
    """
    # ========================================================================
    # Helper Function: Determine Embedding Dimension Based on Vocabulary Size
    # ========================================================================
    # Larger vocabularies need larger embeddings to capture more relationships.
    # This function uses a heuristic:
    # - Small vocab (< 100): 8 dimensions (simple features)
    # - Medium vocab (< 1000): 16 dimensions (moderate complexity)
    # - Large vocab (< 10000): 32 dimensions (high complexity)
    # - Very large vocab (>= 10000): 64 dimensions (very high complexity)
    
    def get_embedding_dim(vocab_size: int) -> int:
        """
        Determine appropriate embedding dimension based on vocabulary size.
        
        Rationale:
        - Small vocabularies don't need many dimensions (fewer relationships to learn)
        - Large vocabularies need more dimensions (more complex relationships)
        - This balances model capacity with computational efficiency
        """
        if vocab_size < 100:
            return 8   # Small: simple categorical features
        elif vocab_size < 1000:
            return 16  # Medium: moderate complexity
        elif vocab_size < 10000:
            return 32  # Large: high complexity
        else:
            return 64  # Very large: maximum complexity
    
    # ========================================================================
    # Build Categorical Configuration
    # ========================================================================
    # For each categorical feature, create a (vocab_size, embedding_dim) tuple
    # The embedding dimension is automatically determined based on vocab size.
    
    categorical_config = {
        name: (vocab_size, get_embedding_dim(vocab_size))
        for name, vocab_size in categorical_vocab_sizes.items()
    }
    
    # ========================================================================
    # Build Numerical Configuration
    # ========================================================================
    # Set up numerical feature configuration with optional normalization stats.
    
    numerical_config = {
        'num_features': num_numerical_features  # Number of numerical features
    }
    
    # Add normalization statistics if provided
    if normalization_stats:
        numerical_config['normalization_stats'] = normalization_stats
    
    # ========================================================================
    # Return Complete Configuration
    # ========================================================================
    # Return a configuration dictionary with sensible defaults:
    # - Hidden dimensions: [256, 128, 64] (progressive dimensionality reduction)
    # - Output dimension: 64 (standard size for encoded representations)
    # - Dropout: 0.1 (10% dropout for regularization)
    
    return {
        'categorical_config': categorical_config,
        'numerical_config': numerical_config,
        'hidden_dims': [256, 128, 64],  # Progressive reduction: 256 -> 128 -> 64
        'output_dim': 64,               # Final encoded representation size
        'dropout': 0.1                  # 10% dropout for regularization
    }


"""
Simple Encoder for HARP

Takes categorical codes (ICD9, DRG) and numerical features (amounts, counts)
and converts them into a dense representation.

Input:
    - Categorical: Integer codes -> Embeddings -> Concatenate
    - Numerical: Floats -> Normalize -> Concatenate
    - Combined -> Linear -> ReLU -> Linear -> Output
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class Encoder(nn.Module):
    """
    Simple encoder: embeddings for categorical, normalization for numerical, then feedforward.
    
    Args:
        categorical_config: Dict of {feature_name: (vocab_size, embedding_dim)}
        numerical_config: Dict with 'num_features' and optional 'normalization_stats'
        hidden_dim: Hidden layer size [default: 64]
        output_dim: Output size [default: 64]
    """
    
    def __init__(
        self,
        categorical_config: Dict[str, Tuple[int, int]],
        numerical_config: Dict,
        hidden_dim: int = 64,
        output_dim: int = 64
    ):
        super(Encoder, self).__init__()
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feature_name, (vocab_size, embedding_dim) in categorical_config.items():
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            total_embedding_dim += embedding_dim
        
        # Normalization for numerical features
        num_numerical = numerical_config.get('num_features', 0)
        self.numerical_dim = num_numerical
        
        if 'normalization_stats' in numerical_config:
            mean = torch.tensor(numerical_config['normalization_stats']['mean'], dtype=torch.float32)
            std = torch.tensor(numerical_config['normalization_stats']['std'], dtype=torch.float32)
            std = torch.clamp(std, min=1e-8)
            self.register_buffer('numerical_mean', mean)
            self.register_buffer('numerical_std', std)
        else:
            self.register_buffer('numerical_mean', None)
            self.register_buffer('numerical_std', None)
        
        # Simple feedforward: Linear -> ReLU -> Linear
        input_dim = total_embedding_dim + num_numerical
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        categorical_features: Dict[str, torch.Tensor],
        numerical_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass: embeddings -> normalize -> concatenate -> feedforward"""
        
        # Get categorical embeddings
        embedding_list = []
        for feature_name, indices in categorical_features.items():
            if feature_name in self.embeddings:
                embedding_list.append(self.embeddings[feature_name](indices))
        
        if embedding_list:
            categorical_encoded = torch.cat(embedding_list, dim=-1)
        else:
            batch_size = next(iter(categorical_features.values())).size(0) if categorical_features else (
                numerical_features.size(0) if numerical_features is not None else 1
            )
            device = next(self.parameters()).device
            categorical_encoded = torch.zeros(batch_size, 0, device=device)
        
        # Normalize numerical features
        if numerical_features is not None and self.numerical_dim > 0:
            if self.numerical_mean is not None and self.numerical_std is not None:
                numerical_normalized = (numerical_features - self.numerical_mean) / self.numerical_std
            else:
                numerical_normalized = numerical_features
            combined = torch.cat([categorical_encoded, numerical_normalized], dim=-1)
        else:
            combined = categorical_encoded
        
        # Feedforward
        return self.feedforward(combined)
    
    def get_output_dim(self) -> int:
        """Return output dimension"""
        return self.feedforward[-1].out_features


def create_encoder_from_dataset(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    hidden_dim: int = 64,
    output_dim: int = 64
) -> Encoder:
    """
    Create encoder from dataset.
    
    Auto-detects categorical (DRG/DGNS/PRCDR) and numerical (AMT/CNT/LBLTY) columns.
    """
    exclude_columns = ['claim_status', 'reviewer']
    df_analysis = df.drop(columns=[col for col in exclude_columns if col in df.columns])
    
    # Auto-detect columns
    if categorical_columns is None:
        categorical_columns = [
            col for col in df_analysis.columns
            if any(kw in col for kw in ['DRG', 'DGNS', 'PRCDR'])
        ]
    
    if numerical_columns is None:
        numerical_columns = [
            col for col in df_analysis.columns
            if col not in categorical_columns and
            any(kw in col for kw in ['AMT', 'CNT', 'LBLTY'])
        ]
    
    # Get vocab sizes
    categorical_config = {}
    for col in categorical_columns:
        if col in df_analysis.columns:
            max_val = int(df_analysis[col].max()) if len(df_analysis[col]) > 0 else 0
            vocab_size = max(max_val + 1, 10)
            # Simple embedding dim based on vocab size
            embedding_dim = 16 if vocab_size < 1000 else 32 if vocab_size < 10000 else 64
            categorical_config[col] = (vocab_size, embedding_dim)
    
    # Get normalization stats
    normalization_stats = None
    if numerical_columns:
        means = [float(df_analysis[col].mean()) for col in numerical_columns if col in df_analysis.columns]
        stds = [max(float(df_analysis[col].std()), 1e-8) for col in numerical_columns if col in df_analysis.columns]
        if means and stds:
            normalization_stats = {'mean': means, 'std': stds}
    
    numerical_config = {
        'num_features': len(numerical_columns),
        'normalization_stats': normalization_stats
    }
    
    return Encoder(
        categorical_config=categorical_config,
        numerical_config=numerical_config,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )

from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import torch
import numpy as np
import ast


class HARPDataset(Dataset):
    """Simple dataset for HARP. Provides x (features), d (decisions), c (costs)."""
    
    def __init__(self, dataset_dir="harp/data/raw/harp_dataset_encoded", mode="train", costs=None):
        """
        Args:
            dataset_dir: Directory with CSV files
            mode: "train", "val", or "test"
            costs: List/tensor of costs per reviewer. Default: [0.0, 1.0, 2.0, ...]
        """
        # Find CSV file
        files = glob.glob(os.path.join(dataset_dir, "*.csv"))
        if not files:
            files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        
        mode_file = None
        for f in files:
            if mode in os.path.basename(f).lower():
                mode_file = f
                break
        
        if mode_file is None:
            raise ValueError(f"Mode '{mode}' not found in {dataset_dir}")
        
        self.df = pd.read_csv(mode_file)
        
        # Get categorical and numerical columns
        self.categorical_columns = [
            col for col in self.df.columns
            if any(kw in col for kw in ['DRG', 'DGNS', 'PRCDR'])
            and col not in ['claim_status', 'reviewer', 'reviewer_correct']
        ]
        
        self.numerical_columns = [
            col for col in self.df.columns
            if col not in self.categorical_columns
            and col not in ['claim_status', 'reviewer', 'reviewer_correct']
            and any(kw in col for kw in ['AMT', 'CNT', 'LBLTY'])
        ]
        
        # Store data
        self.categorical_data = {
            col: self.df[col].fillna(-1).astype(np.int64).values
            for col in self.categorical_columns
        }
        
        if self.numerical_columns:
            self.numerical_data = self.df[self.numerical_columns].fillna(0.0).astype(np.float32).values
        else:
            self.numerical_data = None
        
        # Get reviewer_correct (d)
        if 'reviewer_correct' in self.df.columns:
            self.d = []
            for val in self.df['reviewer_correct']:
                if isinstance(val, str):
                    parsed = ast.literal_eval(val)
                    self.d.append([bool(x) for x in parsed])
                elif isinstance(val, (list, np.ndarray)):
                    self.d.append([bool(x) for x in val])
                else:
                    self.d.append([bool(val)])
            self.d = np.array(self.d, dtype=object)
            self.n_reviewers = len(self.d[0]) if len(self.d) > 0 else 0
        else:
            self.d = None
            self.n_reviewers = 0
        
        # Set costs (c)
        if costs is None:
            self.c = torch.tensor([float(i) for i in range(self.n_reviewers)], dtype=torch.float32) if self.n_reviewers > 0 else torch.tensor([0.0], dtype=torch.float32)
        else:
            self.c = torch.tensor(costs, dtype=torch.float32) if isinstance(costs, (list, np.ndarray)) else costs
        
        if self.n_reviewers > 0 and len(self.c) != self.n_reviewers:
            raise ValueError(f"Costs length ({len(self.c)}) must match reviewers ({self.n_reviewers})")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Returns dict with 'x' (features), 'd' (decisions), 'c' (costs)."""
        # x: categorical and numerical features
        x = {
            'categorical': {
                col: torch.tensor(self.categorical_data[col][idx], dtype=torch.long)
                for col in self.categorical_columns
            },
            'numerical': torch.tensor(self.numerical_data[idx], dtype=torch.float32) if self.numerical_data is not None else None
        }
        
        # d: reviewer decisions
        d = torch.tensor([bool(x) for x in self.d[idx]], dtype=torch.float32) if self.d is not None else None
        
        return {
            'x': x,
            'd': d,
            'c': self.c  # Same for all samples
        }

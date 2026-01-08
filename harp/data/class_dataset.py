from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import torch
import numpy as np


class HARPDataset(Dataset):
    """
    Dataset class for HARP that loads preprocessed data and prepares it for the encoder.
    
    After preprocessing (process.py, encodings.py), the data contains:
    - Categorical features: CLM_DRG_CD, ADMTNG_ICD9_DGNS_CD, ICD9_DGNS_CD_1-10, ICD9_PRCDR_CD_1-6
    - Numerical features: CLM_PMT_AMT, CLM_UTLZTN_DAY_CNT, NCH_PRMRY_PYR_CLM_PD_AMT,
                          NCH_BENE_IP_DDCTBL_AMT, NCH_BENE_PTA_COINSRNC_LBLTY_AM
    - Labels: claim_status (0/1), reviewer (1/2/3)
    
    This dataset class automatically separates categorical and numerical features
    and prepares them in the format expected by the encoder.
    """

    def __init__(self, dataset_dir="harp/data/raw/harp_dataset_encoded", mode="train"):
        """
        Initialize the dataset.
        
        Args:
            dataset_dir: Directory containing encoded CSV files
            mode: One of "train", "val", or "test"
        """
        # Fix file path: look for CSV files in the directory
        files = glob.glob(os.path.join(dataset_dir, "*.csv"))
        if not files:
            # Try alternative path pattern
            files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        
        # Find the file matching the mode
        mode_file = None
        for f in files:
            if mode in os.path.basename(f).lower():
                mode_file = f
                break
        
        if mode_file is None:
            raise ValueError(f"Mode '{mode}' not found in {dataset_dir}. Available files: {[os.path.basename(f) for f in files]}")
        
        self.df = pd.read_csv(mode_file)
        self.load_attributes()

    def load_attributes(self):
        """
        Load and separate features into categorical and numerical.
        
        Based on preprocessing pipeline:
        - Categorical: Columns containing 'DRG', 'DGNS', or 'PRCDR' (encoded as integers)
        - Numerical: Columns containing 'AMT', 'CNT', or 'LBLTY' (floats)
        - Labels: 'claim_status' (0/1), 'reviewer' (1/2/3)
        """
        # Identify categorical columns (ICD9 codes, DRG codes)
        self.categorical_columns = [
            col for col in self.df.columns
            if any(keyword in col for keyword in ['DRG', 'DGNS', 'PRCDR'])
            and col not in ['claim_status', 'reviewer']
        ]
        
        # Identify numerical columns (amounts, counts)
        self.numerical_columns = [
            col for col in self.df.columns
            if col not in self.categorical_columns
            and col not in ['claim_status', 'reviewer']
            and any(keyword in col for keyword in ['AMT', 'CNT', 'LBLTY'])
        ]
        
        # Extract categorical features (convert to numpy, then will convert to tensors)
        self.categorical_data = {}
        for col in self.categorical_columns:
            # Convert to int64 (required for embedding layers)
            # Handle any NaN values by filling with -1 (missing sentinel)
            self.categorical_data[col] = self.df[col].fillna(-1).astype(np.int64).values
        
        # Extract numerical features
        if self.numerical_columns:
            self.numerical_data = self.df[self.numerical_columns].fillna(0.0).astype(np.float32).values
        else:
            self.numerical_data = None
        
        # Extract labels
        self.claim_status = self.df['claim_status'].values if 'claim_status' in self.df.columns else None
        self.reviewer = self.df['reviewer'].values if 'reviewer' in self.df.columns else None
        
        # Store total number of samples
        self.n_samples = len(self.df)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns a dictionary with:
        - 'categorical': Dict of categorical features as LongTensors (for embeddings)
        - 'numerical': FloatTensor of numerical features (or None)
        - 'claim_status': Label (0 or 1) if available
        - 'reviewer': Reviewer level (1, 2, or 3) if available
        
        Note: DataLoader will automatically batch these correctly.
        For categorical dict, it batches each tensor separately.
        For numerical tensor, it stacks along first dimension.
        """
        # Prepare categorical features as dictionary of tensors
        # Shape: (1,) for each feature - DataLoader will stack these to (batch_size,)
        categorical = {
            col: torch.tensor(self.categorical_data[col][idx], dtype=torch.long)
            for col in self.categorical_columns
        }
        
        # Prepare numerical features as tensor
        # Shape: (num_numerical_features,) - DataLoader will stack to (batch_size, num_numerical_features)
        if self.numerical_data is not None:
            numerical = torch.tensor(self.numerical_data[idx], dtype=torch.float32)
        else:
            numerical = None
        
        # Prepare labels
        result = {
            'categorical': categorical,
            'numerical': numerical
        }
        
        if self.claim_status is not None:
            result['claim_status'] = torch.tensor(self.claim_status[idx], dtype=torch.long)
        
        if self.reviewer is not None:
            result['reviewer'] = torch.tensor(self.reviewer[idx], dtype=torch.long)
        
        return result
    
    def get_categorical_columns(self):
        """Return list of categorical column names."""
        return self.categorical_columns.copy()
    
    def get_numerical_columns(self):
        """Return list of numerical column names."""
        return self.numerical_columns.copy()
    
    def get_info(self):
        """Return dataset information."""
        info = {
            'num_samples': self.n_samples,
            'categorical_features': len(self.categorical_columns),
            'numerical_features': len(self.numerical_columns),
            'has_claim_status': self.claim_status is not None,
            'has_reviewer': self.reviewer is not None
        }
        if self.claim_status is not None:
            info['claim_status_distribution'] = {
                'accepted': int((self.claim_status == 1).sum()),
                'rejected': int((self.claim_status == 0).sum())
            }
        if self.reviewer is not None:
            unique_reviewers, counts = np.unique(self.reviewer, return_counts=True)
            info['reviewer_distribution'] = {
                int(r): int(c) for r, c in zip(unique_reviewers, counts)
            }
        return info
          





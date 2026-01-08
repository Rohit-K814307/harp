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
        files = glob.glob(os.path.join(dataset_dir, "/*.csv"))
        try:
            mode_dir = next(fname for fname in files if mode in fname)
        except:
            raise ValueError(f"Mode {mode} not found in {dataset_dir}")
        
        self.df = pd.read_csv(mode_dir)
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
        """
        # Prepare categorical features as dictionary of tensors
        categorical = {
            col: torch.LongTensor([self.categorical_data[col][idx]])
            for col in self.categorical_columns
        }
        
        # Prepare numerical features as tensor
        if self.numerical_data is not None:
            numerical = torch.FloatTensor([self.numerical_data[idx]])
        else:
            numerical = None
        
        # Prepare labels
        result = {
            'categorical': categorical,
            'numerical': numerical
        }
        
        if self.claim_status is not None:
            result['claim_status'] = torch.LongTensor([self.claim_status[idx]])
        
        if self.reviewer is not None:
            result['reviewer'] = torch.LongTensor([self.reviewer[idx]])
        
        return result
    
    def get_categorical_columns(self):
        """Return list of categorical column names."""
        return self.categorical_columns
    
    def get_numerical_columns(self):
        """Return list of numerical column names."""
        return self.numerical_columns
          





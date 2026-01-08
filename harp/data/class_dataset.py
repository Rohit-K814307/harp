from torch.utils.data import Dataset
import glob
import os
import pandas as pd


class HARPDataset(Dataset):

     def __init__(self, dataset_dir="harp/data/raw/harp_dataset_encoded", mode="train"):
          
          files = glob.glob(os.path.join(dataset_dir, "/*.csv"))
          try:
               mode_dir = next(fname for fname in files if mode in fname)
          except:
               raise ValueError(f"Mode {mode} not found in {dataset_dir}")
          
          self.df = pd.read_csv(mode_dir)
          self.load_attributes()

     def load_attributes(self):
          
          #create inputs

          ## x
          x_cols = [x for x in self.df.columns if x not in ["claim_status", "reviewer"]]
          x_vals = self.df[x_cols]
          self.X = x_vals.to_numpy()

          ## d
          





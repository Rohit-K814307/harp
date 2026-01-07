from torch.utils.data import Dataset
import glob
import os
import pandas as pd


class HARPDataset(Dataset):

     def __init__(self, dataset_dir="harp/data/raw/harp_dataset", mode="train"):
          
          files = glob.glob(os.path.join(dataset_dir, "/*.csv"))
          try:
               mode_dir = next(fname for fname in files if mode in fname)
          except:
               raise ValueError(f"Mode {mode} not found in {dataset_dir}")
          
          # df = pd.read_csv(mode_dir)
          # self.load_attributes(df)

     def load_attributes(self, df):
          pass




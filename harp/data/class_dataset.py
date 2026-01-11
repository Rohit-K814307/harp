from torch.utils.data import Dataset
import pandas as pd
import ast
import torch


class HARPDataset(Dataset):
     
    def __init__(self, reviewer_costs, mode="train"):
          
        path_to_dataset = f"harp/data/raw/harp_dataset_encoded/{mode}.csv"
        df = pd.read_csv(path_to_dataset)


        # separate x, y, c, and d

        c = reviewer_costs
        d = df["reviewer_correct"].apply(ast.literal_eval).to_list()
        y_reviewer = df["reviewer"].to_numpy()
        y_claim_status = df["claim_status"].to_numpy()


        ### break x into the required variables

        df_x = df.drop(columns=["reviewer_correct", "reviewer", "claim_status"], axis=1)

        cat_cols = [x for x in list(df_x.columns) if "ICD9" in x]
        num_cols = [x for x in list(df_x.columns) if x not in cat_cols]

        dgns_cols = [x for x in cat_cols if "DGNS" in x]
        prcdr_cols = [x for x in cat_cols if x not in dgns_cols]

        df_x_cat_dgns = df_x[dgns_cols].to_numpy()
        df_x_cat_prcdr = df_x[prcdr_cols].to_numpy()
        df_x_num = df_x[num_cols].to_numpy()


        # prepare into torch tensors

        self.c = torch.tensor(c, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.float32)
        self.y_reviewer = torch.tensor(y_reviewer, dtype=torch.long)
        self.y_claim_status = torch.tensor(y_claim_status, dtype=torch.float32).unsqueeze(1)

        self.x_cat_dgns = torch.tensor(df_x_cat_dgns, dtype=torch.long)
        self.x_cat_prcdr = torch.tensor(df_x_cat_prcdr, dtype=torch.long)
        self.x_num = torch.tensor(df_x_num, dtype=torch.float32)


    def __len__(self):
        return len(self.y_reviewer)


    def __getitem__(self, idx):
        return {
            "x_cat_dgns": self.x_cat_dgns[idx],
            "x_cat_prcdr": self.x_cat_prcdr[idx],
            "x_num": self.x_num[idx],

            "y_reviewer": self.y_reviewer[idx],
            "y_claim_status": self.y_claim_status[idx],

            "d": self.d[idx],

            "c": self.c
        }




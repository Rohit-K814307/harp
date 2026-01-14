import torch
import torch.nn as nn
import sys
import os
import numpy as np
import torch.nn.functional as F

submodule_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrated_selective_classification")
if submodule_path not in sys.path:
    sys.path.append(submodule_path)

from src.models import SelectiveNet




def confidence_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    confidence, _ = probs.max(dim=1)
    return confidence

def margin_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]

def entropy_from_logits(logits, eps=1e-8):
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs + eps)).sum(dim=1)

class CSCSelector(nn.Module):

     def __init__(
          self, 
          f, # must be trained
          csc_hidden_dim,
          csc_num_layers,
          csc_dropout=0.0,
     ):
          
          super().__init__()

          self.f = f.eval()

          self.g = SelectiveNet(
               3,
               csc_hidden_dim,
               csc_num_layers,
               csc_dropout
          )
  

     def forward(self, x_dgns, x_prcdr, x_numeric):

          with torch.no_grad():
                logits = self.f(x_dgns, x_prcdr, x_numeric)
                logits = torch.cat([-logits, logits], dim=1)

          conf = confidence_from_logits(logits)
          marg = margin_from_logits(logits)
          ent  = entropy_from_logits(logits)

          features = torch.stack([conf, marg, ent], dim=1)
          return self.g(features), logits


# calculate csc thresholds given f, g, and val dataset

def calculate_csc_thresholds(
    g,
    data_loader,
    n_reviewers,
    coverages=None,  # tot coverage: automated + reviewers; [0.5, 0.8, 1.0] means 50% automated, next 30% rev1, final 20% rev 2
):

    g.eval()

    selector_scores = []

    # selector scores
    with torch.no_grad():
        for batch in data_loader:
            x = (
                batch["x_cat_dgns"],
                batch["x_cat_prcdr"],
                batch["x_num"],
            )
            # forward
            sel_scores, _ = g(*x).view(-1)  # robust to batch=1
            selector_scores.append(sel_scores.cpu().numpy())

    selector_scores = np.concatenate(selector_scores)
    scores_sorted = np.sort(selector_scores)[::-1]  # descending

    N = len(scores_sorted)

    
    # use default coverages
    if coverages is None:
        total_stages = n_reviewers + 1
        coverages = [(i + 1) / total_stages for i in range(total_stages)]

    assert len(coverages) == n_reviewers + 1
    assert all(coverages[i] < coverages[i + 1] for i in range(len(coverages) - 1))

    
    # calc thresholds from csc
    thresholds = []
    prev_cut = 0

    for cov in coverages:
        remaining = scores_sorted[prev_cut:]
        idx = int(np.ceil(cov * len(remaining))) - 1
        idx = min(idx, len(remaining) - 1)
        tau = remaining[idx]
        thresholds.append(tau)
        prev_cut += idx + 1

    return thresholds



class CSC_Router(nn.Module):

    def __init__(self, g, data_loader, n_reviewers, coverages, g_weights_path):
        super().__init__()

        g.load_state_dict(torch.load(g_weights_path, map_location='cpu'))
        g.eval()
        self.g = g

        raw_thresholds = calculate_csc_thresholds(
            g=g,
            data_loader=data_loader,
            n_reviewers=n_reviewers,
            coverages=coverages
        )
        
        full_thresholds = raw_thresholds + [-1.0] 

        self.register_buffer("thresholds", torch.tensor(full_thresholds))

    @torch.no_grad()
    def forward(self, x_dgns, x_prcdr, x_numeric):

        sel_scores, _ = self.g(x_dgns, x_prcdr, x_numeric).view(-1)
        
        scores_exp = sel_scores.unsqueeze(1)
        
        mask = scores_exp >= self.thresholds.unsqueeze(0)

        stages = mask.float().argmax(dim=1)

        return stages
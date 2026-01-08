import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def calculate_complexity(row, scores):

     score = 0

     if row['NCH_PRMRY_PYR_CLM_PD_AMT'] > 0:
          score += scores["multi_payer"]

     if row['CLM_UTLZTN_DAY_CNT'] <= 1:
          score += scores["short_stay"]

     op_phys = row['OP_PHYSN_NPI']
     if pd.notna(op_phys) and str(op_phys).upper() != "MISSING":
          score += scores["surgical"]

     if row['CLM_PMT_AMT'] > 20000:
          score += scores["high_cost"]

     return pd.Series([score])



def calculate_cutoffs(df, scores, n_reviewers, n_init, random_state, plot=False):
     
     total_scores = df.apply(lambda row: calculate_complexity(row, scores))

     data = total_scores.values.reshape(-1, 1)

     kmeans = KMeans(n_clusters=n_reviewers, random_state=random_state, n_init=n_init)
     kmeans.fit(data)

     centroids = np.sort(kmeans.cluster_centers_.flatten())

     cutoffs = [0.0]
    
     for i in range(len(centroids) - 1):
          
          # calc midpt as cutoff
          boundary = (centroids[i] + centroids[i+1]) / 2
          cutoffs.append(round(boundary, 2))

     max_score = df['reviewer_score'].max()
     cutoffs.append(float(max_score) + 1.0) 

     if plot:
          plot_cutoffs(total_scores, cutoffs)


     return cutoffs


def plot_cutoffs(scores, cutoffs):

     plt.hist(scores, bins=20, alpha=0.6, color='g')
     for c in cutoffs[1:-1]:
          plt.axvline(c, color='r', linestyle='--', label=f'Cutoff: {c}')

     plt.legend()
     plt.show()

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


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
     
     total_scores = df.apply(lambda row: calculate_complexity(row, scores), axis=1)

     data = total_scores.values.reshape(-1, 1)

     kmeans = KMeans(n_clusters=n_reviewers, random_state=random_state, n_init=n_init)
     kmeans.fit(data)

     centroids = np.sort(kmeans.cluster_centers_.flatten())

     cutoffs = [0.0]
    
     for i in range(len(centroids) - 1):
          
          # calc midpt as cutoff
          boundary = (centroids[i] + centroids[i+1]) / 2
          cutoffs.append(round(boundary, 2))

     max_score = total_scores.max()
     cutoffs.append(float(max_score.iloc[0]) + 1.0)
     cutoffs = [float(x) for x in cutoffs]

     print("Data Per-Reviewer Distribution")
     binned = pd.cut(total_scores.iloc[:, 0], bins=cutoffs, right=False, include_lowest=True)
     counts = binned.value_counts().sort_index()

     for i, (interval, count) in enumerate(counts.items()):
          print(f"Reviewer {i + 1} (Score {interval.left} - {interval.right}): {count} claims")

     if plot:
          plot_cutoffs(total_scores, cutoffs)
          print()

     return cutoffs


def plot_cutoffs(scores, cutoffs):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 3), dpi=300)

    counts, _, _ = plt.hist(scores, bins=20, alpha=0.85, color='#34495e', edgecolor='white')
    y_max = max(counts)

    for i,c in enumerate(cutoffs[1:-1]):
        plt.axvline(c, color="#697274AB", linestyle='--', linewidth=1)
        plt.text(c, y_max * 1.05, f'{round(c, 1)}\n<-R{i+1}\nR{i+2}->', 
                 color="#000000", ha='center', fontweight="bold", va='bottom', fontsize=6)

    plt.title("Complexity Score Distribution & Tier Thresholds", fontweight='bold', pad=10, fontsize=8)
    plt.xlabel("Complexity Score", fontsize=7)
    plt.ylabel("Count", fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    
    plt.ylim(top=y_max * 1.35)
    
    plt.tight_layout()
    plt.show()
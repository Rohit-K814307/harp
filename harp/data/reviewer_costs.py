import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# calculate costs based on marginal risk reduction formula:
# c_n = penalty * (E_n-1 - E_n)

def calculate_error_rate(claim_status, reviewer_correct, base_error_rate, n):

     if n == 0:
          return base_error_rate

     claim_status = np.asarray(claim_status)
     reviewer_correct = reviewer_correct[:, n-1]

     return np.mean(claim_status != reviewer_correct)

def marginal_risk_reduction(df, n, penalty, base_error_rate):
     
     E_n_next_best = lambda x: calculate_error_rate(df["claim_status"].to_numpy(), np.array(df["reviewer_correct"].to_list()), base_error_rate, x-1)
     E_n = lambda x: calculate_error_rate(df["claim_status"].to_numpy(), np.array(df["reviewer_correct"].to_list()), base_error_rate, x)

     return [penalty * (E_n_next_best(x) - E_n(x)) for x in range(1, n+1)]


#calculate costs based on scarcity:
# c_n = c_base * (count_claims_reviewer_1 / count_claims_reviewer_n)

def count_claims(reviewer, n):

     return np.sum(reviewer == n)

def scarcity(df, n, c_base):

     reviewer = df["reviewer"].to_numpy()
     claims_rev_1 = count_claims(reviewer, 1)

     return [c_base * (claims_rev_1 / count_claims(reviewer, x)) for x in range(1, n+1)]


# calc all costs in one function and return based on mode

def calculate_c(df, n, args):

     if args["cost_mode"] == "mrr":
          raw_costs = np.array(marginal_risk_reduction(df, n, args["mrr_penalty"], args["mrr_base_error_rate"]))
          return np.cumsum(raw_costs).round(decimals=3).tolist()
     else:
          return np.array(scarcity(df, n, args["scarcity_c_base"])).round(decimals=3).tolist()


def plot_c_modes(mrr_costs, scarcity_costs):
    tiers = [f"Tier {i+1}" for i in range(len(mrr_costs))]
    data = pd.DataFrame({
        'Reviewer Tier': tiers * 2,
        'Cost Value': mrr_costs + scarcity_costs,
        'Cost Mode': ['MRR'] * len(mrr_costs) + ['Scarcity'] * len(scarcity_costs)
    })

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(3.15, 3.5), dpi=300)

    ax = sns.barplot(
        data=data, 
        x='Reviewer Tier', 
        y='Cost Value', 
        hue='Cost Mode',
        palette=["#2d6195", "#3FD0F1AB"],
        edgecolor='white',
        linewidth=0.8
    )

    plt.title("Institutional Cost Comparison: MRR vs. Scarcity", 
              fontweight='bold', pad=12, fontsize=8)
    
    plt.ylabel("Modeled Institutional Cost ($c_s$)", fontsize=7)
    plt.xlabel("Reviewer Hierarchy", fontsize=7)
    
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.legend(title='Cost Type', title_fontsize='6', fontsize='6', 
               frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    sns.despine()
    plt.tight_layout()
    plt.show()
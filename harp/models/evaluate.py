import torch
import torch.nn as nn
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
import argparse

from harp.data.class_dataset import HARPDataset
from harp.models.baselines import Naive, Oracle, CSC_Router, CSCSelector
from harp.models.pi_networks import HARPRouter
from harp.models.f_networks import MinimalFNet
from harp.models.encoder import HARPEncoder
from harp.data.load_config import load_config


def plot_static_metrics(results, output_dir):
    """Generates bar charts for system accuracy and safety violation rates."""
    models = [k for k in results.keys() if k != 'oracle'] 
    
    accs = [results[m]['metrics']['System_Accuracy'] for m in models]
    ftrs = [results[m]['metrics']['False_Trust_Rate'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accs, width, label='System Accuracy', color='skyblue')
    rects2 = ax.bar(x + width/2, ftrs, width, label='False Trust Rate', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('System Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    plt.savefig(os.path.join(output_dir, "system_metrics.png"), dpi=300)
    plt.close()

def plot_rc_curves(results, output_dir):
    """Generates Risk-Coverage curves by sweeping trust thresholds."""
    plt.figure(figsize=(8, 6))
    
    for name, data in results.items():
        if name == 'oracle': continue

        scores = data['scores']
        correct = data['correct']
        
        desc_sort = np.argsort(scores)[::-1]
        sorted_correct = correct[desc_sort]
        
        n = len(scores)
        coverages = []
        risks = []
        
        steps = np.linspace(1, n, 100, dtype=int)
        
        for k in steps:
            cov = k / n
            acc_at_k = np.mean(sorted_correct[:k])
            risks.append(1.0 - acc_at_k)
            coverages.append(cov)
            
        plt.plot(coverages, risks, label=name.upper(), linewidth=2)
        
        op_cov = data['metrics']['AI_Coverage']
        op_risk = data['metrics']['AI_Risk']
        plt.scatter(op_cov, op_risk, s=50, marker='x', zorder=5)

    plt.title('Risk-Coverage Curve')
    plt.xlabel('Coverage (Automation Rate)')
    plt.ylabel('Risk (Error Rate on Automated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(bottom=0)
    
    plt.savefig(os.path.join(output_dir, "rc_curve.png"), dpi=300)
    plt.close()


def compute_metrics(route_indices, ai_trust_scores, model_preds, human_preds, targets):
    N = len(targets)
    M = human_preds.shape[1] if human_preds.ndim > 1 else 1
    
    system_preds = np.zeros_like(targets)
    
    # Route 0 is AI
    ai_mask = (route_indices == 0)
    system_preds[ai_mask] = model_preds[ai_mask]
    
    # Route 1..M are Humans
    for i in range(M):
        h_mask = (route_indices == (i + 1))
        if human_preds.ndim > 1:
            system_preds[h_mask] = human_preds[h_mask, i]
        else:
            system_preds[h_mask] = human_preds[h_mask]

    model_correct = (model_preds == targets)

    acc = accuracy_score(targets, system_preds)
    prec = precision_score(targets, system_preds, zero_division=0)
    rec = recall_score(targets, system_preds, zero_division=0)

    coverage = np.mean(ai_mask)
    
    risk = 0.0
    if coverage > 0:
        risk = 1.0 - np.mean(model_correct[ai_mask])

    false_trust_count = np.sum(ai_mask & ~model_correct)
    false_trust_rate = false_trust_count / N

    try:
        trust_auroc = roc_auc_score(model_correct, ai_trust_scores) if len(np.unique(model_correct)) > 1 else 0.5
    except ValueError:
        trust_auroc = 0.5

    return {
        "System_Accuracy": acc,
        "System_Precision": prec,
        "System_Recall": rec,
        "AI_Coverage": coverage,
        "AI_Risk": risk,
        "False_Trust_Rate": false_trust_rate,
        "Trust_AUROC": trust_auroc
    }


def evaluate_model(model_wrapper, dataloader, device, mode='harp'):
    
    if hasattr(model_wrapper, 'eval'): model_wrapper.eval()
    if hasattr(model_wrapper, 'f') and hasattr(model_wrapper.f, 'eval'): model_wrapper.f.eval()
    if hasattr(model_wrapper, 'pi') and hasattr(model_wrapper.pi, 'eval'): model_wrapper.pi.eval()

    all_routes = []
    all_trust_scores = []
    all_model_preds = []
    all_human_preds = []
    all_targets = []
    all_correctness = [] 

    print(f"Evaluating {mode.upper()}...")

    with torch.no_grad():
        for batch in dataloader:
            x_dgns = batch["x_cat_dgns"].to(device)
            x_prcdr = batch["x_cat_prcdr"].to(device)
            x_num = batch["x_num"].to(device)
            y = batch["y_claim_status"].to(device).view(-1, 1)
            
            d_humans = batch["d"].to(device) 
            c_humans = batch["c"].to(device) 

            f_logits = model_wrapper.forward_f(x_dgns, x_prcdr, x_numeric=x_num)
            f_probs = torch.sigmoid(f_logits)
            f_preds = (f_logits > 0).float()

            batch_correct = (f_preds == y).float().cpu().numpy().flatten()
            all_correctness.append(batch_correct)

            if mode == 'harp':
                routing_logits = model_wrapper.forward_pi(f_logits, x_dgns, x_prcdr, x_num)
                routes = torch.argmax(routing_logits, dim=1).cpu().numpy()
                trust_scores = torch.softmax(routing_logits, dim=1)[:, 0].cpu().numpy()

            elif mode == 'csc':
                stages = model_wrapper(x_dgns, x_prcdr, x_num)
                routes = stages.cpu().numpy()
                
                raw_out, _ = model_wrapper.router.g(x_dgns, x_prcdr, x_num)
                trust_scores = torch.sigmoid(raw_out).view(-1).cpu().numpy()

            elif mode == 'naive':
                confidence = torch.max(f_probs, 1 - f_probs).view(-1)
                
                if c_humans.dim() > 1:
                    cheapest_human_idx = torch.argmin(c_humans, dim=1) + 1
                else:
                    global_idx = torch.argmin(c_humans).item() + 1
                    cheapest_human_idx = torch.full_like(confidence, global_idx, dtype=torch.long)

                ai_mask = (confidence >= model_wrapper.threshold)
                routes = cheapest_human_idx.clone()
                routes[ai_mask] = 0
                
                routes = routes.cpu().numpy()
                trust_scores = confidence.cpu().numpy()

            elif mode == 'oracle':
                penalty = model_wrapper.error_penalty
                
                ai_wrong = (f_preds != y).float()
                cost_ai = penalty * ai_wrong 
                
                if c_humans.dim() == 1:
                    c_humans = c_humans.unsqueeze(0).expand(d_humans.size(0), -1)
                
                humans_wrong = (d_humans != y).float()
                cost_humans = c_humans + (penalty * humans_wrong)
                
                all_costs = torch.cat([cost_ai, cost_humans], dim=1)
                best_routes = torch.argmin(all_costs, dim=1)
                
                routes = best_routes.cpu().numpy()
                trust_scores = (best_routes == 0).float().cpu().numpy()

            else:
                raise ValueError(f"Unknown mode: {mode}")

            all_routes.append(routes)
            all_trust_scores.append(trust_scores)
            all_model_preds.append(f_preds.cpu().numpy().flatten())
            all_human_preds.append(d_humans.cpu().numpy())
            all_targets.append(y.cpu().numpy().flatten())

    metrics = compute_metrics(
        route_indices=np.concatenate(all_routes),
        ai_trust_scores=np.concatenate(all_trust_scores),
        model_preds=np.concatenate(all_model_preds),
        human_preds=np.concatenate(all_human_preds),
        targets=np.concatenate(all_targets)
    )
    
    return metrics, np.concatenate(all_trust_scores), np.concatenate(all_correctness)

class HARPWrapper(nn.Module):
    def __init__(self, f_net, pi_net):
        super().__init__()
        self.f = f_net
        self.pi = pi_net

    def forward_f(self, x_dgns, x_prcdr, x_numeric):
        return self.f(x_dgns, x_prcdr, x_numeric)

    def forward_pi(self, f_out, x_dgns, x_prcdr, x_numeric):
        return self.pi(f_out, x_dgns, x_prcdr, x_numeric)
    
class CSCWrapper(nn.Module):
    def __init__(self, f_net, g_net):
        super().__init__()
        self.f = f_net
        self.router = g_net

    def forward_f(self, x_dgns, x_prcdr, x_numeric):
        return self.f(x_dgns, x_prcdr, x_numeric)

    def forward(self, x_dgns, x_prcdr, x_numeric):
        return self.router(x_dgns, x_prcdr, x_numeric)



def evaluate(config):
    device = torch.device(config.device)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    d = HARPDataset(config.reviewer_costs, "test")
    test_loader = DataLoader(d, batch_size=config.batch_size, shuffle=False)

    val = HARPDataset(config.reviewer_costs, "val")
    csc_val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False)

    encoder = HARPEncoder(
        config.encoding_dim,
        config.numeric_dim,
        config.num_enc_heads,
        config.num_enc_layers,
        config.dgns_vocab_size,
        config.prcdr_vocab_size
    ).to(device)

    f = MinimalFNet(
        encoder, 
        config.encoding_dim,
        config.f_hidden_dims,
        1,
        config.f_drouput
    ).to(device).eval()

    if os.path.exists(config.f_weights_path):
        f.load_state_dict(torch.load(config.f_weights_path, map_location=device, weights_only=True))
        print(f"Loaded F-Net from {config.f_weights_path}")
    else:
        print("Warning: F-Net weights not found.")

    encoder = f.encoder

    g = CSCSelector(
        f,
        config.csc_hidden_dim,
        config.csc_num_layers,
        config.csc_drouput
    ).to(device).eval()

    if os.path.exists(config.g_weights_path):
        g.load_state_dict(torch.load(config.g_weights_path, map_location=device, weights_only=True))
    
    pi = HARPRouter(
        encoder, 
        len(config.reviewer_costs),
        config.encoding_dim,
        config.pi_hidden_dims
    ).to(device).eval()
    
    if hasattr(config, 'pi_weights_path') and os.path.exists(config.pi_weights_path):
        pi.load_state_dict(torch.load(config.pi_weights_path, map_location=device, weights_only=True))
        print(f"Loaded HARP Router from {config.pi_weights_path}")


    models_to_test = {
        "naive": (Naive(f, config.naive_threshold), "naive"),
        "oracle": (Oracle(f, config.error_penalty), "oracle"),
        "csc": (CSCWrapper(f, CSC_Router(g, csc_val_loader, len(config.reviewer_costs), config.csc_coverages)).eval(), "csc"),
        "harp": (HARPWrapper(f, pi).eval(), "harp")
    }

    full_results = {}
    json_results = {}
    
    print("\n--- Starting Evaluation ---")
    
    for name, (model, mode) in models_to_test.items():
        try:
            metrics, scores, correct = evaluate_model(model, test_loader, device, mode=mode)
            
            json_results[name] = metrics
            
            full_results[name] = {
                "metrics": metrics,
                "scores": scores,
                "correct": correct
            }
            
            print(f"Done. Acc: {metrics['System_Accuracy']:.4f}, Risk: {metrics['AI_Risk']:.4f}")
        except Exception as e:
            print(f"Failed to evaluate {name}: {str(e)}")
            json_results[name] = {"error": str(e)}

    out_file = os.path.join(output_dir, "evaluation_metrics.json")
    with open(out_file, "w") as f:
        json.dump(json_results, f, indent=4)
    
    print("Generating plots...")
    plot_static_metrics(full_results, output_dir)
    plot_rc_curves(full_results, output_dir)
    
    print(f"\nEvaluation complete. Results and plots saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate HARP Models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="harp/config/evaluate_config.yaml", 
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    evaluate(config)
import torch
import argparse
from torch.utils.data import DataLoader

from harp.models.f_networks import FNetwork, window_early_exit_route
from harp.models.encoder import HARPEncoder
from harp.data.load_config import load_config
from harp.data.class_dataset import EarlyExitDataset


def evaluate(config, model_path):
    """Evaluate Window Early Exit model"""
    print(f"{'='*60}")
    print(f"Evaluating Window Early Exit Model")
    print(f"Model: {model_path}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    device = torch.device(config.device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = EarlyExitDataset(mode="test")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Create encoder
    encoder = HARPEncoder(
        encoding_dim=config.encoding_dim,
        numeric_dim=config.numeric_dim,
        num_enc_heads=config.num_enc_heads,
        num_enc_layers=config.num_enc_layers,
        dgns_vocab_size=config.dgns_vocab_size,
        prcdr_vocab_size=config.prcdr_vocab_size
    ).to(device)
    
    # Create model
    junior_hidden = getattr(config, 'junior_hidden', [128, 64])
    senior_hidden = getattr(config, 'senior_hidden', [256, 256, 128, 64])
    
    model = FNetwork(
        encoder=encoder,
        input_dim=config.encoding_dim,
        junior_hidden=junior_hidden,
        senior_hidden=senior_hidden,
        output_dim=1
    ).to(device)
    
    # Load weights
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded.\n")
    
    # Evaluation
    print("Running evaluation...")
    all_predictions = []
    all_labels = []
    all_stages = []
    all_confidences = []
    
    tau_junior = getattr(config, 'tau_junior', 0.85)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_dgns = batch["x_cat_dgns"].to(device)
            x_prcdr = batch["x_cat_prcdr"].to(device)
            x_numeric = batch["x_num"].to(device)
            y = batch["y_claim_status"].to(device)
            
            # Process each sample in batch
            for i in range(len(x_dgns)):
                result = window_early_exit_route(
                    model,
                    x_dgns[i:i+1],
                    x_prcdr[i:i+1],
                    x_numeric[i:i+1],
                    tau_junior=tau_junior
                )
                
                all_predictions.append(result['prediction'].item())
                all_labels.append(y[i].item())
                all_stages.append(result['stage'])
                all_confidences.append(result['confidence'].item())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")
    
    # Calculate metrics
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    
    accuracy = (all_predictions == all_labels).float().mean().item()
    
    junior_count = sum(1 for s in all_stages if s == 'junior')
    senior_count = sum(1 for s in all_stages if s == 'senior')
    
    avg_confidence = sum(all_confidences) / len(all_confidences)
    
    # Stage-specific accuracy
    junior_preds = [p for p, s in zip(all_predictions, all_stages) if s == 'junior']
    junior_labels = [l for l, s in zip(all_labels, all_stages) if s == 'junior']
    senior_preds = [p for p, s in zip(all_predictions, all_stages) if s == 'senior']
    senior_labels = [l for l, s in zip(all_labels, all_stages) if s == 'senior']
    
    junior_acc = (torch.tensor(junior_preds) == torch.tensor(junior_labels)).float().mean().item() if junior_preds else 0.0
    senior_acc = (torch.tensor(senior_preds) == torch.tensor(senior_labels)).float().mean().item() if senior_preds else 0.0
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"")
    print(f"Junior Exits: {junior_count} ({junior_count/len(all_stages)*100:.1f}%)")
    print(f"  - Accuracy: {junior_acc*100:.2f}%")
    print(f"Senior Exits: {senior_count} ({senior_count/len(all_stages)*100:.1f}%)")
    print(f"  - Accuracy: {senior_acc*100:.2f}%")
    print(f"")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Window Early Exit Model")
    parser.add_argument(
        "--config",
        type=str,
        default="harp/config/window_early_exit_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--tau_junior",
        type=float,
        default=0.85,
        help="Junior confidence threshold (default: 0.85)"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    config.tau_junior = args.tau_junior
    evaluate(config, args.model)


import torch
import time
import os
import argparse
from torch.utils.data import DataLoader

from harp.models import ClassWindowEarlyExit
from harp.data.load_config import load_config
from harp.data.class_dataset import EarlyExitDataset


def train(config):
    """Training loop for Window Early Exit"""
    print(f"{'='*60}")
    print(f"Starting Window Early Exit Training...")
    print(f"Train Stage: {config.train_stage}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs} | Batch Size: {config.batch_size} | LR: {config.lr}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = EarlyExitDataset(mode="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,  # Parallel data loading (faster)
        pin_memory=True if config.device != 'cpu' else False  # Faster GPU transfer
    )
    print(f"Dataset loaded: {len(train_dataset)} samples\n")
    
    # Create model wrapper
    model_wrapper = ClassWindowEarlyExit(config, train_stage=config.train_stage)
    
    # Create save directory
    if config.save_dir:
        os.makedirs(config.save_dir, exist_ok=True)
    
    total_start_time = time.time()
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        running_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            loss = model_wrapper.train_step(batch)
            running_loss += loss
            
            if batch_idx % config.log_interval == 0 and batch_idx > 0:
                print(f"   [Epoch {epoch}/{config.epochs}] Batch {batch_idx}/{num_batches} | "
                     f"Loss: {loss:.4f}")
        
        avg_loss = running_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"--> [Epoch {epoch}/{config.epochs}] Completed in {epoch_time:.2f}s | "
             f"Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if config.save_dir and epoch % config.save_freq == 0:
            save_path = os.path.join(config.save_dir, f"window_early_exit_epoch_{epoch}.pt")
            torch.save(model_wrapper.get_model().state_dict(), save_path)
            print(f"    Saved checkpoint: {save_path}")
    
    # Save final model
    if config.save_dir:
        final_path = os.path.join(config.save_dir, "window_early_exit_final.pt")
        torch.save(model_wrapper.get_model().state_dict(), final_path)
        print(f"    Saved final model: {final_path}")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Training Complete. Total time: {total_time/60:.2f} minutes.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Window Early Exit Model")
    parser.add_argument(
        "--config",
        type=str,
        default="harp/config/window_early_exit_config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


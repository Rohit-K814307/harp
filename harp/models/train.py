import torch
import time
import os
import argparse
from torch.utils.data import DataLoader

from harp.models import *
from harp.data.load_config import load_config
from harp.data.class_dataset import HARPDataset

def train(args, train_loader):
     """
     Training loop

     Args:
          args: Namespace object containing configuration (hyperparams, paths, device).
          train_loader: PyTorch DataLoader returning dicts with keys:
                         ['x_cat_dgns', 'x_cat_prcdr', 'x_num', 'y_claim_status']
     """
     print(f"{'='*60}")
     print(f"Starting Training...")
     print(f"Mode: {'CSC Selector' if args.train_mode == 'g_net' else 'Minimal F-Net'}")
     print(f"Device: {args.device}")
     print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
     print(f"{'='*60}\n")

     if args.train_mode == 'f_net':
          model_wrapper = ClassMinimalFNet(args)
          model_name = "minimal_fnet"
     elif args.train_mode == 'g_net':
          model_wrapper = ClassCSCSelector(args)
          model_name = "csc_selector"
     else:
          raise ValueError(f"Invalid train_mode in args: {args.train_mode}. Choose 'f_net' or 'csc'.")
     
     if args.save_dir:
          os.makedirs(args.save_dir, exist_ok=True)

     total_start_time = time.time()

     for epoch in range(1, args.epochs + 1):
          epoch_start_time = time.time()
          running_loss = 0.0
          
          num_batches = len(train_loader)
          
          for batch_idx, batch in enumerate(train_loader):
               loss = model_wrapper.train_step(batch)
               running_loss += loss

               if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    print(f"   [Epoch {epoch}/{args.epochs}] Batch {batch_idx}/{num_batches} | "
                         f"Current Loss: {loss:.4f}")

          avg_loss = running_loss / num_batches
          epoch_time = time.time() - epoch_start_time
          
          print(f"--> [Epoch {epoch}/{args.epochs}] Completed in {epoch_time:.2f}s | "
               f"Avg Loss: {avg_loss:.6f}")

          if args.save_dir and epoch % args.save_freq == 0:
               save_path = os.path.join(args.save_dir, f"{model_name}_epoch_{epoch}.pt")
               
               if args.train_mode == 'f_net':
                    torch.save(model_wrapper.f.state_dict(), save_path)
               elif args.train_mode == 'csc':
                    torch.save(model_wrapper.g.state_dict(), save_path)
                    
               print(f"    Saved checkpoint: {save_path}")

     total_time = time.time() - total_start_time
     print(f"\n{'='*60}")
     print(f"Training Complete. Total time: {total_time/60:.2f} minutes.")
     print(f"{'='*60}")


if __name__ == "__main__":
     
     parser = argparse.ArgumentParser(description="Generate HARP Dataset from CMS Claims")
     parser.add_argument(
          "--config", 
          type=str, 
          default="harp/config/csc_f_train_config.yaml", 
          help="Path to the YAML configuration file (default: harp/config/csc_f_train_config.yaml)"
     )

     args = parser.parse_args()
     config = load_config(args.config)

     d = HARPDataset(config.reviewer_costs, "train")
     train(config, DataLoader(d, batch_size=config.batch_size, shuffle=True))

     
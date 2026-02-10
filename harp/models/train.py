import torch
import time
import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from harp.models import *
from harp.data.load_config import load_config
from harp.data.class_dataset import HARPDataset

def train(args, train_loader, val_loader):
     """
     Training loop

     Args:
          args: Namespace object containing configuration (hyperparams, paths, device).
          train_loader: PyTorch DataLoader returning dicts with keys:
                         ['x_cat_dgns', 'x_cat_prcdr', 'x_num', 'y_claim_status']
     """
     print(f"{'='*60}")
     print(f"Starting Training...")
     print(f"Mode: {args.train_mode}")
     print(f"Device: {args.device}")
     print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
     print(f"{'='*60}\n")

     if args.train_mode == 'f_net':
          model_wrapper = ClassMinimalFNet(args)
          model_name = "minimal_fnet"
     elif args.train_mode == 'g_net':
          model_wrapper = ClassCSCSelector(args)
          model_name = "g_net"
     elif args.train_mode == 'harp_pi':
          model_wrapper = ClassHARPNet(args)
          model_name = "harp_pi"
     else:
          raise ValueError(f"Invalid train_mode in args: {args.train_mode}. Choose 'f_net' or 'csc' or 'g_net' or 'harp_pi'.")
     
     if args.save_dir:
          os.makedirs(args.save_dir, exist_ok=True)

     os.makedirs(args.log_dir, exist_ok=True)
     run_name = f"{args.train_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
     writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

     total_start_time = time.time()
     global_step = 0

     for epoch in range(1, args.epochs + 1):

          epoch_start_time = time.time()
          running_loss = 0.0
          num_batches = len(train_loader)
          
          for batch_idx, batch in enumerate(train_loader):
               loss = model_wrapper.train_step(batch, epoch)
               running_loss += loss

               writer.add_scalar('Loss/Train_Iter', loss, global_step)
               global_step += 1

               if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    print(f"   [Epoch {epoch}/{args.epochs}] Batch {batch_idx}/{num_batches} | "
                         f"Current Loss: {loss:.4f}")

          avg_train_loss = running_loss / num_batches
          epoch_time = time.time() - epoch_start_time

          writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
          
          avg_val_loss = None

          if epoch % args.val_freq == 0:
               model_wrapper.eval()
               running_val_loss = 0.0
               num_val_batches = len(val_loader)
               
               with torch.no_grad():
                    for batch in val_loader:                    
                         val_loss = model_wrapper.val_step(batch)
                         running_val_loss += val_loss

               if num_val_batches > 0:
                    avg_val_loss = running_val_loss / num_val_batches
                    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
                    writer.add_scalars('Loss/Combined', {
                         'Train': avg_train_loss,
                         'Validation': avg_val_loss
                    }, epoch)
               else:
                    avg_val_loss = float('nan')
               
               epoch_time = time.time() - epoch_start_time

               model_wrapper.train()
          
          val_str = f"Val Loss: {avg_val_loss:.6f}" if avg_val_loss is not None else "Val Loss: N/A"
          print(f"--> [Epoch {epoch}/{args.epochs}] Completed in {epoch_time:.2f}s | "
               f"Train Loss: {avg_train_loss:.6f} | {val_str}")

          if args.save_dir and epoch % args.save_freq == 0:
               save_path = os.path.join(args.save_dir, f"{model_name}_epoch_{epoch}.pt")
               
               if args.train_mode == 'f_net':
                    torch.save(model_wrapper.f.state_dict(), save_path)
               elif args.train_mode == 'g_net':
                    torch.save(model_wrapper.g.state_dict(), save_path)
               elif args.train_mode == 'harp_pi':
                    torch.save(model_wrapper.pi.state_dict(), save_path)
                    
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
     val = HARPDataset(config.reviewer_costs, "val")
     train(config, DataLoader(d, batch_size=config.batch_size, shuffle=True), DataLoader(val, batch_size=config.batch_size, shuffle=True))

     
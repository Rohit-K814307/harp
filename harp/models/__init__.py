import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

from harp.models.baselines import CSCSelector
from harp.models.f_networks import MinimalFNet
from harp.models.encoder import HARPEncoder
from harp.models.pi_networks import HARPRouter


class ClassMinimalFNet:

     def __init__(self, args):
          self.device = torch.device(args.device)

          self.encoder = HARPEncoder(
               encoding_dim = args.encoding_dim,
               numeric_dim = args.numeric_dim,
               num_enc_heads = args.num_enc_heads,
               num_enc_layers = args.num_enc_layers,
               dgns_vocab_size = args.dgns_vocab_size,
               prcdr_vocab_size = args.prcdr_vocab_size
          ).to(self.device)

          self.f = MinimalFNet(
               encoder = self.encoder,
               input_dim = args.encoding_dim,
               hidden_dims = args.f_hidden_dims,
               output_dim = 1,
               dropout = args.f_drouput
          ).to(self.device)

          self.encoder.train()
          self.f.train()

          self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.bce_pos_weight))
          self.optimizer = optim.Adam(self.f.parameters(), lr=args.lr)

     def forward(self, x_dgns, x_prcdr, x_numeric):
          return self.f(x_dgns, x_prcdr, x_numeric)
     
     def backward(self, loss):
          loss.backward()

     def get_model(self):
          return self.f

     def train_step(self, batch, epoch=None):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          
          # Safely reshaping just in case, but keeping your logic if it works
          y = batch["y_claim_status"].to(self.device).float().view(-1, 1)

          self.optimizer.zero_grad()

          y_hat = self.forward(x_dgns, x_prcdr, x_numeric)
          loss = self.criterion(y_hat, y)
          self.backward(loss)
          self.optimizer.step()

          return loss.item()
     
     def val_step(self, batch):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          y = batch["y_claim_status"].to(self.device).float().view(-1, 1)
          
          y_hat = self.forward(x_dgns, x_prcdr, x_numeric).detach()
          loss = self.criterion(y_hat, y).detach()

          return loss.item()
     
     def eval(self):
          self.f.eval()

     def train(self):
          self.f.train()


class ClassCSCSelector:

     def __init__(self, args):
          self.device = torch.device(args.device)

          encoder = HARPEncoder(
               encoding_dim = args.encoding_dim,
               numeric_dim = args.numeric_dim,
               num_enc_heads = args.num_enc_heads,
               num_enc_layers = args.num_enc_layers,
               dgns_vocab_size = args.dgns_vocab_size,
               prcdr_vocab_size = args.prcdr_vocab_size
          ).to(self.device)

          self.f = MinimalFNet(
               encoder = encoder,
               input_dim = args.encoding_dim,
               hidden_dims = args.f_hidden_dims,
               output_dim = 1,
               dropout = args.f_drouput
          ).to(self.device)

          if args.f_weights_path:
               self.f.load_state_dict(torch.load(args.f_weights_path, map_location=self.device, weights_only=True))

          self.encoder = self.f.encoder
          self.f.eval()
          self.encoder.eval()

          self.g = CSCSelector(
               f = self.f,
               csc_hidden_dim = args.csc_hidden_dim,
               csc_num_layers = args.csc_num_layers,
               csc_dropout = args.csc_dropout,
          ).train().to(self.device)

          self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.bce_pos_weight))
          self.optimizer = optim.Adam(self.g.parameters(), lr=args.lr)

     def forward(self, x_dgns, x_prcdr, x_numeric):
          with torch.no_grad():
               logits = self.f(x_dgns, x_prcdr, x_numeric)
               preds = (torch.sigmoid(logits) > 0.5).float()
               
          trust, _ = self.g(x_dgns, x_prcdr, x_numeric)
          return trust, preds
     
     def backward(self, loss):
          loss.backward()

     def train_step(self, batch, epoch=None):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          
          y = batch["y_claim_status"].to(self.device).view(-1, 1)

          self.optimizer.zero_grad()

          trust, preds = self.forward(x_dgns, x_prcdr, x_numeric)

          y_trust = (preds == y).float().view_as(trust)

          loss = self.criterion(trust, y_trust)
          self.backward(loss)
          self.optimizer.step()

          return loss.item()

     def val_step(self, batch):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          y = batch["y_claim_status"].to(self.device).view(-1, 1)
          
          trust, preds = self.forward(x_dgns, x_prcdr, x_numeric)
          trust, preds = trust.detach(), preds.detach()

          y_trust = (preds == y).float().view_as(trust)

          loss = self.criterion(trust, y_trust).detach()

          return loss.item()
     
     def eval(self):
          self.g.eval()

     def train(self):
          self.g.train()


class ClassHARPNet:

     def __init__(self, args):
          self.device = torch.device(args.device)
          self.error_penalty = args.error_penalty 

          self.tau_schedule = {}
          tau_start = args.gumbel_tau
          tau_end = 0.5                 
          
          for e in range(args.epochs):
               progress = e / max(1, args.epochs - 1) 
               tau_val = tau_start - (tau_start - tau_end) * progress
               self.tau_schedule[e] = max(tau_val, tau_end)

          encoder = HARPEncoder(
               encoding_dim = args.encoding_dim,
               numeric_dim = args.numeric_dim,
               num_enc_heads = args.num_enc_heads,
               num_enc_layers = args.num_enc_layers,
               dgns_vocab_size = args.dgns_vocab_size,
               prcdr_vocab_size = args.prcdr_vocab_size
          ).to(self.device)

          self.f = MinimalFNet(
               encoder = encoder,
               input_dim = args.encoding_dim,
               hidden_dims = args.f_hidden_dims,
               output_dim = 1,
               dropout = args.f_drouput
          ).to(self.device)

          if args.f_weights_path:
               self.f.load_state_dict(torch.load(args.f_weights_path, map_location=self.device, weights_only=True))

          self.encoder = self.f.encoder

          self.f.eval()
          self.encoder.eval() 

          self.pi = HARPRouter(
               encoder = self.encoder,
               n_reviewers = len(args.reviewer_costs),
               encoding_dim = args.encoding_dim,
               hidden_dims = args.pi_hidden_dims
          ).to(self.device)

          self.optimizer = optim.Adam(self.pi.parameters(), lr=args.lr)

     def forward_f(self, x_dgns, x_prcdr, x_numeric):
          with torch.no_grad():
               logits = self.f(x_dgns, x_prcdr, x_numeric)
          return logits

     def forward_pi(self, f_out, x_dgns, x_prcdr, x_numeric):
          return self.pi(f_out, x_dgns, x_prcdr, x_numeric)

     def backward(self, loss):
          loss.backward()
          
     def train_step(self, batch, epoch):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          y = batch["y_claim_status"].to(self.device).view(-1, 1)
          d_humans = batch["d"].to(self.device)
          c_humans = batch["c"].to(self.device)

          self.optimizer.zero_grad()

          f_logits = self.forward_f(x_dgns, x_prcdr, x_numeric)
          f_preds = (f_logits > 0).float()

          pi_logits = self.forward_pi(f_logits, x_dgns, x_prcdr, x_numeric)
          decisions = F.gumbel_softmax(pi_logits, tau=self.tau_schedule.get(epoch, 0.5), hard=True)

          # Calculate Expected Cost
          ai_wrong = (f_preds != y).float()
          cost_ai = self.error_penalty * ai_wrong

          humans_wrong = (d_humans != y).float()
          cost_humans = c_humans + (self.error_penalty * humans_wrong)

          all_costs = torch.cat([cost_ai, cost_humans], dim=1)
          loss = (decisions * all_costs).sum(dim=1).mean()
          
          self.backward(loss)
          self.optimizer.step()

          return loss.item()
     
     def val_step(self, batch):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          y = batch["y_claim_status"].to(self.device).view(-1, 1)
          d_humans = batch["d"].to(self.device)
          c_humans = batch["c"].to(self.device)

          f_logits = self.forward_f(x_dgns, x_prcdr, x_numeric).detach()
          f_preds = (f_logits > 0).float()

          pi_logits = self.forward_pi(f_logits, x_dgns, x_prcdr, x_numeric).detach()
          
          # For validation, we use hard decisions with low temp to approximate real usage
          decisions = F.gumbel_softmax(pi_logits, tau=0.5, hard=True)

          # Calculate Validation Cost (Loss)
          ai_wrong = (f_preds != y).float()
          cost_ai = self.error_penalty * ai_wrong

          humans_wrong = (d_humans != y).float()
          cost_humans = c_humans + (self.error_penalty * humans_wrong)

          all_costs = torch.cat([cost_ai, cost_humans], dim=1)
          loss = (decisions * all_costs).sum(dim=1).mean()

          return loss.item()
     
     def eval(self):
          self.pi.eval()

     def train(self):
          self.pi.train()
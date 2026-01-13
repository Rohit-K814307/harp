import torch.optim as optim
import torch.nn as nn
import torch

from harp.models.csc import CSCSelector
from harp.models.f_networks import MinimalFNet
from harp.models.encoder import HARPEncoder


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
          ).train().to(self.device)

          self.f = MinimalFNet(
               encoder = self.encoder,
               input_dim = args.encoding_dim,
               hidden_dims = args.hidden_dims,
               output_dim = 1,
               dropout = args.f_drouput
          ).train().to(self.device)

          self.criterion = nn.BCEWithLogitsLoss(pos_weight=args.bce_pos_weight)
          self.optimizer = optim.Adam(self.f.parameters(), lr=args.lr)


     def forward(self, x_dgns, x_prcdr, x_numeric):

          return self.f(x_dgns, x_prcdr, x_numeric)
     

     def backward(self, loss):
          loss.backward()

     
     def get_model(self):
          return self.f


     def train_step(self, batch):
  
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          y = batch["y_claim_status"].to(self.device)

          self.optimizer.zero_grad()

          y_hat = self.forward(x_dgns, x_prcdr, x_numeric)
          loss = self.criterion(y_hat, y)
          self.backward(loss)
          self.optimizer.step()

          return loss.item()

               

class ClassCSCSelector:

     def __init__(self, args):

          self.device = torch.device(args.device)

          self.encoder = HARPEncoder(
               encoding_dim = args.encoding_dim,
               numeric_dim = args.numeric_dim,
               num_enc_heads = args.num_enc_heads,
               num_enc_layers = args.num_enc_layers,
               dgns_vocab_size = args.dgns_vocab_size,
               prcdr_vocab_size = args.prcdr_vocab_size
          ).eval().to(self.device)

          self.f = MinimalFNet(
               encoder = self.encoder,
               input_dim = args.encoding_dim,
               hidden_dims = args.hidden_dims,
               output_dim = 1,
               dropout = args.f_drouput
          ).eval().to(self.device)

          self.f.load_state_dict(torch.load(args.f_weights_path), map_location=self.device)

          self.g = CSCSelector(
               f = self.f,
               csc_input_dim = args.csc_input_dim,
               csc_hidden_dim = args.csc_hidden_dim,
               csc_num_layers = args.csc_num_layers,
               csc_dropout = args.csc_dropout,
          ).train().to(self.device)

          self.criterion = nn.BCELoss()
          self.optimizer = optim.Adam(self.g.parameters(), lr=args.lr)

     
     def forward(self, x_dgns, x_prcdr, x_numeric):

          with torch.no_grad():
               logits = self.f(x_dgns, x_prcdr, x_numeric)
               preds = logits.argmax(dim=1)
               

          trust = self.g(x_dgns, x_prcdr, x_numeric)

          return trust, preds
     
     
     def backward(self, loss):

          loss.backward()

     def train_step(self, batch):
          x_dgns, x_prcdr, x_numeric = batch["x_cat_dgns"].to(self.device), batch["x_cat_prcdr"].to(self.device), batch["x_num"].to(self.device)
          y = batch["y_claim_status"].to(self.device)

          self.optimizer.zero_grad()

          trust, preds = self.forward(x_dgns, x_prcdr, x_numeric)
          y_trust = (preds == y).float()

          loss = self.criterion(trust, y_trust)
          self.backward(loss)

          self.optimizer.step()

          return loss.item()




               




import torch.nn.functional as F
import torch.nn as nn
import torch



class HARPRouter(nn.Module):

     def __init__(self, encoder, n_reviewers, encoding_dim, hidden_dims):
          super().__init__()

          self.encoder = encoder # grads dont flow back to avoid double backward

          self.sigmoid = nn.Sigmoid()

          input_dim = encoding_dim + 1 
          output_dim = 1 + n_reviewers 

          layers = []
          curr_dim = input_dim
          
          for h_dim in hidden_dims:
               layers.append(nn.Linear(curr_dim, h_dim))
               layers.append(nn.ReLU())
               layers.append(nn.BatchNorm1d(h_dim))
               curr_dim = h_dim
          
          layers.append(nn.Linear(curr_dim, output_dim))
          
          self.pi = nn.Sequential(*layers)

     
     def forward(self, f_out, x_dgns, x_prcdr, x_numeric):
          
          f_pdf = self.sigmoid(f_out.detach())

          f_entropy = F.binary_cross_entropy(f_pdf, f_pdf, reduction='none')

          with torch.no_grad():
               x_enc = self.encoder(x_dgns, x_prcdr, x_numeric).detach()

          rout_x = torch.cat((x_enc, f_entropy), dim=-1)

          return self.pi(rout_x) # during training, run GUMBEL SOFTMAX ON THIS
     


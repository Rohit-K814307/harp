import torch.nn as nn


# define a minimal f_net simple logistic regressor/DNN
class MinimalFNet(nn.Module):

    def __init__(self, encoder, input_dim, hidden_dims=[128, 64, 32, 16], output_dim=1, dropout=0.1):
        super().__init__()
        
        self.encoder = encoder

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_dgns, x_prcdr, x_numeric):
        x = self.encoder(x_dgns, x_prcdr, x_numeric)
        x = self.net(x)
        return x


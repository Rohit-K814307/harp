import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Window Early Exit Network
class FNetwork(nn.Module):
    """Variable-size network with junior (shallow) and senior (deeper) exit points."""
    
    def __init__(
        self,
        encoder,
        input_dim=32,
        junior_hidden=[128, 64],
        senior_hidden=[256, 256, 128, 64],
        output_dim=1
    ):
        super().__init__()
        
        self.encoder = encoder
        
        # Shared base layers
        self.base_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Junior exit point (shallow)
        self.junior_layers = nn.Sequential(
            nn.Linear(128, junior_hidden[0]),
            nn.ReLU(),
            nn.Linear(junior_hidden[0], junior_hidden[1]),
            nn.ReLU()
        )
        self.junior_head = nn.Linear(junior_hidden[1], output_dim)
        
        # Senior exit point (deeper)
        self.senior_layers = nn.Sequential(
            nn.Linear(128, senior_hidden[0]),
            nn.ReLU(),
            nn.Linear(senior_hidden[0], senior_hidden[1]),
            nn.ReLU(),
            nn.Linear(senior_hidden[1], senior_hidden[2]),
            nn.ReLU(),
            nn.Linear(senior_hidden[2], senior_hidden[3]),
            nn.ReLU()
        )
        self.senior_head = nn.Linear(senior_hidden[3], output_dim)
    
    def forward(self, x_dgns, x_prcdr, x_numeric, exit_point='both'):
        """Forward pass. exit_point: 'junior', 'senior', or 'both'"""
        # Encode input features
        encoded = self.encoder(x_dgns, x_prcdr, x_numeric)
        
        # Shared base layers
        base_out = self.base_layers(encoded)
        
        result = {}
        
        # Junior exit point
        if exit_point in ['junior', 'both']:
            junior_features = self.junior_layers(base_out)
            junior_pred = self.junior_head(junior_features)
            junior_conf = self.get_confidence(junior_pred)
            result['junior_pred'] = junior_pred
            result['junior_conf'] = junior_conf
        
        # Senior exit point
        if exit_point in ['senior', 'both']:
            senior_features = self.senior_layers(base_out)
            senior_pred = self.senior_head(senior_features)
            senior_conf = self.get_confidence(senior_pred)
            result['senior_pred'] = senior_pred
            result['senior_conf'] = senior_conf
        
        return result
    
    def get_confidence(self, logits):
        """Convert logits to confidence: |prob - 0.5| * 2"""
        probs = torch.sigmoid(logits).squeeze(-1)
        return torch.abs(probs - 0.5) * 2


def window_early_exit_route(model, x_dgns, x_prcdr, x_numeric, tau_junior=0.85):
    """Route through junior/senior cascade with early exit."""
    model.eval()
    
    with torch.no_grad():
        outputs = model(x_dgns, x_prcdr, x_numeric, exit_point='both')
        
        if outputs['junior_conf'] >= tau_junior:
            pred_logits = outputs['junior_pred']
            prediction = (torch.sigmoid(pred_logits) > 0.5).float()
            return {
                'prediction': prediction.squeeze(-1),
                'confidence': outputs['junior_conf'],
                'stage': 'junior'
            }
        else:
            pred_logits = outputs['senior_pred']
            prediction = (torch.sigmoid(pred_logits) > 0.5).float()
            return {
                'prediction': prediction.squeeze(-1),
                'confidence': outputs['senior_conf'],
                'stage': 'senior'
            }


# f_net using small llm
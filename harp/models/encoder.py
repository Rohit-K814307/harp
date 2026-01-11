import torch
import torch.nn as nn



class HARPEncoder(nn.Module):

    def __init__(self, encoding_dim, numeric_dim, num_enc_heads, num_enc_layers, dgns_vocab_size, prcdr_vocab_size):
        super().__init__()

        # define the embedding for dgns + prcdr
        self.dgns_emb = nn.Embedding(dgns_vocab_size, encoding_dim)
        self.prcdr_emb = nn.Embedding(prcdr_vocab_size, encoding_dim)

        # define layernorm for numerical features
        #self.num_norm = nn.LayerNorm(numeric_dim)

        # create numeric context for transformer
        self.num_proj = nn.Linear(numeric_dim, encoding_dim)

        # define combined encoding model

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, encoding_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(encoding_dim * 2, encoding_dim)
        # )

        encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_dim, nhead=num_enc_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # linear proj
        self.output_head = nn.Linear(encoding_dim, encoding_dim)

    def forward(self, x_dgns, x_prcdr, x_numeric):
        
        # (Batch, Num_Dgns, Encoding_Dim)
        d = self.dgns_emb(x_dgns)
        #print(d.shape)
        
        # (Batch, Num_Prcdr, Encoding_Dim)
        p = self.prcdr_emb(x_prcdr)
        #print(p.shape)
        
        # (Batch, 1, Encoding_Dim)
        n = self.num_proj(x_numeric).unsqueeze(1) # seq len for transformer
        #print(n.shape)
        
        # concat; num_dgns + num_prcdr + 1
        x = torch.cat([d, p, n], dim=1)
        #print(x.shape)
        
        # transformer
        x = self.transformer(x)
        #print(x.shape)
        
        # mean vec per claim
        x = x.mean(dim=1)
        #print(x.shape)

        return self.output_head(x)


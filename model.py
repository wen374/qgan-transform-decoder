# model decoder for 17 qubit planar code
import torch
import torch.nn as nn
import torch.nn.init as init

class SurfaceCodeDecoder(nn.Module):
    '''
    A decoder for the 17 qubit planar code.

    Takes in two items: 
    - a tensor of shape (batch, len, 8) representing the syndrome measurements 
        or possibly the syndrome increments
    - a tensor of shape (batch, 9) representing the logical X/Z measurements

    The architecture is as follows: 
    syndrome_head:
        a transformer encoder which resembles a ViT. It takes in the sequential
        syndrome measurements, applies a feedforward network, and feeds those
        into a transformer encoder after positional embedding and learned 
        "class" embedding. Embeddings are of dimension n_attn_dims.

    classifier_head:
        An FC layer which takes in the "class" embedding from the transformer
        along with the logical X/Z measurements and outputs a single parity bit.
        
    '''
    def __init__(self, n_attn_dims, n_heads, n_attn_layers, n_ff_dims, n_ff_layers, dropout=0.1, max_seq_len=500):
        super().__init__()
        self.n_dims = n_attn_dims
        self.n_heads = n_heads
        self.n_layers = n_attn_layers
        self.dropout = dropout

        self.class_embedding = nn.Parameter(torch.randn(n_attn_dims))
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, n_attn_dims))

        self.embedder = nn.Linear(8, n_attn_dims)

        layer_norm = nn.LayerNorm(n_attn_dims)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_attn_dims, nhead=n_heads, dim_feedforward=4*n_attn_dims, dropout=dropout)
        self.syndrome_head = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers, norm=layer_norm)

        self.classifier_head = nn.ModuleList()
        for i in range(n_ff_layers):
            from_dim = n_attn_dims + 9 if i == 0 else n_ff_dims
            to_dim = n_ff_dims if i < n_ff_layers - 1 else 1
            self.classifier_head.append(nn.Linear(from_dim, to_dim))
            if to_dim != 1:
                self.classifier_head.append(nn.ReLU())
                self.classifier_head.append(nn.Dropout(dropout))
            else:
                self.classifier_head.append(nn.Sigmoid())
        
        self.classifier_head = nn.Sequential(*self.classifier_head)
    
    def forward(self, syndrome, logical):
        '''
        syndrome: (batch, len, 8)
        logical: (batch, 9)
        '''
        batch_size = syndrome.shape[0]
        seq_len = syndrome.shape[1]
        
        # Embed the syndrome measurements
        syndrome = self.embedder(syndrome)

        # Add class embedding. First, get the shape to be (batch, 1, n_dims)
        class_emb = self.class_embedding.repeat(batch_size, 1, 1)
        # Then, concatenate along the sequence dimension, (batch, len + 1, n_dims)
        syndrome = torch.cat([class_emb, syndrome], dim=1)
        
        # Add positional embedding
        syndrome += self.pos_embedding[:seq_len + 1, :].repeat(batch_size, 1, 1)

        # Hittem with that transformer
        syndrome = self.syndrome_head(syndrome)
        syndrome_out = syndrome[:, 0, :] # (batch, n_dims)

        classificand = torch.cat([syndrome_out, logical], dim=1)

        return self.classifier_head(classificand)
        



    

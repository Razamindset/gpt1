from config import *
import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),    
        )
    
    def forward(self, x):
        return self.net(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):

        super().__init__()
        
        # layer Norm 1
        self.ln1 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # layer Norm 2
        self.ln2 = nn.LayerNorm(embedding_dim)

        # FFN
        self.ffn = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):

        # first of all we normalize
        x_norm = self.ln1(x)

        # Do multi head attention
        # First we need to create a mask for the attention 
        # to prevent from attenidng ti future tokens 
        seq_len = x.shape[1]
        
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        # this creates a matrix with elments above the daignonal to be true and rest to be false 
        # False True  True  True
        # False False True  True
        # False False False True
        # False False False False

        # wherever there is true that will masked and unnattended

        atten_out, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=mask
        )

        x = x + atten_out

        x_norm = self.ln2(x)

        ffn_out = self.ffn(x_norm)

        x = x + ffn_out    

        return x

import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size,embedding_dim, context_length):

        super().__init__()
        
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        postions = torch.arange(
            seq_len,
            device=x.device
        )
        token_emb = self.token_embedding(x)

        pos_emb = self.position_embedding(
            postions
        ) 

        return token_emb + pos_emb
    
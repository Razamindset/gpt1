import torch.nn as nn

from embedding import GPTEmbedding
from transformer import TransformerBlock

class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        context_length,
        num_heads,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = GPTEmbedding(
            vocab_size,
            embedding_dim,
            context_length
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.blocks(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits

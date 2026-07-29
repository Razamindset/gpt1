import torch.nn as nn
import torch
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

        self.context_length = context_length

        self.ln_final = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

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
        

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits

    # ! Legacy, Now using generate.py instead
    # def generate(self, idx, max_new_tokens):
    #     # idx.shape = (batch_size, current_sequence_length)
    #     # [[12, 45, 83]]
    #     # Batch = 1
    #     # Prompt:
    #     # "The king"

    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -self.context_length:]

    #         logits = self(idx_cond)
    #         # (batch, seq_len, vocab_size)

    #         # Use the last token fpr prediction
    #         logits = logits[:, -1, :]

    #         probs = torch.softmax(logits, dim=-1)

    #         # Sampling  
    #         next_token = torch.multinomial(
    #             probs,
    #             num_samples=1
    #         )

    #         # Append this new token to the seq
    #         idx = torch.cat(
    #             (idx, next_token),
    #             dim=1
    #         )

    #     return idx
from tokenizer import BPETokenizer
from dataset import GPTDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from model import GPT
from config import *
    
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

# Load tokenizer
tokenizer = BPETokenizer()

tokenizer.load("tokenizer.json")

ids = tokenizer.encode(text, add_special_tokens=True)

dataset = GPTDataset(
    ids,
    block_size=CONTEXT_LENGTH,
    stride=DATASET_STRIDE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

vocab_size = len(tokenizer.token_to_id)

VOCAB_SIZE = vocab_size

model = GPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


print("Number of token IDs:", len(ids))
print("Dataset size:", len(dataset))
print("Number of batches:", len(loader))
print("Max batches per epoch:", MAX_BATCHES_PER_EPOCH)

for epoch in range(EPOCHS):

    model.train()

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= MAX_BATCHES_PER_EPOCH:
            break

        logits = model(x)

        vocab_size = logits.shape[-1]

        logits = logits.view(-1, vocab_size)
        y = y.view(-1)

        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch_idx % PRINT_EVERY == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{min(len(loader), MAX_BATCHES_PER_EPOCH)} | "
                f"Loss {loss.item():.4f}"
            )

model.eval()

prompt = "To be"

ids = tokenizer.encode(
    prompt,
    add_special_tokens=True
)

# convert ot respective size
# (batch_size, sequence_length)
ids = torch.tensor(
    ids,
    dtype=torch.long
).unsqueeze(0)


with torch.no_grad():
    generated = model.generate(
        ids,
        max_new_tokens=100
    )

generated = generated.squeeze(0).tolist()

text = tokenizer.decode(generated)

print(text)

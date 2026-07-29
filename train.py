from tokenizer import BPETokenizer
from dataset import GPTDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from model import GPT
from config import *
import os


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

with open("input.txt", encoding="utf-8") as f:
    text = f.read()

# Load tokenizer
tokenizer = BPETokenizer()

tokenizer.load("tokenizer.json")

ids = tokenizer.encode(text, add_special_tokens=True)

split_idx = int(0.9 * len(ids))

train_ids = ids[:split_idx]
val_ids = ids[split_idx:]

train_dataset = GPTDataset(
    train_ids,
    block_size=CONTEXT_LENGTH
)

val_dataset = GPTDataset(
    val_ids,
    block_size=CONTEXT_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


print("Number of token IDs:", len(ids))

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

print("Max batches per epoch:", MAX_BATCHES_PER_EPOCH)

def evaluate(model, loader):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            vocab_size = logits.shape[-1]

            logits = logits.view(-1, vocab_size)
            y = y.view(-1)

            loss = F.cross_entropy(logits, y)

            total_loss += loss.item()

    return total_loss / len(loader)


best_val_loss = float("inf")
start_epoch = 0

if os.path.exists("best_model.pt"):
    checkpoint = torch.load("best_model.pt", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_loss = checkpoint["val_loss"]
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):

    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= MAX_BATCHES_PER_EPOCH:
            break

        x = x.to(device)
        y = y.to(device)

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
                f"Batch {batch_idx}/{min(len(train_loader), MAX_BATCHES_PER_EPOCH)} | "
                f"Loss {loss.item():.4f}"
            )

    
    train_loss = loss.item()

    val_loss = evaluate(
            model,
            val_loader
        )

    print(
            f"\nEpoch {epoch+1}"
        )

    print(
            f"Train Loss: {train_loss:.4f}"
        )

    print(
            f"Validation Loss: {val_loss:.4f}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            "best_model.pt",
        )
        print("Best model saved")

print("Saving final model")
torch.save(
    model.state_dict(),
    "final_model.pt"
)
from tokenizer import BPETokenizer
from dataset import GPTDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from model import GPT
from config import *
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open("input.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
tokenizer.load("tokenizer.json")

ids = tokenizer.encode(text, add_special_tokens=True)

split_idx = int(0.9 * len(ids))
train_ids = ids[:split_idx]
val_ids = ids[split_idx:]

# stride now explicit instead of silently defaulting to 1
train_dataset = GPTDataset(train_ids, block_size=CONTEXT_LENGTH, stride=DATASET_STRIDE)
val_dataset = GPTDataset(val_ids, block_size=CONTEXT_LENGTH, stride=DATASET_STRIDE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,       # new: keep GPU fed
    pin_memory=True,     # new
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

vocab_size = len(tokenizer.token_to_id)
VOCAB_SIZE = vocab_size

model = GPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# resume from last checkpoint not just "best" so a Colab
# disconnect never loses more than one epoch of progress ---
resume_path = "last_model.pt" if os.path.exists("last_model.pt") else (
    "best_model.pt" if os.path.exists("best_model.pt") else None
)

if resume_path:
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)
    print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
    

# 2. now create the scheduler, telling it where to resume from
total_steps = min(len(train_loader), MAX_BATCHES_PER_EPOCH) * EPOCHS

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda, last_epoch=global_step - 1
)


# mixed precision 
use_amp = USE_AMP and device.type == "cuda"
scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

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
            x, y = x.to(device), y.to(device)
            logits = model(x)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


best_val_loss = float("inf")
start_epoch = 0
global_step = 0


for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= MAX_BATCHES_PER_EPOCH:
            break

        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            logits = model(x)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # gradient clipping (unscale first when using AMP)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

        if batch_idx % PRINT_EVERY == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{min(len(train_loader), MAX_BATCHES_PER_EPOCH)} | "
                f"Loss {loss.item():.4f} | LR {lr_now:.2e}"
            )

    train_loss = running_loss / max(1, num_batches)  # real epoch average now
    val_loss = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    # always save "last" so you can resume after a disconnect ---
    if SAVE_EVERY_EPOCH:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "global_step": global_step,
            },
            "last_model.pt",
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "global_step": global_step,
            },
            "best_model.pt",
        )
        print("Best model saved")

print("Saving final model")
torch.save(model.state_dict(), "final_model.pt")
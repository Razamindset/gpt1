"""
Train the GPT model. Designed to run comfortably on Colab:

  - All data/tokenizer/checkpoints/logs/plots live under a Drive-backed
    project folder (see utils.setup_dirs), so a Colab disconnect never
    loses more than the current batch of work.
  - Resumes automatically from the last checkpoint if one exists.
  - Saves a loss-curve PNG after every epoch.

Usage:
    python train.py --epochs 20 --batch-size 64
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from dataset import GPTDataset
from model import GPT
from tokenizer import BPETokenizer
from utils import TrainingLogger, load_checkpoint, save_checkpoint, setup_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Train the gpt1 model.")
    parser.add_argument("--project", default=config.PROJECT_NAME)
    parser.add_argument("--data-name", default="input.txt")
    parser.add_argument("--tokenizer-name", default="tokenizer.json")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max-batches-per-epoch", type=int, default=config.MAX_BATCHES_PER_EPOCH,
                         help="Cap on batches per epoch. Omit, or pass 0, to train on the full dataset each epoch (default).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints, start fresh.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA.")
    args = parser.parse_args()
    return args


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    args = parse_args()
    dirs = setup_dirs(project_name=args.project)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    data_path = os.path.join(dirs["data"], args.data_name)
    tokenizer_path = os.path.join(dirs["checkpoints"], args.tokenizer_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No corpus at {data_path}. Run download_dataset.py first.")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"No tokenizer at {tokenizer_path}. Run train_tokenizer.py first.")

    with open(data_path, encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    ids = tokenizer.encode(text, add_special_tokens=True)
    split_idx = int((1 - config.VAL_SPLIT) * len(ids))
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]

    train_dataset = GPTDataset(train_ids, block_size=config.CONTEXT_LENGTH, stride=config.DATASET_STRIDE)
    val_dataset = GPTDataset(val_ids, block_size=config.CONTEXT_LENGTH, stride=config.DATASET_STRIDE)

    common_loader_kwargs = dict(
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, **common_loader_kwargs)

    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocab size: {vocab_size}")
    print(f"Token IDs: {len(ids):,} | Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")

    max_batches = args.max_batches_per_epoch
    if max_batches is not None and max_batches <= 0:
        max_batches = None
    steps_per_epoch = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
    if max_batches is None:
        print(f"Train batches/epoch: {steps_per_epoch} (full dataset, no cap)")
    else:
        print(f"Train batches/epoch: {steps_per_epoch} (capped at {max_batches})")

    # --- model / optimizer ---
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        context_length=config.CONTEXT_LENGTH,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    # --- resume (fixed: these are always defined now, even with no checkpoint) ---
    best_val_loss = float("inf")
    start_epoch = 0
    global_step = 0

    last_ckpt_path = os.path.join(dirs["checkpoints"], "last_model.pt")
    best_ckpt_path = os.path.join(dirs["checkpoints"], "best_model.pt")

    resume_path = None
    if not args.no_resume:
        if os.path.exists(last_ckpt_path):
            resume_path = last_ckpt_path
        elif os.path.exists(best_ckpt_path):
            resume_path = best_ckpt_path

    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from {resume_path} at epoch {start_epoch}, step {global_step}")
    else:
        print("Starting a fresh run (no checkpoint found).")

    if start_epoch >= args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch} >= --epochs {args.epochs}. Nothing to do.")
        return

    # --- LR schedule (warmup + cosine decay), resume-aware ---
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < config.WARMUP_STEPS:
            return step / max(1, config.WARMUP_STEPS)
        progress = (step - config.WARMUP_STEPS) / max(1, total_steps - config.WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=global_step - 1)

    # --- mixed precision (fixed: single modern torch.amp API, no deprecated calls) ---
    use_amp = config.USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    logger = TrainingLogger(dirs["logs"], dirs["plots"], resume=(resume_path is not None))

    config_snapshot = {
        "vocab_size": vocab_size,
        "embedding_dim": config.EMBEDDING_DIM,
        "context_length": config.CONTEXT_LENGTH,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
    }

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                vocab_size_ = logits.shape[-1]
                loss = F.cross_entropy(logits.view(-1, vocab_size_), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = loss.item()
            running_loss += loss_value
            num_batches += 1
            global_step += 1

            if batch_idx % config.PRINT_EVERY == 0:
                logger.log_step(global_step, loss_value)

            pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / max(1, num_batches)
        val_loss = evaluate(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch + 1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}")

        logger.log_epoch(epoch + 1, train_loss, val_loss, lr_now)
        plot_path = logger.plot()
        print(f"Loss curve updated: {plot_path}")

        if config.SAVE_EVERY_EPOCH:
            save_checkpoint(last_ckpt_path, model, optimizer, epoch, global_step, val_loss, best_val_loss, config_snapshot)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, global_step, val_loss, best_val_loss, config_snapshot)
            print("New best model saved.")

    final_path = os.path.join(dirs["checkpoints"], "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()

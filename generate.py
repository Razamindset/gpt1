"""
Generate text from a trained gpt1 checkpoint.

Usage:
    python generate.py --prompt "To be or not to be" --checkpoint best
    python generate.py   # prompts interactively
"""

import argparse
import os

import torch

import config
from model import GPT
from tokenizer import BPETokenizer
from utils import setup_dirs


def top_k_sample(logits, k, temperature):
    logits = logits / temperature
    values, indices = torch.topk(logits, min(k, logits.shape[-1]))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, indices, values)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate(model, tokenizer, prompt, device, context_length,
             max_new_tokens=200, temperature=0.8, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    eos_id = tokenizer.token_to_id.get("<eos>")

    for _ in range(max_new_tokens):
        input_ids = ids[:, -context_length:]
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        next_token = top_k_sample(next_token_logits, top_k, temperature)
        ids = torch.cat([ids, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(ids.squeeze(0).tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained gpt1 model.")
    parser.add_argument("--project", default=config.PROJECT_NAME)
    parser.add_argument("--tokenizer-name", default="tokenizer.json")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last", "final"],
                         help="Which saved checkpoint to load.")
    parser.add_argument("--prompt", default=None, help="If omitted, you'll be prompted interactively.")
    parser.add_argument("--max-new-tokens", type=int, default=config.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.SAMPLE_FROM_K)
    args = parser.parse_args()

    dirs = setup_dirs(project_name=args.project)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer_path = os.path.join(dirs["checkpoints"], args.tokenizer_name)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"No tokenizer at {tokenizer_path}. Run train_tokenizer.py first.")

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token_to_id)

    ckpt_name = {"best": "best_model.pt", "last": "last_model.pt", "final": "final_model.pt"}[args.checkpoint]
    ckpt_path = os.path.join(dirs["checkpoints"], ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train the model first.")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Prefer architecture stored in the checkpoint (guarantees a match);
    # fall back to config.py for older checkpoints / final_model.pt (state_dict only).
    arch = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if arch is None:
        arch = dict(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            context_length=config.CONTEXT_LENGTH,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
        )

    model = GPT(
        vocab_size=arch["vocab_size"],
        embedding_dim=arch["embedding_dim"],
        context_length=arch["context_length"],
        num_heads=arch["num_heads"],
        num_layers=arch["num_layers"],
        dropout=arch.get("dropout", config.DROPOUT),
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    prompt = args.prompt if args.prompt is not None else input("Prompt: ")

    output = generate(
        model, tokenizer, prompt, device,
        context_length=arch["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n" + output)


if __name__ == "__main__":
    main()

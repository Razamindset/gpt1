"""
Generate a batch of sample completions from a trained checkpoint and save
them to disk. Handy for eyeballing quality after training, or for grabbing
a few good examples to show off.

Usage:
    python generate_samples.py --checkpoint best
    python generate_samples.py --prompts "Once upon a time" "The scientist said" --temperatures 0.7 1.0
"""

import argparse
import os
from datetime import datetime

import torch

import config
from generate import generate
from model import GPT
from tokenizer import BPETokenizer
from utils import setup_dirs

DEFAULT_PROMPTS = [
    "Once upon a time",
    "The meaning of life is",
    "In the beginning",
    "To be or not to be",
    "The scientist looked at the data and said",
]


def load_model_and_tokenizer(dirs, checkpoint_name, device, tokenizer_name="tokenizer.json"):
    tokenizer_path = os.path.join(dirs["checkpoints"], tokenizer_name)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"No tokenizer at {tokenizer_path}. Run train_tokenizer.py first.")

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    ckpt_path = os.path.join(dirs["checkpoints"], checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train the model first.")

    checkpoint = torch.load(ckpt_path, map_location=device)

    arch = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if arch is None:
        arch = dict(
            vocab_size=len(tokenizer.token_to_id),
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
    return model, tokenizer, arch


def main():
    parser = argparse.ArgumentParser(description="Generate a batch of samples from a gpt1 checkpoint.")
    parser.add_argument("--project", default=config.PROJECT_NAME)
    parser.add_argument("--tokenizer-name", default="tokenizer.json")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last", "final"])
    parser.add_argument("--prompts", nargs="*", default=None, help="Custom prompts; defaults to a built-in set.")
    parser.add_argument("--num-samples", type=int, default=1, help="How many samples per (prompt, temperature).")
    parser.add_argument("--max-new-tokens", type=int, default=config.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperatures", nargs="*", type=float, default=[0.6, 0.8, 1.0],
                         help="Sample once per temperature listed, per prompt.")
    parser.add_argument("--top-k", type=int, default=config.SAMPLE_FROM_K)
    args = parser.parse_args()

    dirs = setup_dirs(project_name=args.project)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_name = {"best": "best_model.pt", "last": "last_model.pt", "final": "final_model.pt"}[args.checkpoint]
    model, tokenizer, arch = load_model_and_tokenizer(dirs, ckpt_name, device, args.tokenizer_name)

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    lines = [
        "# gpt1 sample outputs",
        f"checkpoint: {ckpt_name}  |  generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for prompt in prompts:
        for temp in args.temperatures:
            for i in range(args.num_samples):
                text = generate(
                    model, tokenizer, prompt, device,
                    context_length=arch["context_length"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=temp,
                    top_k=args.top_k,
                )
                header = f'Prompt: "{prompt}"  (temperature={temp}, sample {i + 1}/{args.num_samples})'
                print(header)
                print(text)
                print("-" * 60)
                lines += [header, text, "-" * 60]

    out_path = os.path.join(dirs["logs"], "samples.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")

    print(f"\nAll samples appended to {out_path}")


if __name__ == "__main__":
    main()

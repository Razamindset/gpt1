"""
Train the BPE tokenizer on the downloaded corpus and save it into the
run's persistent (Drive-backed on Colab) directory.

Usage:
    python train_tokenizer.py --num-merges 4000
"""

import argparse
import os

from tokenizer import BPETokenizer
from utils import setup_dirs
import config


def main():
    parser = argparse.ArgumentParser(description="Train the gpt1 BPE tokenizer.")
    parser.add_argument("--project", default=config.PROJECT_NAME)
    parser.add_argument("--data-name", default="input.txt")
    parser.add_argument("--num-merges", type=int, default=config.NUM_MERGES)
    parser.add_argument("--tokenizer-name", default="tokenizer.json")
    args = parser.parse_args()

    dirs = setup_dirs(project_name=args.project)
    data_path = os.path.join(dirs["data"], args.data_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No corpus found at {data_path}. Run download_dataset.py first."
        )

    with open(data_path, encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded corpus: {len(text):,} characters from {data_path}")

    tokenizer = BPETokenizer()
    print(f"Training BPE tokenizer with {args.num_merges} merges...")
    tokenizer.train(text, num_merges=args.num_merges)

    out_path = os.path.join(dirs["checkpoints"], args.tokenizer_name)
    tokenizer.save(out_path)

    print(f"Vocab size: {len(tokenizer.token_to_id)}")
    print(f"Tokenizer saved to {out_path}")


if __name__ == "__main__":
    main()

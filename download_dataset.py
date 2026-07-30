"""
Download a training corpus for gpt1.

Presets give you a knob for "a little larger" without having to hunt for
URLs yourself:

  tiny    ~1.1MB   Tiny Shakespeare only (the original default)
  small   ~3-4MB   Tiny Shakespeare + a few short Gutenberg novels
  medium  ~8-10MB  small + a handful more Gutenberg novels

All text is concatenated into a single input.txt inside the run's data
directory (Drive-backed on Colab, see utils.setup_dirs), so
train_tokenizer.py / train.py just need to know the project name.

Usage:
    python download_dataset.py --preset small
    python download_dataset.py --urls URL1 URL2 --out-name my_corpus.txt
"""

import argparse
import os
import re
import urllib.request

from utils import setup_dirs

TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Stable, plain-text Project Gutenberg mirrors (UTF-8 "-0.txt" editions).
GUTENBERG_BOOKS = {
    "alice_in_wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "dracula": "https://www.gutenberg.org/files/345/345-0.txt",
}

PRESETS = {
    "tiny": [TINY_SHAKESPEARE],
    "small": [TINY_SHAKESPEARE]
    + [GUTENBERG_BOOKS[k] for k in ["alice_in_wonderland", "sherlock_holmes", "frankenstein"]],
    "medium": [TINY_SHAKESPEARE] + list(GUTENBERG_BOOKS.values()),
}

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def strip_gutenberg_boilerplate(text):
    """Cut Project Gutenberg's license header/footer if present, keep the book."""
    start_match = GUTENBERG_START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = GUTENBERG_END_RE.search(text)
    if end_match:
        text = text[: end_match.start()]
    return text.strip()


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "gpt1-dataset-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="ignore")
    if "gutenberg" in url.lower():
        text = strip_gutenberg_boilerplate(text)
    return text


def build_corpus(urls, limit_mb=None):
    chunks = []
    total_chars = 0
    limit_chars = None if limit_mb is None else int(limit_mb * 1_000_000)

    for url in urls:
        print(f"Downloading: {url}")
        try:
            text = fetch(url)
        except Exception as e:
            print(f"  Skipped ({e})")
            continue
        chunks.append(text)
        total_chars += len(text)
        print(f"  +{len(text):,} chars (running total {total_chars:,})")
        if limit_chars and total_chars >= limit_chars:
            break

    corpus = "\n\n".join(chunks)
    if limit_chars:
        corpus = corpus[:limit_chars]
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Download a training corpus for gpt1.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="tiny",
                         help="Built-in dataset size preset (default: tiny).")
    parser.add_argument("--urls", nargs="*", default=None,
                         help="Custom list of plain-text URLs; overrides --preset.")
    parser.add_argument("--limit-mb", type=float, default=None,
                         help="Optional cap on corpus size in megabytes.")
    parser.add_argument("--project", default="gpt1", help="Project name (Drive/local run folder).")
    parser.add_argument("--out-name", default="input.txt", help="Output filename inside the data dir.")
    args = parser.parse_args()

    dirs = setup_dirs(project_name=args.project)
    urls = args.urls if args.urls else PRESETS[args.preset]

    corpus = build_corpus(urls, limit_mb=args.limit_mb)
    if not corpus.strip():
        raise RuntimeError("No data was downloaded successfully. Check your internet connection / URLs.")

    out_path = os.path.join(dirs["data"], args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    size_mb = os.path.getsize(out_path) / 1_000_000
    print(f"\nSaved {size_mb:.2f}MB to {out_path}")


if __name__ == "__main__":
    main()

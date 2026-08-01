# gpt1

A GPT language model built from scratch in PyTorch — no `transformers`, no
shortcuts. Implemented from scratch:

- Byte Pair Encoding (BPE) tokenizer
- GPT dataset and DataLoader
- Token and positional embeddings
- Transformer decoder blocks
- Causal masked multi-head self-attention
- Feed-forward network (GELU)
- Residual connections and LayerNorm
- GPT language model
- Cross-entropy training loop with AdamW, warmup + cosine LR decay, grad clipping, mixed precision
- Autoregressive text generation (top-k sampling)

## Quickstart — train on Colab (recommended)

Open `notebooks/Train_on_Colab.ipynb` in Google Colab (pick a GPU runtime) and
run it top to bottom. It mounts Google Drive and stores everything —
dataset, tokenizer, checkpoints, logs, loss-curve plots — under
`/content/drive/MyDrive/<project_name>/`, so **nothing is lost if the Colab
runtime disconnects**, and training auto-resumes from the last checkpoint.

## Quickstart — run locally

```bash
pip install -r requirements.txt

# 1. Download a corpus (presets: tiny ~1MB, small ~3-4MB, medium ~8-10MB)
python download_dataset.py --preset small

# 2. Train the BPE tokenizer
python train_tokenizer.py --num-merges 4000

# 3. Train the model (auto-resumes if interrupted; just re-run)
python train.py --epochs 20 --batch-size 64

# 4. Generate text
python generate.py --prompt "Once upon a time" --checkpoint best

# 5. Generate a batch of sample completions across a few prompts/temperatures
python generate_samples.py --checkpoint best
```

By default `train.py` now trains on **every batch in the dataset each epoch**
(`MAX_BATCHES_PER_EPOCH = None` in `config.py`) for `EPOCHS = 40`. Pass
`--max-batches-per-epoch N` to `train.py` if you want to cap it again.

By default everything is written under `./runs/gpt1/` (`data/`,
`checkpoints/`, `logs/`, `plots/`). Pass `--project my_experiment` to any
script to keep separate runs side by side.

Every script prints `[gpt1] Run directory: ...` on startup — check that
line to confirm you're actually writing to Drive (`/content/drive/MyDrive/...`)
and not a local/ephemeral path.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Model/training/generation hyperparameters |
| `utils.py` | Drive/Colab detection, run-directory setup, checkpoint I/O, loss-curve logging & plotting |
| `download_dataset.py` | Fetches and concatenates a training corpus (size presets or custom URLs) |
| `tokenizer.py` | From-scratch BPE tokenizer |
| `train_tokenizer.py` | Trains the tokenizer on the downloaded corpus |
| `dataset.py` | `GPTDataset` — sliding-window (x, y) pairs over token IDs |
| `embedding.py` | Token + learned positional embeddings |
| `transformer.py` | Pre-norm transformer decoder block (self-attn + FFN) |
| `model.py` | Full GPT model (stacked decoder blocks + LM head) |
| `train.py` | Training loop: AMP, warmup+cosine LR, grad clipping, checkpointing/resume, plots |
| `generate.py` | Top-k sampling text generation from a checkpoint |
| `generate_samples.py` | Generates a batch of samples across prompts/temperatures, saves to `logs/samples.txt` |
| `notebooks/Train_on_Colab.ipynb` | End-to-end Colab pipeline |

## Notes

- **Drive persistence fix:** scripts run via `!python train.py` in Colab spawn a
  fresh subprocess, so detecting Colab via already-imported modules doesn't
  work there. `utils.is_colab()` now checks Colab's environment variables and
  package availability instead, so every script correctly finds and writes to
  `/content/drive/MyDrive/...` even when launched with `!python`. Every script
  prints its resolved run directory on startup so you can verify this at a
  glance.
- **Tokenizer training speed:** `train_tokenizer.py` used to rescan the entire
  vocabulary on every single BPE merge — with 4000 merges over a few million
  characters that meant redoing a full pass thousands of times. It now
  maintains an incremental pair-frequency index and only touches the words
  affected by each merge, which is several times faster (speedup grows with
  corpus/vocab size — larger corpora benefit more).
- Checkpoints save the model config alongside the weights, so `generate.py`
  always rebuilds the exact right architecture — no manual hyperparameter
  syncing needed.
- `train.py` saves `last_model.pt` every epoch (for resuming) and
  `best_model.pt` whenever validation loss improves, plus a final
  `final_model.pt` at the end of the run.
- Loss curves (`plots/loss_curve.png`) are regenerated after every epoch:
  per-step training loss, train/val loss per epoch, and the LR schedule.

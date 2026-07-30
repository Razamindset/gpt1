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
```

By default everything is written under `./runs/gpt1/` (`data/`,
`checkpoints/`, `logs/`, `plots/`). Pass `--project my_experiment` to any
script to keep separate runs side by side.

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
| `notebooks/Train_on_Colab.ipynb` | End-to-end Colab pipeline |

## Notes

- Checkpoints save the model config alongside the weights, so `generate.py`
  always rebuilds the exact right architecture — no manual hyperparameter
  syncing needed.
- `train.py` saves `last_model.pt` every epoch (for resuming) and
  `best_model.pt` whenever validation loss improves, plus a final
  `final_model.pt` at the end of the run.
- Loss curves (`plots/loss_curve.png`) are regenerated after every epoch:
  per-step training loss, train/val loss per epoch, and the LR schedule.

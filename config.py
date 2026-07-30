"""
Central hyperparameter / run configuration for gpt1.

Everything a script needs to know about model size, optimization, and
generation lives here so train.py / generate.py / train_tokenizer.py stay
in sync. Paths (where data/checkpoints/logs live, incl. Google Drive on
Colab) are handled separately in utils.setup_dirs() so this file only
holds numbers.
"""

# --- Tokenizer ---
NUM_MERGES = 4000          # BPE merges learned by train_tokenizer.py
VOCAB_SIZE = None          # filled in at runtime from the trained tokenizer

# --- Model architecture ---
CONTEXT_LENGTH = 128
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1

# --- Data ---
DATASET_STRIDE = CONTEXT_LENGTH   # window stride when building training examples
VAL_SPLIT = 0.1

# --- Optimization ---
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 200
GRAD_CLIP = 1.0
MAX_BATCHES_PER_EPOCH = 1000   # cap so an epoch is a bounded amount of work
USE_AMP = True                 # mixed precision on CUDA

# --- Logging / checkpointing ---
PRINT_EVERY = 25
SAVE_EVERY_EPOCH = True
PROJECT_NAME = "gpt1"          # sub-folder name under Drive / ./runs

# --- Generation ---
SAMPLE_FROM_K = 50
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_NEW_TOKENS = 200

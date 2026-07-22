from tokenizer import BPETokenizer
from dataset import GPTDataset
from torch.utils.data import DataLoader
from model import GPT
from config import *
    
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

# Load tokenizer
tokenizer = BPETokenizer()

tokenizer.load("tokenizer.json")

ids = tokenizer.encode(text, add_special_tokens=True)

dataset = GPTDataset(ids, block_size=128)

loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

vocab_size = len(tokenizer.token_to_id)

model = GPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

x, y = next(iter(loader))

logits = model(x)

print(x.shape)

print(logits.shape)
from tokenizer import BPETokenizer
from dataset import GPTDataset
from torch.utils.data import DataLoader
from model import GPTEmbedding
    
with open("input.txt", encoding="utf-8") as f:
    text = f.read()


# Load tokenizer
tokenizer = BPETokenizer()

# tokenizer.train(text, num_merges=1500)

tokenizer.load("tokenizer.json")

ids = tokenizer.encode(text, add_special_tokens=True)

dataset = GPTDataset(ids, block_size=128)

loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

vocab_size = len(tokenizer.token_to_id)

embedding = GPTEmbedding(
    vocab_size=vocab_size,
    embedding_dim=768,
    context_length=128
)

x, y = next(iter(loader))

out = embedding(x)

print(out.shape)
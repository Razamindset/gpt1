import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tokenizer import BPETokenizer

class GPTDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
tokenizer.train(text, num_merges=1500)

ids = tokenizer.encode(text, add_special_tokens=False)

dataset = GPTDataset(ids, block_size=128)

loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

xb, yb = next(iter(loader))
print(xb.shape)
print(yb.shape)
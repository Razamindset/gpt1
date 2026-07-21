import torch
from torch.utils.data import Dataset

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
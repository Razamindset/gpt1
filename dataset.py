import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, token_ids, block_size, stride=1):
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride

    def __len__(self):
        available_windows = len(self.token_ids) - self.block_size

        if available_windows <= 0:
            return 0

        return ((available_windows - 1) // self.stride) + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.token_ids[start : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

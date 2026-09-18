"""Packed causal-LM dataset over .bin (uint16) token files."""
import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    def __init__(self, bin_path: str, context_length: int, seed: int = 42):
        self.data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
        assert len(self.data) > context_length + 1, f"too little data: {len(self.data)} tokens"
        self.context_length = context_length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data) - self.context_length - 1

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.context_length]
        y = self.data[idx + 1: idx + self.context_length + 1]
        return torch.from_numpy(x.copy()).long(), torch.from_numpy(y.copy()).long()


def get_loaders(train_path: str, val_path: str, context_length: int,
                batch_size: int, seed: int = 42, num_workers: int = 0):
    from torch.utils.data import DataLoader
    train_ds = PackedDataset(train_path, context_length, seed)
    val_ds = PackedDataset(val_path, context_length, seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True)
    return train_loader, val_loader

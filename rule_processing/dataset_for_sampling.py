import torch
from torch.utils.data import Dataset, DataLoader


class Dataset_for_sampling(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return idx
    
    @staticmethod
    def collate_fn(data):
        return torch.tensor(list(data)).view(-1)
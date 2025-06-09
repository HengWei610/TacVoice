# utils/dataloader.py
import torch
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = [{"audio_features": torch.randn(128), "labels": torch.randint(0, 4, (1,)).item()} for _ in range(size)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def load_training_data(config):
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=config.get("batch_size", 32), shuffle=True)

def load_test_data():
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=32, shuffle=False)

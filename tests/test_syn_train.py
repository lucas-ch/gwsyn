import torch
import pytest
import lightning as L
from torch.utils.data import DataLoader, Dataset

class ConstantDataset(Dataset):
    def __init__(self, value, size=16):
        self.data = torch.full((size, 1, 8, 8), float(value)) 
        self.labels = torch.zeros(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MockDataModule(L.LightningDataModule):
    def __init__(self, value, batch_size=8):
        super().__init__()
        self.value = value
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(ConstantDataset(self.value), batch_size=self.batch_size)

class MockModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.seen_epochs = []

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            data_mean = batch[0].mean().item()
            self.seen_epochs.append((self.current_epoch, data_mean))
        
        return torch.nn.functional.mse_loss(self.layer(torch.ones(1, 1)), torch.ones(1, 1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

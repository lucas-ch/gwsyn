import torch
import pytest
import lightning as L
from torch.utils.data import DataLoader, Dataset
from syn_project.train import SequentialDataModule

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

def test_sequential_data_module_switches_correctly():
    """
    Vérifie que le SequentialDataModule passe bien du DM1 au DM2 
    à l'epoch spécifiée.
    """
    switch_epoch = 2
    max_epochs = 4
    dm1 = MockDataModule(value=0.0)
    dm2 = MockDataModule(value=1.0)
    
    combined_dm = SequentialDataModule(dm1, dm2, switch_epoch=switch_epoch)
    model = MockModel()
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        reload_dataloaders_every_n_epochs=1,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )

    trainer.fit(model, combined_dm)

    assert len(model.seen_epochs) == max_epochs, "Le nombre d'epochs enregistrées est incorrect"

    for epoch, mean in model.seen_epochs:
        expected = 0.0 if epoch < switch_epoch else 1.0
        # Utilisation de pytest.approx pour éviter les erreurs de précision flottante
        assert mean == pytest.approx(expected), f"Erreur à l'epoch {epoch}: attendu {expected}, reçu {mean}"
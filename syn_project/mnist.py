import os
from matplotlib import pyplot as plt
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
from lightning import Trainer
from tqdm import tqdm
from shimmer_ssd.modules.domains.visual import VisualDomainModule


ROOT = os.path.abspath(os.path.join(os.getcwd()))
MNIST_DIR = os.path.join(ROOT, "mnist")
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
LATENTS_DIR = os.path.join(MNIST_DIR, "saved_latents")


def create_dir():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(MNIST_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LATENTS_DIR, split), exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

def get_mnist_data():
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [1,32,32] → [3,32,32]
    ])

    full_train = MNIST(root="./data", train=True,  download=True, transform=mnist_transform)
    test_data  = MNIST(root="./data", train=False, download=True, transform=mnist_transform)

    n_train = int(0.8 * len(full_train))
    n_val   = len(full_train) - n_train
    train_data, val_data = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    return train_data, test_data, val_data

def get_data_loader(train_data, test_data, val_data):
        train_loader = DataLoader(VisualDataset(train_data), batch_size=64, shuffle=True,
                            collate_fn=collate_train, num_workers=4)
        val_loader   = DataLoader(VisualDataset(val_data),   batch_size=64, shuffle=False,
                            collate_fn=collate_val,   num_workers=4)
        test_loader  = DataLoader(VisualDataset(test_data),  batch_size=64, shuffle=False,
                            collate_fn=collate_val,   num_workers=4)
        
        return train_loader, test_loader, val_loader


def save_split(dataset, split_name):
    folder = os.path.join(MNIST_DIR, split_name)
    labels = []

    for idx in tqdm(range(len(dataset)), desc=f"Saving {split_name}"):
        image_tensor, label = dataset[idx]
        labels.append([label])

        img_np = (image_tensor[0].numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode='L')
        img.save(os.path.join(folder, f"{idx}.png"))

    labels_np = np.array(labels)
    np.save(os.path.join(MNIST_DIR, f"{split_name}_labels.npy"), labels_np)
    print(f"Labels shape: {labels_np.shape} → {split_name}_labels.npy")

def save_splits(train_data, test_data, val_data):
    save_split(train_data, "train")
    save_split(val_data,   "val")
    save_split(test_data,  "test")

class VisualDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, _ = self.subset[idx]
        return image

def collate_train(batch):
    return {frozenset(["v"]): {"v": torch.stack(batch)}}

def collate_val(batch):
    return {"v": torch.stack(batch)}

def train_mnist_encoder(train_loader, val_loader):
    total_steps = len(train_loader) * 30

    model = VisualDomainModule(
        num_channels=3,
        latent_dim=12,
        ae_dim=64,
        beta=1.0,
        optim_lr=1e-3,
        scheduler_args={"total_steps": total_steps},
    )

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="domain_v_mnist",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_steps=total_steps,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=50,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Checkpoint sauvegardé : {checkpoint_cb.best_model_path}")

# ── 6. Sauvegarde des latents ─────────────────────────────────────────────────
def save_latents_split(model, loader, split_name):
    model.eval().to("cuda")
    all_latents = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {split_name}"):
            x = None

            if split_name != "train":
                x = batch["v"].to("cuda")
            else:
                x = batch[frozenset({'v'})]["v"].to("cuda")
            z = model.encode(x)
            all_latents.append(z.cpu().numpy())

    latents_np = np.concatenate(all_latents, axis=0)
    save_path = os.path.join(LATENTS_DIR, split_name, "domain_v_mnist.npy")
    np.save(save_path, latents_np)
    print(f"{split_name} latents: {latents_np.shape} → {save_path}")
    print(latents_np[:2])  # aperçu des 2 premiers vecteurs

def save_latents(model, train_loader, val_loader, test_loader):
    save_latents_split(model, train_loader, "train")
    save_latents_split(model, val_loader,   "val")
    save_latents_split(model, test_loader,  "test")

def load_model_from_ckpt():
    model = VisualDomainModule.load_from_checkpoint(
        f"{CKPT_DIR}/domain_v_mnist.ckpt"
    )

    return model

def check_model(model, val_loader):
    model.eval().to("cuda")


    model.eval().to("cuda")
    batch = next(iter(val_loader))
    x = batch["v"].to("cuda")

    with torch.no_grad():
        reconstructed = model(x)

    reconstructed = reconstructed.clamp(0, 1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(x[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().detach().numpy())
        for ax in axes[:, i]:
            ax.axis('off')

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruit")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_data, test_data, val_data = get_mnist_data()
    train_loader, test_loader, val_loader = get_data_loader(train_data, test_data, val_data)
    # train_mnist_encoder(train_loader, val_loader)

    model = load_model_from_ckpt()

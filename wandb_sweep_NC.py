# TO DO : HP more salient in wandb visualizations 










import wandb

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from shimmer import DomainModule, LossOutput
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import GlobalWorkspace2Domains, SchedulerArgs
from shimmer.modules.vae import (
    VAE,
    VAEDecoder,
    VAEEncoder,
    gaussian_nll,
    kl_divergence_loss,
)
from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
from shimmer_ssd.dataset.pre_process import TokenizeCaptions
from shimmer_ssd.logging import (
    LogAttributesCallback,
    LogGWImagesCallback,
    LogVisualCallback,
    batch_to_device,
)
from shimmer_ssd.modules.domains import load_pretrained_domains
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
from shimmer_ssd.modules.vae import RAEDecoder, RAEEncoder
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains

# ------------------------------
# 1. Initialize wandb and load config
# ------------------------------

# Initialize wandb (ensure you run wandb.init() only once)
wandb.init(project="SSD_Optimize_val_loss", name="gw_no_color")

# Load configuration and training parameters
config = load_config("./config", use_cli=False, load_files=["train_gw.yaml"])
print(config)
my_hparams = {"temperature": 1, "alpha": 4}

# ------------------------------
# 2. Override config values with wandb sweep parameters (if present)
# ------------------------------

# Global workspace latent dim override
if "latent_dim" in wandb.config:
    config.global_workspace.latent_dim = wandb.config["latent_dim"]

# Override loss coefficients (handling dictionary structure)
if "cycles" in wandb.config:
    config.global_workspace.loss_coefficients["cycles"] = wandb.config["cycles"]
if "contrastives" in wandb.config:
    config.global_workspace.loss_coefficients["contrastives"] = wandb.config["contrastives"]
if "demi_cycles" in wandb.config:
    config.global_workspace.loss_coefficients["demi_cycles"] = wandb.config["demi_cycles"]
if "translations" in wandb.config:
    config.global_workspace.loss_coefficients["translations"] = wandb.config["translations"]

# Override encoder/decoder parameters if specified:
if "encoders_size" in wandb.config:
    config.global_workspace.encoders.hidden_dim = wandb.config["encoders_size"]
if "n_encoders" in wandb.config:
    config.global_workspace.encoders.n_layers = wandb.config["n_encoders"]
if "decoders_size" in wandb.config:
    config.global_workspace.decoders.hidden_dim = wandb.config["decoders_size"]
if "n_decoders" in wandb.config:
    config.global_workspace.decoders.n_layers = wandb.config["n_decoders"]

# Override training parameters from train_gw.yaml
if "weight_decay" in wandb.config:
    config.training.optim.weight_decay = wandb.config["weight_decay"]
if "max_lr" in wandb.config:
    config.training.optim.max_lr = wandb.config["max_lr"]
if "max_steps" in wandb.config:
    config.training.max_steps = wandb.config["max_steps"]

# Override hyperparameters defined in python
if "temperature" in wandb.config:
    my_hparams["temperature"] = wandb.config["temperature"]
if "alpha" in wandb.config:
    my_hparams["alpha"] = wandb.config["alpha"]

# ------------------------------
# 3. Update remaining configuration as before
# ------------------------------


# define the new learning rate to match the ratio 

config.training.optim.lr = config.training.optim.lr * 3.1 / sum(config.global_workspace.loss_coefficients.values())


checkpoint_path = Path("./checkpoints")
config.domain_proportions = {
    frozenset(["v"]): 1.0,
    frozenset(["attr"]): 1.0,
    frozenset(["v", "attr"]): 1.0,
}

config.domains = [
    LoadedDomainConfig(
        domain_type=DomainModuleVariant.v_latents,
        checkpoint_path=checkpoint_path / "domain_v.ckpt",
    ),
    LoadedDomainConfig(
        domain_type=DomainModuleVariant.attr_legacy_no_color,
        checkpoint_path=checkpoint_path / "domain_attr.ckpt",
        args=my_hparams,
    ),
]

config.domain_data_args["v_latents"]["presaved_path"] = "domain_v.npy"
# The default latent dim might have been overridden above
# config.global_workspace.latent_dim = 12

import torch
from torch.utils.data import default_collate

def custom_collate(batch):
    result = default_collate(batch)
    
    # Check if we need to modify the second tensor in attr list
    if (isinstance(result, dict) and "attr" in result and 
        isinstance(result["attr"], list) and len(result["attr"]) >= 2 and
        isinstance(result["attr"][1], torch.Tensor) and result["attr"][1].size(-1) >= 4):
        
        # Remove the last 3 values from the tensor
        result["attr"][1] = result["attr"][1][..., :-3]
    
    return result

domain_classes = get_default_domains(["v_latents", "attr"])

data_module = SimpleShapesDataModule(
    config.dataset.path,
    domain_classes,
    config.domain_proportions,
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    seed=config.seed,
    domain_args=config.domain_data_args,
    collate_fn=custom_collate
)

# ------------------------------
# 4. Load pretrained domains and create global workspace
# ------------------------------

domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
    config.domains,
    config.global_workspace.latent_dim,
    config.global_workspace.encoders.hidden_dim,
    config.global_workspace.encoders.n_layers,
    config.global_workspace.decoders.hidden_dim,
    config.global_workspace.decoders.n_layers,
)

def get_scheduler(optimizer: Optimizer) -> OneCycleLR:
    return OneCycleLR(optimizer, config.training.optim.max_lr, config.training.max_steps)

global_workspace = GlobalWorkspace2Domains(
    domain_modules,
    gw_encoders,
    gw_decoders,
    config.global_workspace.latent_dim,
    config.global_workspace.loss_coefficients,
    config.training.optim.lr,
    config.training.optim.weight_decay,
    scheduler=get_scheduler,
)

# ------------------------------
# 5. Setup Logging and Checkpoints
# ------------------------------

from lightning.pytorch.loggers.wandb import WandbLogger

logger_wandb = WandbLogger(name="gw_no_color", project="shimmer-ssd")
logger = logger_wandb
logger_wandb.log_hyperparams(my_hparams)

train_samples = data_module.get_samples("train", 32)
val_samples = data_module.get_samples("val", 32)

# Split the unique group in validation into individual groups for logging
for domains in val_samples:
    for domain in domains:
        val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
    break

(config.default_root_dir / "gw").mkdir(exist_ok=True)

callbacks: list[Callback] = [
    LogGWImagesCallback(
        val_samples,
        log_key="images/val",
        mode="val",
        every_n_epochs=config.logging.log_val_medias_every_n_epochs,
        filter=config.logging.filter_images,
        exclude_colors=True
    ),
    LogGWImagesCallback(
        train_samples,
        log_key="images/train",
        mode="train",
        every_n_epochs=config.logging.log_train_medias_every_n_epochs,
        filter=config.logging.filter_images,
        exclude_colors=True
    ),
    ModelCheckpoint(
        dirpath=config.default_root_dir / "gw" / f"version_{logger.version}",
        filename="{epoch}",
        monitor="val/loss",
        mode="min",
        save_last="link",
        save_top_k=1,
    ),
]

gw_checkpoint = config.default_root_dir / "gw" / f"version_{logger.version}"
print(gw_checkpoint)

trainer = Trainer(
    logger=logger,
    max_steps=config.training.max_steps,
    default_root_dir=config.default_root_dir,
    callbacks=callbacks,
    precision=config.training.precision,
    accelerator=config.training.accelerator,
    devices=config.training.devices,
)

trainer.fit(global_workspace, data_module)
trainer.validate(global_workspace, data_module, "best")

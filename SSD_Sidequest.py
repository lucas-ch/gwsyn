from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
import sys
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
import torch
from torch.utils.data import default_collate
config = load_config("./config", use_cli=False)




########################
exclude_colors = False
# f"gw_{'without' if exclude_colors else 'with'} colors"
project_name = "shimmer-ssd_sidequests" 




###########################

    # check if the path of the executable is /home/alexis/Desktop/.conda_side_quest1_(asymetrictran)/bin/python
if sys.executable == "/home/alexis/Desktop/.conda_side_quest1_(asymetrictran)/bin/python":
    sidequest = 1
    print("Asymetric translation")

    ## Set up the loss coefficients
    config.global_workspace.loss_coefficients = {
        "cycles" : 0,
        "contrastives" : 0.01,
        "demi_cycles" : 1,
        "translations" : 1,
    }
    logger_name = "Asymetrictranslation_SD1"

elif sys.executable == "/home/alexis/Desktop/.conda_side_quest_2/bin/python":
    sidequest = 2
    print("remove loss on colors ")
    ## Set up the loss coefficients
    config.global_workspace.loss_coefficients = {       
        "cycles" : 1,
        "contrastives" : 0.01,
        "demi_cycles" : 1,
        "translations" : 1,
    }
    logger_name = "SD2"
else:
    ## Set up the loss coefficients
    config.global_workspace.loss_coefficients = {
        "cycles" : 1,
        "contrastives" : 0.01,
        "demi_cycles" : 1,
        "translations" : 0.7143139842316819,
    }


#################################config######################################


config = load_config("./config", use_cli=False, load_files=["train_gw.yaml"])
config.domain_data_args["v_latents"]["presaved_path"] = "domain_v.npy"
config.global_workspace.latent_dim = 12


config.domain_proportions = {
    frozenset(["v"]): 1.0,
    frozenset(["attr"]): 1.0,
    frozenset(["v", "attr"]): 1.0,
}


my_hparams = {"temperature":1, "alpha": 1}
checkpoint_path = Path("./checkpoints")


# Set up the global workspace configuration
config.global_workspace.encoders.hidden_dim = 256
config.global_workspace.decoders.hidden_dim = 256
config.global_workspace.decoders.n_layers = 1
config.global_workspace.encoders.n_layers = 2




## Set up the training configuration
config.training.optim.lr = 3e-4
config.training.optim.max_lr = 0.002589508359966693
config.training.optim.weight_decay = 0.00000723146540361476
config.training.max_steps = 100000
config.training.batch_size = 2056
config.training.num_workers = 4
config.training.accelerator = "gpu"
##################################################################################





domain_classes = get_default_domains(["v_latents", "attr"])

# Set up domain configurations based on exclude_colors
attr_variant = DomainModuleVariant.attr_legacy_no_color if exclude_colors else DomainModuleVariant.attr_legacy

config.domains = [
    LoadedDomainConfig(
        domain_type=DomainModuleVariant.v_latents,
        checkpoint_path=checkpoint_path / "domain_v.ckpt",
    ),
    LoadedDomainConfig(
        domain_type=attr_variant,
        checkpoint_path=checkpoint_path / "domain_attr.ckpt",
        args=my_hparams,
    ),
]





def custom_collate(batch):
    result = default_collate(batch)
    
    # Check if we need to modify the second tensor in attr list
    if (isinstance(result, dict) and "attr" in result and 
        isinstance(result["attr"], list) and len(result["attr"]) >= 2 and
        isinstance(result["attr"][1], torch.Tensor) and result["attr"][1].size(-1) >= 4):
        
        # Remove the last 3 values from the tensor
        result["attr"][1] = result["attr"][1][..., :-3]
    
    return result








# Create data module with optional collate_fn
data_module = SimpleShapesDataModule(
    config.dataset.path,
    domain_classes,
    config.domain_proportions,
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    seed=config.seed,
    domain_args=config.domain_data_args,
    **({"collate_fn": custom_collate} if exclude_colors else {})
)




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






from lightning.pytorch.loggers.wandb import WandbLogger


logger_wandb = WandbLogger(name=logger_name, project=project_name)


logger = logger_wandb

logger_wandb.log_hyperparams(my_hparams)


train_samples = data_module.get_samples("train", 32)
val_samples = data_module.get_samples("val", 32)


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
        exclude_colors=exclude_colors
    ),

    LogGWImagesCallback(
        train_samples,
        log_key="images/train",
        mode="train",
        every_n_epochs=config.logging.log_train_medias_every_n_epochs,
        filter=config.logging.filter_images,
        exclude_colors=exclude_colors
    ),

    ModelCheckpoint(
        dirpath=config.default_root_dir / "gw" / f"version_color{exclude_colors}_{logger.version}",
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


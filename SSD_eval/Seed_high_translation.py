import torch
from torch.utils.data import default_collate
from shimmer_ssd.config import load_config
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
import numpy as np
from torch.utils.data import Subset
from typing import Any, Dict, List, Tuple
from simple_shapes_dataset import SimpleShapesDataset
import pandas as pd
from simple_shapes_dataset import SimpleShapesDataset, get_default_domains
from torch.utils.data import Subset # Pour sélectionner des indices spécifiques
# from torch.utils.data import default_collate # déjà importé
from pathlib import Path
from torchvision.transforms import Compose # Pour combiner les transformations
from torchvision.transforms import ToTensor # Pour utiliser ToTensor
import numpy as np # déjà importé
import time

def custom_collate_factory(exclude_colors: bool):
    """Returns a collate function that optionally removes color info."""
    if not exclude_colors:
        return default_collate

    def custom_collate(batch):
        """Collate function that removes the last 3 attrs (assumed colors)."""
        result = default_collate(batch)
        # Check if we need to modify the second tensor in attr list
        if (isinstance(result, dict) and "attr" in result and
            isinstance(result["attr"], list) and len(result["attr"]) >= 2 and
            isinstance(result["attr"][1], torch.Tensor) and result["attr"][1].size(-1) >= 4):
            # Remove the last 3 values from the tensor
            result["attr"][1] = result["attr"][1][..., :-3]
        return result
    return custom_collate

def init_weights(m: nn.Module, seed: int):
    """Applies Kaiming Normal initialization to Linear layers."""
    if isinstance(m, nn.Linear):
        # Infer device from the weight tensor itself
        device = m.weight.device
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu',
                                generator=torch.Generator(device=device).manual_seed(seed))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def setup_data_module(config, exclude_colors=True):
    """
    Set up the data module for training.
    
    Args:
        config: Configuration with dataset parameters
        
    Returns:
        SimpleShapesDataModule: Configured data module
    """
    from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
    
    
    domain_classes = get_default_domains(["v_latents", "attr"])
    
    return SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=custom_collate_factory(exclude_colors)
    )


def setup_global_workspace(config, hparams, exclude_colors=True, apply_custom_init=True, load_from_checkpoint=True):
    """
    Set up the global workspace model.
    
    Args:
        config: Configuration with model parameters
        hparams: Hyperparameters dictionary
        
    Returns:
        tuple: (global_workspace, domain_modules)
    """
    from pathlib import Path
    from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
    from shimmer_ssd.modules.domains import load_pretrained_domains
    from shimmer.modules.global_workspace import GlobalWorkspace2Domains
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.optim.optimizer import Optimizer
    
    # Set up domain configurations
    checkpoint_path = Path("./checkpoints")
    domains = [
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.v_latents,
            checkpoint_path= "/mnt/HD2/alexis_data/checkpoints_backup/checkpoints/domain_v.ckpt"
        ),
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.attr_legacy_no_color if exclude_colors else DomainModuleVariant.attr_legacy,
            checkpoint_path=checkpoint_path / "domain_attr.ckpt",
            args=hparams,
        ),
    ]
    
    # Create scheduler function
    def get_scheduler(optimizer: Optimizer, scheduler_type: str = "onecycle"):
        if scheduler_type == "onecycle":
            return OneCycleLR(optimizer, config.training.optim.max_lr, 
                          int(config.training.max_steps), 
                          pct_start=config.training.optim.pct_start, div_factor=.38, final_div_factor=5)
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.training.max_steps
                )
        elif scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=config.training.max_steps,
                eta_min=config.training.optim.lr / 100
            )
        else:
            print(f"Scheduler type {scheduler_type} not supported")
            return None
    # Load domains and create GW
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )
    
    # Adjust learning rate based on loss coefficients
    # lr = config.training.optim.lr * 3.1 / sum(config.global_workspace.loss_coefficients.values()) #### TODO: check if this is correct
    lr = config.training.optim.lr
    # Create global workspace
    

    if apply_custom_init:
        # Only apply initialization to GW encoders and decoders, not to the pretrained visual module
        for modality in gw_encoders:
            encoder = gw_encoders[modality]
            encoder.apply(lambda m: init_weights(m, config.seed))
        for modality in gw_decoders:
            decoder = gw_decoders[modality]
            decoder.apply(lambda m: init_weights(m, config.seed))

    global_workspace = GlobalWorkspace2Domains(
        domain_modules,
        gw_encoders,
        gw_decoders,
        config.global_workspace.latent_dim,
        config.global_workspace.loss_coefficients,
        lr,
        config.training.optim.weight_decay,
        scheduler=get_scheduler,
    )
    
    # Load from checkpoint if provided
    if load_from_checkpoint:
        print(f"Loading model from checkpoint: {load_from_checkpoint}")
        # Load default weights
        CHECKPOINT_PATH = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"
        global_workspace = GlobalWorkspace2Domains.load_from_checkpoint(
        CHECKPOINT_PATH,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )
        # Reset the optimizer and scheduler
        global_workspace.optimizer = None
        global_workspace.scheduler = None
        # Reset the callbacks
        global_workspace.callbacks = None
        
        # Reset the training state
        print(f"Loaded default weights from {CHECKPOINT_PATH}")
    
    return global_workspace, domain_modules


def setup_logger_and_callbacks(config, 
                               data_module, 
                               experiment_name="gw_no_color", 
                               project_name="shimmer-ssd", 
                               exclude_colors=True):
    """
    Set up logging and callbacks for training.
    
    Args:
        config: Configuration with logging parameters
        data_module: Data module to get samples from
        experiment_name: Name for the wandb experiment
        
    Returns:
        tuple: (logger, callbacks, checkpoint_dir)
    """
    from lightning.pytorch.loggers.wandb import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from shimmer_ssd.logging import LogGWImagesCallback
    
    output_dir = config.default_root_dir / "training_logs" / project_name / experiment_name

    # Set up logger
    logger = WandbLogger(
            name=experiment_name,
            project=project_name,
            save_dir=output_dir,
            log_model=False, # Usually handled by ModelCheckpoint
            
        )
    

    # Get samples for visualization
    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    
    
    # Split validation samples
    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
        break
    
    # Create checkpoint directory
    # run_version = logger.version if logger and hasattr(logger, 'version') else 'unknown_version'
    version_dir = output_dir / "checkpoints"
    print(f"Model checkpoints will be saved to: {version_dir}")
    
    # Set up callbacks
    callbacks = [
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
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=version_dir,
            filename="{epoch}",
            monitor="val/loss",
            mode="min",
            save_last="link",
            save_top_k=1,
        ),
    ]
    
    return logger, callbacks, version_dir


def train_global_workspace(config, custom_hparams=None, experiment_name="debugging", project_name="shimmer-ssd_debugging", apply_custom_init=True, exclude_colors=True, load_from_checkpoint=True):
    """
    Train a global workspace model with the given configuration.
    
    Args:
        config: Configuration object with model and training parameters
        custom_hparams: Optional dictionary of hyperparameters to override defaults
        experiment_name: Name for the wandb experiment
        
    Returns:
        tuple: (trained_model, checkpoint_path)
    """
    from lightning.pytorch import Trainer

    
    # Initialize default hyperparameters
    hparams = {"temperature": 1, "alpha": 1}
    if custom_hparams:
        hparams.update(custom_hparams)
    
    # 1. Set up data module
    data_module = setup_data_module(config, exclude_colors=exclude_colors)
    
    # 2. Set up global workspace
    global_workspace, _ = setup_global_workspace(config, hparams, exclude_colors=exclude_colors, apply_custom_init=apply_custom_init, load_from_checkpoint=load_from_checkpoint)
    
    # 3. Set up logger and callbacks
    logger, callbacks, checkpoint_dir = setup_logger_and_callbacks(
        config, data_module, experiment_name, project_name, exclude_colors=exclude_colors
    )
    
    # Log hyperparameters
    hparams_to_log = {
        # Domain HParams
        **hparams, # Logs temperature and alpha
        # Model Architecture
        "encoder_size": config.global_workspace.encoders.hidden_dim,
        "decoder_size": config.global_workspace.decoders.hidden_dim,
        "encoder_layers": config.global_workspace.encoders.n_layers,
        "decoder_layers": config.global_workspace.decoders.n_layers,
        "latent_dim": config.global_workspace.latent_dim,
        # Optimizer / Training
        "lr_base": config.training.optim.lr, # Log the base LR from config
        "max_lr": config.training.optim.max_lr,
        "weight_decay": config.training.optim.weight_decay,
        "max_steps": config.training.max_steps,
        "batch_size": config.training.batch_size,
        # Other settings
        "seed": config.seed, # Log seed if set in config
        "exclude_colors": exclude_colors # Logging the setting used for this run
    }
    logger.log_hyperparams(hparams_to_log)
    
    # 4. Create trainer
    trainer = Trainer(
        logger=logger,
        max_steps=config.training.max_steps,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        gradient_clip_val=1.0,  # Set your desired clipping value here
        gradient_clip_algorithm="value",
    )
    
    # 5. Train and validate
    trainer.fit(global_workspace, data_module)
    trainer.validate(global_workspace, data_module, "best")

    wandb.finish()

    
    return global_workspace, checkpoint_dir

torch.autograd.set_detect_anomaly(True)
# Ensuite, lancez votre entraînement. Quand un NaN apparaît dans un gradient, PyTorch lèvera une erreur
# et vous montrera la stack trace de l'opération responsable *dans le backward pass*.
if __name__ == "__main__":

    config = load_config("./config", use_cli=False, load_files=["base_params.yaml"])
    config.seed = 1
    # Set up alpha and temperature
    custom_hparams = {
        "temperature": 1,
        "alpha": 1
    }
    config.training.max_steps = 300000
    config.global_workspace.loss_coefficients["translations"] = 10
    # Train the model with optional custom hyperparameters
    model, checkpoint_path = train_global_workspace(
        config, 
        custom_hparams=custom_hparams,
        project_name="Removing_colors",
        experiment_name="(seed 1) H_T (10) v5",
        apply_custom_init=True,
        exclude_colors=True,
        load_from_checkpoint=False
    )







# import multiprocessing



    
#     # Chargement de la configuration
# config = load_config("./config", use_cli=False, load_files=["base_params.yaml"])
# config.training.max_steps = 300000  # Valeur réduite pour le test
# # Configuration des hyperparamètres
# custom_hparams = {
#     "temperature": 1,
#     "alpha": 1
# }

# experiment_name= "High alpha 5 (color) - corrected v5 (Logsoftmax --> exp --> log)"


# apply_custom_init = False


# def run_training_with_colors():
#     """Lance l'entraînement avec couleurs"""
#     print("=== DÉMARRAGE DE L'ENTRAÎNEMENT AVEC COULEURS ===")
    
#     # Chargement de la configuration
#     config = load_config("./config", use_cli=False, load_files=["base_params.yaml"])
#     config.seed = 1
    
    
#     # Configuration des étapes d'entraînement

#     # Entraînement du modèle
#     model, checkpoint_path = train_global_workspace(
#         config,
#         custom_hparams=custom_hparams,
#         project_name="Removing_colors",
#         experiment_name=experiment_name,
#         apply_custom_init=apply_custom_init,
#         exclude_colors=True,  # Avec couleurs
#         load_from_checkpoint=False
#     )
    
#     print("=== FIN DE L'ENTRAÎNEMENT AVEC COULEURS ===")
#     return checkpoint_path

# def run_training_without_colors():
#     """Lance l'entraînement sans couleurs"""
#     print("=== DÉMARRAGE DE L'ENTRAÎNEMENT SANS COULEURS ===")
    
#     # Importation des modules nécessaires

#     config.seed = 1
    

    
#     # Configuration des étapes d'entraînement
#     config.training.max_steps = 10000  # Valeur réduite pour le test
    
#     # Entraînement du modèle
#     model, checkpoint_path = train_global_workspace(
#         config,
#         custom_hparams=custom_hparams,
#         project_name="Removing_colors",
#         experiment_name=project_name,
#         apply_custom_init=apply_custom_init,
#         exclude_colors=True,  # Sans couleurs
#         load_from_checkpoint=False
#     )
    
#     print("=== FIN DE L'ENTRAÎNEMENT SANS COULEURS ===")
#     return checkpoint_path
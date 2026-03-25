import os

from lightning import LightningDataModule, Trainer
import torch
from torch.utils.data import default_collate
from shimmer_ssd.config import Config, load_config
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, Callback
from .utils_train import save_training_params_pickle, get_experiment_name, get_project_root
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

ROOT_PATH = get_project_root()
REGULAR_DATASET_PATH = f"{ROOT_PATH}/simple_shapes_dataset_biased_00"

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

def setup_data_module(data_path, config:Config, exclude_colors=True):
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
        data_path,
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=custom_collate_factory(exclude_colors),
    )

class SequentialDataModule(LightningDataModule):
    def __init__(self, data_module_1:LightningDataModule, data_module_2:LightningDataModule, switch_epoch:int):
        super().__init__()
        self.data_module_1 = data_module_1
        self.data_module_2 = data_module_2
        self.switch_epoch = switch_epoch

    def setup(self, stage=None):
        self.data_module_1.setup(stage)
        self.data_module_2.setup(stage)

    def train_dataloader(self):
        epoch = self.trainer.current_epoch
        if self.trainer.current_epoch < self.switch_epoch:
            print(f"--- [DEBUG] Chargement TrainLoader PHASE 1 (Epoch {epoch}) ---")
            return self.data_module_1.train_dataloader()
        else:
            print(f"--- [DEBUG] Chargement TrainLoader PHASE 2 (Epoch {epoch}) ---")
            return self.data_module_2.train_dataloader()

    def val_dataloader(self):
        if self.trainer.current_epoch < self.switch_epoch:
            return self.data_module_1.val_dataloader()
        else:
            return self.data_module_2.val_dataloader()

class CustomFlexibleCheckpoint(Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # +1 car l'index commence à 0
        
        should_save = False

        if epoch in [1, 3, 5, 10, 15, 20, 30, 40, 50, 100]:
            should_save = True
            
        # 3. Après 100, toutes les 50 epochs (150, 200, etc.)
        elif epoch > 100 and epoch % 50 == 0:
            should_save = True

        if should_save:
            ckpt_path = f"{self.dirpath}/save-epoch={epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)

def setup_global_workspace(config, hparams, exclude_colors=True, apply_custom_init=True, load_from_checkpoint=True, gw_checkpoint_path=None):
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
    checkpoint_path = Path(f"{ROOT_PATH}/checkpoints")
    domains = [
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.v_latents,
            checkpoint_path= checkpoint_path / "domain_v.ckpt"
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
    if load_from_checkpoint and gw_checkpoint_path is not None:
        print(f"Loading model from checkpoint: {gw_checkpoint_path}")
        # Load default weights
        CHECKPOINT_PATH = gw_checkpoint_path
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
                               experiment_name="gw_no_color", 
                               project_name="shimmer-ssd"):
    """
    Set up logging and callbacks for training.
    
    Args:
        config: Configuration with logging parameters
        experiment_name: Name for the wandb experiment
        
    Returns:
        tuple: (logger, callbacks, checkpoint_dir)
    """
    
    output_dir = config.default_root_dir / project_name / experiment_name

    # Set up logger
    logger = WandbLogger(
            name=experiment_name,
            project=project_name,
            save_dir=output_dir,
            log_model=False, # Usually handled by ModelCheckpoint
            
        )
            
    # Create checkpoint directory
    # run_version = logger.version if logger and hasattr(logger, 'version') else 'unknown_version'
    version_dir = output_dir / "checkpoints"
    print(f"Model checkpoints will be saved to: {version_dir}")
    
    # Set up callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=version_dir,
            filename="{epoch}",
            monitor="val/loss",
            mode="min",
            save_last="link",
            save_top_k=1,
        ),
        CustomFlexibleCheckpoint(dirpath=version_dir)    
        ]
    
    return logger, callbacks, version_dir


def train_global_workspace(
    config:Config,
    custom_hparams=None,
    experiment_name="debugging",
    project_name="shimmer-ssd_debugging",
    apply_custom_init=True,
    exclude_colors=True,
    load_from_checkpoint=True,
    switch_epoch=0):
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
    
    data_module = None
    if switch_epoch == 0:
        data_module_1 = setup_data_module(REGULAR_DATASET_PATH, config, exclude_colors=exclude_colors)
        data_module_2 = setup_data_module(REGULAR_DATASET_PATH, config, exclude_colors=exclude_colors)
        data_module = SequentialDataModule(data_module_1, data_module_2, switch_epoch=switch_epoch)
    if switch_epoch>0:
        data_module_1 = setup_data_module(config.dataset.path, config, exclude_colors=exclude_colors)
        data_module_2 = setup_data_module(REGULAR_DATASET_PATH, config, exclude_colors=exclude_colors)
        data_module = SequentialDataModule(data_module_1, data_module_2, switch_epoch=switch_epoch)

    # 2. Set up global workspace
    global_workspace, _ = setup_global_workspace(config, hparams, exclude_colors=exclude_colors, apply_custom_init=apply_custom_init, load_from_checkpoint=load_from_checkpoint)
    
    # 3. Set up logger and callbacks
    logger, callbacks, checkpoint_dir = setup_logger_and_callbacks(config, experiment_name, project_name)
    
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
        "exclude_colors": exclude_colors, # Logging the setting used for this run
        "switch_epoch": switch_epoch
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
        reload_dataloaders_every_n_epochs=1
    )
    
    # 5. Train and validate
    trainer.fit(global_workspace, data_module)
    trainer.validate(global_workspace, data_module, "best")

    wandb.finish()

    
    return global_workspace, checkpoint_dir

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=["high_cycles.yaml"])
    
    project_name = "syn"
    condition = "test"
    data = "biased_80"
    switch_epoch = 600

    experiment_name = get_experiment_name(condition, data, switch_epoch)
    experiment_name = f"{condition}_{data}"

    if switch_epoch > 0:
        experiment_name = f"{experiment_name}_switch_{switch_epoch}"
    
    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    exclude_colors = False if condition == "control" else True
    apply_custom_init = True
    config.seed = 0

    custom_hparams = {
        "temperature": 1,
        "alpha": 1
    }

    config.training.max_steps = 150000

    log_training_params = {
        "experiment_name": experiment_name,
        "exclude_colors": exclude_colors,
        "apply_custom_init": apply_custom_init,
        "config": config,
        "custom_hparams": custom_hparams,
        "swith_epoch": switch_epoch
    }

    save_training_params_pickle(log_training_params, project_name, experiment_name)

    model, checkpoint_path = train_global_workspace(
        config,
        custom_hparams=custom_hparams, 
        project_name=project_name,
        experiment_name=experiment_name,
        apply_custom_init=apply_custom_init,
        exclude_colors=exclude_colors,
        load_from_checkpoint=False,
        switch_epoch=switch_epoch
    )
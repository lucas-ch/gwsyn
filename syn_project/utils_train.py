from datetime import datetime
import os
import pickle
from pathlib import Path

import wandb
import torch
from torch import nn
from torch.utils.data import default_collate
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR
from torch.optim.optimizer import Optimizer

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from shimmer_ssd.config import Config, LoadedDomainConfig, DomainModuleVariant
from shimmer_ssd.modules.domains import load_pretrained_domains

from .utils_attention import *
from .utils_custom_gw import *
from .utils_color_analysis import *

def get_project_root():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

ROOT_PATH = get_project_root()
REGULAR_DATASET_PATH = f"{ROOT_PATH}/simple_shapes_dataset_biased_00"

DOMAIN_DEFAULT_CHECKPOINT = "domain_attr.ckpt"
DOMAIN_V_CHECKPOINT = "domain_v.ckpt"

DOMAIN_CONFIGS = {
    "attr":          lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.attr_legacy_no_color if excl else DomainModuleVariant.attr_legacy,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
    "v_latents":     lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.v_latents,
                         checkpoint_path=cp / DOMAIN_V_CHECKPOINT),
    "color":         lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.color,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
    "cat":           lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.cat,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
    "position":      lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.position,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
    "positioncolor": lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.positioncolor,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
    "action":        lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.action,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
    "task":          lambda cp, hp, excl: LoadedDomainConfig(
                         domain_type=DomainModuleVariant.task,
                         checkpoint_path=cp / DOMAIN_DEFAULT_CHECKPOINT,
                         args=hp),
}

class CustomFlexibleCheckpoint(Callback):
    def __init__(self, project_name, experiment_name, dirpath, switch_epoch=None):
        super().__init__()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.dirpath = dirpath
        self.switch_epoch = switch_epoch

        os.makedirs(dirpath, exist_ok=True)

    def save_checkpoint(self, trainer:Trainer):
        epoch = trainer.current_epoch + 1
        should_save = False

        if epoch < 10:
            should_save = True

        else:
                if epoch in [1, 10, 20, 40, 60, 80, 100]:
                    should_save = True
                elif epoch > 100 and epoch % 50 == 0:
                    should_save = True

        if should_save:
            ckpt_path = f"{self.dirpath}/save-epoch={epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)


    def on_train_epoch_end(self, trainer, pl_module):
        self.save_checkpoint(trainer)

def get_training_params(project_name, experiment_name):
    training_params = load_training_params_pickle(project_name, experiment_name)
    return training_params

def get_global_workspace(project_name, experiment_name, checkpoint_path=None, epoch=0, modules=['attr', 'v_latents']):
    root_path = get_project_root()
    training_params = get_training_params(project_name, experiment_name)
    exclude_colors = training_params["exclude_colors"]
    
    gw_checkpoint_path = f"{root_path}/checkpoints/{project_name}/{experiment_name}/checkpoints/last.ckpt"
    if checkpoint_path is not None:
        gw_checkpoint_path = checkpoint_path
    if epoch > 0:
        gw_checkpoint_path = f"{root_path}/checkpoints/{project_name}/{experiment_name}/checkpoints/save-epoch={epoch}.ckpt"

    config = training_params["config"]
    hparams = training_params["hparams"] if "hparams" in training_params else {"temperature": 1, "alpha": 1}
    apply_custom_init = training_params["apply_custom_init"]
    attention_tree_config = training_params["attention_tree_config"] if "hparams" in training_params else None

    # Construire les arguments nécessaires à MyGlobalWorkspace
    global_workspace, domain_modules = setup_global_workspace(
        config,
        hparams,
        exclude_colors,
        apply_custom_init,
        load_from_checkpoint=False,  # Ne pas charger ici
        modules=modules,
        attention_tree_config=attention_tree_config
    )

    # Charger directement via Lightning, qui gère les shape mismatches proprement
    global_workspace = MyGlobalWorkspace.load_from_checkpoint(
        gw_checkpoint_path,
        strict=False,  # Ignore les clés manquantes ou en shape mismatch
        map_location="cpu",
        # Passer les arguments du constructeur
        domain_mods=global_workspace.domain_mods,
        gw_encoders=global_workspace.gw_mod.gw_encoders,
        gw_decoders=global_workspace.gw_mod.gw_decoders,
        workspace_dim=config.global_workspace.latent_dim,
        loss_coefs=config.global_workspace.loss_coefficients,
        attention_tree_config=attention_tree_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_workspace.to(device)

    return global_workspace

def init_weights(m: nn.Module, seed: int):
    """Applies Kaiming Normal initialization to Linear layers."""
    if isinstance(m, nn.Linear):
        # Infer device from the weight tensor itself
        device = m.weight.device
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu',
                                generator=torch.Generator(device=device).manual_seed(seed))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def save_training_params_pickle(config, project_name, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    root_dir = Path.cwd()
    log_dir = root_dir / "checkpoints" / project_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = log_dir / f"config_{timestamp}.pkl"

    with open(file_path, 'wb') as f:
        pickle.dump(config, f)
    
    return file_path

def load_training_params_pickle(project_name, experiment_name, file_path=None):
    if file_path:
        target_path = Path(file_path)
    else:
        root_dir = get_project_root()
        log_dir = root_dir / "checkpoints" / project_name / experiment_name
        
        list_of_files = list(log_dir.glob("config_*.pkl"))
        
        if not list_of_files:
            raise FileNotFoundError(f"Aucun fichier pickle trouvé dans {log_dir}")
        
        target_path = max(list_of_files) 
    
    with open(target_path, 'rb') as f:
        return pickle.load(f)

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

def load_global_workspace_from_checkpoint(global_workspace, gw_checkpoint_path):
        print(f"Loading model from checkpoint: {gw_checkpoint_path}")
        
        checkpoint = torch.load(gw_checkpoint_path, "cpu")
        full_state_dict = checkpoint['state_dict']
        
        all_modules = set(global_workspace.gw_mod.gw_encoders.keys())
        gw_prefixes = ("gw_mod.gw_encoders.", "gw_mod.gw_decoders.")
        
        present_modules = {
            key[len(prefix):].split(".", 1)[0]
            for key in full_state_dict
            for prefix in gw_prefixes
            if key.startswith(prefix)
        }
        missing_modules = all_modules - present_modules

        filtered_state_dict = {
            key: value
            for key, value in full_state_dict.items()
            if not key.startswith(gw_prefixes)
            or any(
                key.startswith(f"{prefix}{module}.")
                for prefix in gw_prefixes
                for module in present_modules
            )
        }

        missing_keys, unexpected_keys = global_workspace.load_state_dict(filtered_state_dict, strict=False)
        
        global_workspace.domain_mods["v_latents"].freeze()
        global_workspace.domain_mods["v_latents"].eval()
        for i in global_workspace.gw_mod.gw_encoders:
            global_workspace.gw_mod.gw_encoders[i].eval()
            global_workspace.gw_mod.gw_decoders[i].eval()

        print(f"Modules chargés depuis le checkpoint   : {sorted(present_modules & all_modules)}")
        print(f"Modules en poids par défaut (init)      : {sorted(missing_modules)}")
        if unexpected_keys:
            print(f"⚠️  Clés inattendues ignorées : {unexpected_keys}")
        if missing_keys:
            print(f"Clés non trouvées (poids par défaut conservés) : {missing_keys}")
        print(f"Loaded weights from {gw_checkpoint_path}")

        return global_workspace

def setup_domains(
    modules: list[str],
    checkpoint_path: Path,
    hparams: dict,
    exclude_colors: bool,
) -> list[LoadedDomainConfig]:
    return [
        DOMAIN_CONFIGS[name](checkpoint_path, hparams, exclude_colors)
        for name in modules
        if name in DOMAIN_CONFIGS
    ]

def get_scheduler(optimizer: Optimizer, config, scheduler_type: str = "onecycle"):
    if scheduler_type == "onecycle":
        return OneCycleLR(
            optimizer,
            config.training.optim.max_lr,
            int(config.training.max_steps),
            pct_start=config.training.optim.pct_start,
            div_factor=0.38,
            final_div_factor=5,
        )
    elif scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.training.max_steps,
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_steps,
            eta_min=config.training.optim.lr / 100,
        )
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported")

def build_global_workspace(
    config,
    domain_modules,
    gw_encoders,
    gw_decoders,
    custom_weights,
    noise,
    modules_to_freeze,
    fusion_activation_fn,
    attention_tree_config,
    apply_custom_init: bool,
) -> MyGlobalWorkspace:
    if apply_custom_init:
        for encoder in gw_encoders.values():
            encoder.apply(lambda m: init_weights(m, config.seed))
        for decoder in gw_decoders.values():
            decoder.apply(lambda m: init_weights(m, config.seed))

    gw = MyGlobalWorkspace(
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=config.global_workspace.latent_dim,
        loss_coefs=config.global_workspace.loss_coefficients,
        custom_weights=custom_weights,
        noise=noise,
        modules_to_freeze=modules_to_freeze,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler=lambda opt, stype="onecycle": get_scheduler(opt, config, stype),
        fusion_activation_function=fusion_activation_fn,
        attention_tree_config=attention_tree_config,
    )

    gw.domain_mods["v_latents"].freeze()
    gw.domain_mods["v_latents"].eval()
    return gw

def setup_global_workspace(
    config,
    hparams,
    exclude_colors: bool = True,
    apply_custom_init: bool = True,
    load_from_checkpoint: bool = True,
    gw_checkpoint_path=None,
    custom_weights=None,
    noise=None,
    modules: list[str] = ("attr", "v_latents"),
    modules_to_freeze: list = [],
    fusion_activation_fn=torch.nn.Identity(),
    attention_tree_config=None,
):
    checkpoint_path = Path(get_project_root()) / "checkpoints"

    domains = setup_domains(modules, checkpoint_path, hparams, exclude_colors)
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )

    global_workspace = build_global_workspace(
        config, domain_modules, gw_encoders, gw_decoders,
        custom_weights, noise, modules_to_freeze,
        fusion_activation_fn, attention_tree_config, apply_custom_init,
    )

    if load_from_checkpoint and gw_checkpoint_path is not None:
        load_global_workspace_from_checkpoint(global_workspace, gw_checkpoint_path)

    return global_workspace, domain_modules

def setup_data_module(data_path, config:Config, exclude_colors=True, modules=['attr', 'v_latents']):
    """
    Set up the data module for training.
    
    Args:
        config: Configuration with dataset parameters
        
    Returns:
        SimpleShapesDataModule: Configured data module
    """
    from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
    
    domain_classes = get_default_domains(modules)
    
    return SimpleShapesDataModule(
        data_path,
        domain_classes,
        config.domain_proportions,
        max_train_size=config.max_train_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=custom_collate_factory(exclude_colors),
    )

def setup_logger_and_callbacks(config, 
                               experiment_name="gw_no_color", 
                               project_name="shimmer-ssd",
                               switch_epoch=0):
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
        CustomFlexibleCheckpoint(
            project_name= project_name,
            experiment_name= experiment_name,
            dirpath=version_dir,
            switch_epoch=switch_epoch)    
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
    gw_checkpoint_path=None,
    switch_epoch=0,
    custom_weights=None,
    noise=None,
    modules=['attr', 'v_latents'],
    modules_to_freeze=[],
    fusion_activation_fn=torch.tanh,
    attention_tree_config=None):
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
    
    hparams = {"temperature": 1, "alpha": 1}
    if custom_hparams:
        hparams.update(custom_hparams)
    
    data_module = setup_data_module(config.dataset.path, config, exclude_colors=exclude_colors,modules=modules)

    global_workspace, _ = setup_global_workspace(
        config,
        hparams,
        exclude_colors=exclude_colors,
        apply_custom_init=apply_custom_init,
        load_from_checkpoint=load_from_checkpoint,
        gw_checkpoint_path = gw_checkpoint_path,
        custom_weights=custom_weights,
        noise=noise,
        modules=modules,
        modules_to_freeze=modules_to_freeze,
        fusion_activation_fn=fusion_activation_fn,
        attention_tree_config=attention_tree_config)
    
    logger, callbacks, checkpoint_dir = setup_logger_and_callbacks(config, experiment_name, project_name, switch_epoch)
    callbacks.append(FixAttentionLR(attention_lr=global_workspace.attention_lr, attention_group_idx=1))
    
    hparams_to_log = {
        **hparams,
        "encoder_size": str(config.global_workspace.encoders.hidden_dim),
        "decoder_size": str(config.global_workspace.decoders.hidden_dim),
        "encoder_layers": config.global_workspace.encoders.n_layers,
        "decoder_layers": config.global_workspace.decoders.n_layers,
        "latent_dim": config.global_workspace.latent_dim,
        "lr_base": config.training.optim.lr, # Log the base LR from config
        "max_lr": config.training.optim.max_lr,
        "weight_decay": config.training.optim.weight_decay,
        "max_steps": config.training.max_steps,
        "batch_size": config.training.batch_size,
        "seed": config.seed, # Log seed if set in config
        "exclude_colors": exclude_colors, # Logging the setting used for this run
        "switch_epoch": switch_epoch
    }
    logger.log_hyperparams(hparams_to_log)

    trainer = Trainer(
        logger=logger,
        max_epochs=30,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        gradient_clip_val=1.0,  # Set your desired clipping value here
        gradient_clip_algorithm="value",
        reload_dataloaders_every_n_epochs=1
    )
    
    trainer.fit(global_workspace, data_module)
    trainer.validate(global_workspace, data_module, "best")

    wandb.finish()

    return global_workspace, checkpoint_dir

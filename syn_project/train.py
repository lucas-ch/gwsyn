import os

from lightning import LightningDataModule, Trainer
import torch
from shimmer_ssd.config import Config, load_config
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, Callback
from .utils_notebook import *
from .utils_analyse import *
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

ROOT_PATH = get_project_root()
REGULAR_DATASET_PATH = f"{ROOT_PATH}/simple_shapes_dataset_biased_00"


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
    def __init__(self, project_name, experiment_name, dirpath, switch_epoch=None):
        super().__init__()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.dirpath = dirpath
        self.switch_epoch = switch_epoch

    def save_checkpoint(self, trainer:Trainer):
        epoch = trainer.current_epoch + 1
        should_save = False

        if self.switch_epoch is not None and self.switch_epoch <= epoch < (self.switch_epoch + 100):
            if (epoch - self.switch_epoch) % 10 == 0:
                should_save = True
        else:
            if epoch in [1, 10, 20, 40, 60, 80, 100]:
                should_save = True
            elif epoch > 100 and epoch % 50 == 0:
                should_save = True

        if should_save:
            ckpt_path = f"{self.dirpath}/save-epoch={epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)

    def run_color_analysis(self, trainer:Trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            n_samples = 1000
            data_module = get_data_module(self.project_name,  self.experiment_name)
            test_samples = get_data_samples(data_module, n_samples, split= "test")
            data_translated = get_data_translated(pl_module, test_samples, n_samples)            
            
            masks = get_mask_from_shape_batch(data_translated["train_images"])
            masks_decoded = get_mask_from_shape_batch(data_translated["images_decoded"])

            colors_from_data_attr = get_color_from_attr_batch(data_translated['train_attr']).detach().cpu().numpy()
            colors_from_data_img = get_color_from_img_batch(data_translated["train_images"], masks)
            colors_from_decoded_img = get_color_from_img_batch(data_translated["images_decoded"], masks_decoded)

            categories_from_data_attr = data_translated['train_attr'][:, 0:3].detach().cpu().numpy()

            categories_from_decoded_attr = categorize_decoded_attr(data_translated["attr_decoded"])
            categories_indices = categories_from_decoded_attr.argmax(dim=1).detach().cpu().numpy()

            save_path = os.path.join(self.dirpath, f"stats_epoch_{trainer.current_epoch:03d}.npz")

            np.savez_compressed(
                save_path,
                colors_from_data_attr=colors_from_data_attr,
                colors_from_data_img=colors_from_data_img,
                colors_from_decoded_img=colors_from_decoded_img,
                categories_from_data_attr=categories_from_data_attr,
                categories_from_decoded_attr=categories_from_decoded_attr.detach().cpu().numpy()
            )
        
            fig = plot_original_translated_comparison(data_translated["train_images"], data_translated["images_decoded"])
            output_dir = os.path.join(self.dirpath, "visual_logs")
            os.makedirs(output_dir, exist_ok=True)

            file_name = f"fig_comparison_epoch_{trainer.current_epoch:03d}.png"
            save_path = os.path.join(output_dir, file_name)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

            trainer.logger.experiment.log({
                "visuals/comparison": wandb.Image(fig),
                "epoch": trainer.current_epoch
            }) 
            plt.close(fig)

            categories_indices = categories_from_decoded_attr.argmax(dim=1).detach().cpu().numpy()
            colors_np = np.vstack(colors_from_decoded_img)

            fig = plot_rgb_distribution(colors_np, categories_indices)
            output_dir = os.path.join(self.dirpath, "visual_logs")
            os.makedirs(output_dir, exist_ok=True)

            file_name = f"fig_color_distrib_epoch_{trainer.current_epoch:03d}.png"
            save_path = os.path.join(output_dir, file_name)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

            trainer.logger.experiment.log({
                "visuals/color_distrib": wandb.Image(fig),
                "epoch": trainer.current_epoch
            })

            plt.close(fig)

        pl_module.train()


    def on_train_epoch_end(self, trainer, pl_module):
        self.save_checkpoint(trainer)
        self.run_color_analysis(trainer, pl_module)


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
    switch_epoch = 300

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
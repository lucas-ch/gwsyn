from datetime import datetime
import os
import pickle
from typing import Any, cast
from collections.abc import Mapping

from shimmer import ContrastiveLoss, GWLosses2Domains, GlobalWorkspace2Domains, LatentsDomainGroupsT, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase, SingleDomainSelection, combine_loss
from shimmer_ssd.logging import batch_to_device
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule, mse_loss
from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains

import numpy as np
import torch
from torch import nn


from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import default_collate

from lightning import LightningDataModule, Trainer
import torch
from shimmer_ssd.config import Config
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, Callback
from .utils_color_analysis import *
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import copy

def get_mask_from_shapes_diff(images: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """
    Version différentiable de l'extraction de masque.
    
    Args:
        images (torch.Tensor): Tensor de forme (B, C, H, W)
        temperature (float): Contrôle la "dureté" du seuil. 
                             Plus c'est élevé, plus c'est proche d'un masque binaire.
    
    Returns:
        torch.Tensor: Masque "soft" entre 0 et 1.
    """
    # 1. Conversion en niveaux de gris si nécessaire (moyenne des canaux RGB)
    if images.shape[1] == 3:
        grayscale = images.mean(dim=1, keepdim=True).clone()
    else:
        grayscale = images.clone()

    # 2. Normalisation optionnelle (si vos images ne sont pas déjà en [0, 1])
    # On suppose ici que le fond est à 0 et l'objet est > 0.
    
    # 3. Sigmoid "Hard" : Approximation de (x > 0)
    # On centre la sigmoid légèrement au dessus de 0 (ex: 0.05) pour éviter le bruit de fond
    threshold = 0.1
    masks = torch.sigmoid((grayscale - threshold) * temperature)
    
    return masks

def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()

def get_centroid(mask: torch.Tensor):
    """
    Calcule les coordonnées (x, y) du centre de masse d'un masque.
    mask: Tensor de forme (B, 1, H, W)
    """
    B, C, H, W = mask.shape
    device = mask.device
    
    # Création des grilles de coordonnées normalisées entre 0 et 1
    # grid_x: [[0, 1, 2...], [0, 1, 2...]]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    
    # Somme des intensités pour la normalisation (la "masse" totale)
    # On ajoute un epsilon pour éviter la division par zéro si le masque est noir
    total_mass = mask.sum(dim=(2, 3)) + 1e-8
    
    # Calcul du centre de masse pondéré par l'intensité des pixels
    pos_x = (mask * grid_x).sum(dim=(2, 3)) / total_mass
    pos_y = (mask * grid_y).sum(dim=(2, 3)) / total_mass
    
    # Retourne un tenseur (B, 2) contenant les coordonnées [x, y]
    return torch.cat([pos_x, pos_y], dim=1)

def centroid_loss(m_pred, m_orig):
    """Calcule la distance MSE entre les centres de masse."""
    center_pred = get_centroid(m_pred)
    center_orig = get_centroid(m_orig)
    return F.mse_loss(center_pred, center_orig)

def area_loss(m_pred, m_orig):
    """
    Force la surface totale (nombre de pixels activés) à être identique.
    m_pred, m_orig: (B, 1, H, W)
    """
    # Somme des intensités des pixels (approximation de l'aire)
    area_pred = m_pred.sum(dim=(1, 2, 3))
    area_orig = m_orig.sum(dim=(1, 2, 3))
    
    # On utilise L1 pour une pénalité stable et linéaire
    return F.l1_loss(area_pred, area_orig) / (32 * 32)

def shape_loss(gw_mod: GlobalWorkspace2Domains, domain_latents: LatentsDomainGroupsT, raw_data: RawDomainGroupsT):
    visual_module = cast(VisualLatentDomainModule, gw_mod.domain_mods["v_latents"])
    visual_module.eval()
    with torch.no_grad():
            v_raw = raw_data[frozenset({'v_latents'})]['v_latents']
            x_original = visual_module.decode_images(v_raw)
            target = get_mask_from_shapes_diff(x_original).detach()

    v_raw = raw_data[frozenset({'v_latents'})]['v_latents']
    x_original = visual_module.decode_images(v_raw)
    target = get_mask_from_shapes_diff(x_original)

    latents = domain_latents[frozenset({'v_latents', 'attr'})]
    domain_sources = {'attr': latents['attr']}
    z = gw_mod.encode_and_fuse(domain_sources, SingleDomainSelection())

    gw_decoded_latents = gw_mod.decode(z, domains={'v_latents'})['v_latents']
    x_recons = visual_module.decode_images(gw_decoded_latents)
    prediction = get_mask_from_shapes_diff(x_recons)

    
    centroid_l=centroid_loss(prediction, target)* 100

    loss_output = LossOutput(centroid_l)

    return loss_output 


def traduction_loss(gw_mod: GlobalWorkspace2Domains, domain_latents: LatentsDomainGroupsT, raw_data: RawDomainGroupsT):
    
    domain_name_target = 'v_latents'

    domains = frozenset({'v_latents', 'attr'})
    latents = domain_latents[frozenset({'v_latents', 'attr'})]
    domain_sources = {
                domain: latents[domain]
                for domain in domains
                if domain != domain_name_target
            }
    z = gw_mod.encode_and_fuse(domain_sources, SingleDomainSelection())

    prediction = gw_mod.decode(z, domains = {domain_name_target})[domain_name_target]
    loss_output = LossOutput(F.mse_loss(
        prediction, latents[domain_name_target], reduction="mean"))
    
    return loss_output

def get_project_root():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

ROOT_PATH = get_project_root()
REGULAR_DATASET_PATH = f"{ROOT_PATH}/simple_shapes_dataset_biased_00"

def add_noise(data, noise_params={"mean": 0.0, "std": 0.0}):
    data_copy = copy.deepcopy(data)

    key_fs = frozenset(["v_latents"])
    latents = data_copy[key_fs]["v_latents"]

    noise = torch.randn_like(latents) * noise_params['std'] + noise_params['mean']

    data_copy[key_fs]["v_latents"] = latents + noise
    
    return data_copy


class MyCustomGWLosses(GWLosses2Domains):
    def __init__(self, gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn, custom_weights=None, noise=None):
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.custom_weights = custom_weights
        self.noise=noise

    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        metrics: dict[str, torch.Tensor] = {}

        noisy_data = add_noise(domain_latents, self.noise)

        if "shape_loss" in self.custom_weights.keys() and self.custom_weights["shape_loss"] > 0:
            current_shape_loss = shape_loss(self.gw_mod, domain_latents, raw_data)
            print(current_shape_loss)
            print(current_shape_loss)
            metrics.update({"shape_loss": current_shape_loss.loss})

        # 1. Calcul des métriques de base
        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(noisy_data, raw_data))
        metrics.update(self.translation_loss(noisy_data, raw_data))
        metrics.update(self.contrastive_loss(noisy_data))

#        test = traduction_loss(self.gw_mod, domain_latents, raw_data)
        custom_weights = self.custom_weights

        weighted_losses = []
        for key, weight in custom_weights.items():
            if key in metrics:
                weighted_losses.append(metrics[key] * weight)

        if not weighted_losses:
            # Fallback au cas où aucune clé ne correspond
            return LossOutput(torch.tensor(0.0, device=raw_data.device), metrics)

        # On fait la somme de toutes les pertes pondérées
        custom_loss = torch.stack(weighted_losses).sum()


        return LossOutput(custom_loss, metrics)

class MyGlobalWorkspace(GlobalWorkspace2Domains):
    def __init__(
            self,
            domain_mods,
            gw_encoders,
            gw_decoders,
            workspace_dim,
            loss_coefs,
            custom_weights,
            noise,
            *args,
            **kwargs):
        super().__init__(domain_mods, gw_encoders, gw_decoders, workspace_dim, loss_coefs, *args, **kwargs)

        contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", False
            )
        self.loss_mod = MyCustomGWLosses(
            self.gw_mod, 
            self.selection_mod, 
            self.domain_mods, 
            loss_coefs,
            contrastive_loss,
            custom_weights,
            noise
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

        os.makedirs(dirpath, exist_ok=True)

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
            n_samples = 1 if "test1" in self.experiment_name else 1000
            data_module = get_data_module(self.project_name,  self.experiment_name)
            test_samples = get_data_samples(data_module, n_samples, split= "test")
            data_translated = get_data_translated(pl_module, test_samples, n_samples)            
            
            masks = get_mask_from_shapes(data_translated["train_images"])
            masks_decoded = get_mask_from_shapes(data_translated["images_decoded"])

            colors_from_data_img = get_color_from_images(data_translated["train_images"], masks)
            colors_from_decoded_img = get_color_from_images(data_translated["images_decoded"], masks_decoded)

            categories_from_data_attr = data_translated['train_attr'][:, 0:3].detach().cpu().numpy()

            categories_from_decoded_attr = categorize_decoded_attr(data_translated["attr_decoded"])
            categories_indices = categories_from_decoded_attr.argmax(dim=1).detach().cpu().numpy()

            save_path = os.path.join(self.dirpath, f"stats_epoch_{trainer.current_epoch:03d}.npz")

            np.savez_compressed(
                save_path,
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

class FusionMethod(SelectionBase):
    def __init__(self, n_samples: int = 32, fusion_attr_weight: float = 0.5):
        super().__init__()
        self.n_samples = n_samples
        self.fusion_attr_weight = fusion_attr_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self, domains: Mapping[str, torch.Tensor], encodings_pre_fusion: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # If only one domain, we keep it
        if len(domains) == 1:
            return {domain: torch.ones(self.n_samples).to(self.device) for domain in domains}
        else:
            selection: dict[str, torch.Tensor] = {}
            for domain in domains:
                # We only keep the visual latents
                if domain == "v_latents":
                    selection["v_latents"] = (torch.ones(self.n_samples)*(1 - self.fusion_attr_weight)).to(self.device)
                else:
                    # and set 0 to all other domain
                    selection[domain] = (torch.ones(self.n_samples)*self.fusion_attr_weight).to(self.device)
            return selection


def get_training_params(project_name, experiment_name):
    training_params = load_training_params_pickle(project_name,  experiment_name)
    return training_params

def get_global_workspace(project_name, experiment_name, checkpoint_path=None, epoch=0):
    root_path = get_project_root()

    training_params = get_training_params(project_name,  experiment_name)

    exclude_colors = training_params["exclude_colors"]
    gw_checkpoint_path = f"{root_path}/checkpoints/{project_name}/{experiment_name}/checkpoints/last.ckpt"
    if checkpoint_path is not None: 
        gw_checkpoint_path = checkpoint_path
    if epoch>0:
        gw_checkpoint_path = f"{root_path}/checkpoints/{project_name}/{experiment_name}/checkpoints/save-epoch={epoch}.ckpt"

    config = training_params["config"]
    hparams = training_params["hparams"] if "hparams" in training_params else {"temperature": 1, "alpha": 1}
    apply_custom_init = training_params["apply_custom_init"]

    global_workspace, domain_modules = setup_global_workspace(
        config,
        hparams,
        exclude_colors,
        apply_custom_init,
        load_from_checkpoint = True,
        gw_checkpoint_path = gw_checkpoint_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_workspace.to(device)

    return global_workspace

def get_data_module(project_name,  experiment_name):
    training_params = load_training_params_pickle(project_name,  experiment_name)
    config = training_params["config"]
    exclude_colors = training_params["exclude_colors"]

    domain_classes = get_default_domains(["v_latents", "attr"])

    root_path = get_project_root()

    if str(root_path) in config.dataset.path:
        data_path = f"{config.dataset.path}"
    else:
        data_path = f"{root_path}/{config.dataset.path}"
       
    data_module = SimpleShapesDataModule(
        data_path,
        domain_classes,
        config.domain_proportions,
        config.training.batch_size,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=custom_collate_factory(exclude_colors=exclude_colors),
    )

    return data_module

def get_data_samples(data_module:SimpleShapesDataModule, n_samples:int, split="train", noise = 0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = data_module.get_samples(split, n_samples)
    train_samples = batch_to_device(train_samples, device)

    if noise > 0:
        category = train_samples[frozenset({'attr'})]["attr"][0]
        category = category.float()
        mean = 0.0      # Moyenne du bruit
        std = noise     # Écart-type (plus c'est haut, plus le bruit est fort)

        noise = torch.randn_like(category) * std + mean
        category_noisy =  torch.clamp(category + noise, min=1e-8)

        category_noisy_normalized = category_noisy / (category_noisy.sum(dim=-1, keepdim=True))

        train_samples[frozenset({'attr'})]["attr"][0] = category_noisy_normalized
        train_samples[frozenset({'attr', 'v_latents'})]["attr"][0] = category_noisy_normalized


    return train_samples

@torch.no_grad()
def get_data_translated(global_workspace, train_samples, n_samples=32, fusion_attr_weight=1.0, show_results_fusion=False):
    selection_mod = FusionMethod(n_samples, fusion_attr_weight)

    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])
    train_paired_samples = train_samples[frozenset(["v_latents", "attr"])]

    train_images = visual_module.decode_images(train_paired_samples["v_latents"]).detach().cpu()
    
    train_attr = torch.cat((train_paired_samples["attr"][0], train_paired_samples["attr"][1]), dim=1).detach().cpu()

    unimodal_latents = global_workspace.encode_domains(train_samples)
    gw_latents = global_workspace.encode(unimodal_latents)

    gw_latents_decoded = global_workspace.decode(gw_latents[frozenset({'v_latents', 'attr'})], ["v_latents", "attr"])

    # Extraction et nettoyage
    attr_decoded = gw_latents_decoded["attr"]["attr"]
    images_decoded = visual_module.decode_images(gw_latents_decoded['attr']['v_latents']).detach().cpu()


    if show_results_fusion:
        gw_latents_fusion = global_workspace.encode_and_fuse(unimodal_latents, selection_mod)
        gw_latents_fusion_decoded = global_workspace.decode(gw_latents_fusion, ["v_latents", "attr"])
        attr_fusion_decoded = gw_latents_fusion_decoded[frozenset({'v_latents', 'attr'})]['attr']
        images_fusion_decoded = visual_module.decode_images(gw_latents_fusion_decoded[frozenset({'v_latents', 'attr'})]['v_latents']).detach().cpu()

        result_attr = attr_fusion_decoded
        result_images = images_fusion_decoded
    else:
        result_attr = attr_decoded
        result_images = images_decoded

    torch.cuda.empty_cache()

    return {
        "train_images": train_images,
        "train_attr": train_attr,
        "images_decoded": result_images,
        "attr_decoded": result_attr
    }



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


def get_experiment_name(condition, data, switch_epoch):
    experiment_name = f"{condition}_{data}"

    if switch_epoch > 0:
        experiment_name = f"{experiment_name}_switch_{switch_epoch}"

    return experiment_name

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

def setup_global_workspace(
        config, hparams,
        exclude_colors=True,
        apply_custom_init=True,
        load_from_checkpoint=True,
        gw_checkpoint_path=None,
        custom_weights=None,
        noise=None):
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
    
    root_path = get_project_root()
    # Set up domain configurations
    checkpoint_path = Path(f"{root_path}/checkpoints")
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

    global_workspace = MyGlobalWorkspace(
        domain_mods= domain_modules,
        gw_encoders = gw_encoders,
        gw_decoders = gw_decoders,
        workspace_dim = config.global_workspace.latent_dim,
        loss_coefs = config.global_workspace.loss_coefficients,
        custom_weights=custom_weights,
        noise=noise,
        optim_lr=lr,
        optim_weight_decay = config.training.optim.weight_decay,
        scheduler=get_scheduler,
    )

    global_workspace.domain_mods["v_latents"].freeze()
    global_workspace.domain_mods["v_latents"].eval()
    
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

def get_grid_numpy(samples, nrow=8):
    grid = make_grid(samples, nrow=nrow, pad_value=1).permute(1, 2, 0)
    return grid.detach().cpu().numpy()

def plot_original_translated_comparison(original_images, result_images, max_images=32):
    num_to_show = min(len(original_images), max_images)
    orig_subset = original_images[:num_to_show]
    res_subset = result_images[:num_to_show]    
    
    grid_train = get_grid_numpy(orig_subset)
    grid_decoded = get_grid_numpy(res_subset)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)) 

    ax1.imshow(grid_train)
    ax1.set_title("Images originales")
    ax1.axis('off')

    ax2.imshow(grid_decoded)
    ax2.set_title("Images traduites: attr => GW => v_latents")
    ax2.axis('off')

    plt.tight_layout()
    return fig

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
    switch_epoch=0,
    custom_weights=None,
    noise=None):
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
    global_workspace, _ = setup_global_workspace(
        config,
        hparams,
        exclude_colors=exclude_colors,
        apply_custom_init=apply_custom_init,
        load_from_checkpoint=load_from_checkpoint,
        custom_weights=custom_weights,
        noise=noise)
    
    # 3. Set up logger and callbacks
    logger, callbacks, checkpoint_dir = setup_logger_and_callbacks(config, experiment_name, project_name, switch_epoch)
    
    # Log hyperparameters
    hparams_to_log = {
        # Domain HParams
        **hparams, # Logs temperature and alpha
        # Model Architecture
        "encoder_size": str(config.global_workspace.encoders.hidden_dim),
        "decoder_size": str(config.global_workspace.decoders.hidden_dim),
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

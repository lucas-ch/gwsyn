from datetime import datetime
import os
import pickle
from typing import cast
from collections.abc import Mapping

from shimmer import SelectionBase
from shimmer_ssd.logging import batch_to_device
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains

import io
import math
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import one_hot

from simple_shapes_dataset.cli import generate_image

from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import default_collate

def get_project_root():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

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

import torch

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


CAT2IDX = {"Diamond": 0, "Egg": 1, "Triangle": 2}

def get_image(cat, x, y, size, rot, color_r, color_g, color_b):
    fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
    # The dataset generatoion tool has function to generate a matplotlib shape
    # from the attributes. 
    generate_image(
        ax,
        CAT2IDX[cat],
        [int(x * 18 + 7), int(y * 18 + 7)],
        size * 7 + 7,
        rot * 2 * math.pi,
        np.array([color_r * 255, color_g * 255, color_b * 255]),
        imsize=32,
    )
    ax.set_facecolor("black")
    plt.tight_layout(pad=0)
    # Return this as a PIL Image.
    # This is to have the same dpi as saved images
    # otherwise matplotlib will render this in very high quality
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image

def get_decoded_image(cat, x, y, size, rot, color_r, color_g, color_b, training_params, device, global_workspace):
    exclude_colors = training_params["exclude_colors"]
    category = one_hot(torch.tensor([CAT2IDX[cat]]), 3)
    rotx = math.cos(rot * 2 * math.pi)
    roty = math.sin(rot * 2 * math.pi)
    
    attributes = torch.tensor(
        [[x * 2 - 1, y * 2 - 1, size * 2 - 1, rotx, roty, ]]
    )

    if not exclude_colors:
        attributes = torch.tensor(
            [[x * 2 - 1, y * 2 - 1, size * 2 - 1, rotx, roty, color_r * 2 - 1, color_g * 2 - 1, color_b * 2 - 1]]
        )

    samples = [category.to(device), attributes.to(device)]
    attr_gw_latent = global_workspace.gw_mod.encode({"attr": global_workspace.encode_domain(samples, "attr")})
    gw_latent = global_workspace.gw_mod.fuse(
        attr_gw_latent, {"attr": torch.ones(attr_gw_latent["attr"].size(0)).to(device)}
    )
    decoded_latents = global_workspace.gw_mod.decode(gw_latent)["v_latents"]
    decoded_images = (
        global_workspace.domain_mods["v_latents"]
        .decode_images(decoded_latents)[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    return decoded_images

def plot_interactive(image, decoded_image):
    fig, axes = plt.subplots(1, 2)
    axes[0].set_facecolor("black")
    axes[0].set_title("Original image from attributes")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].imshow(image)

    # normalize the attribute for the global workspace.
    axes[1].imshow(decoded_image)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Translated image through GW")
    plt.show()


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


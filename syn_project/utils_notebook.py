import os
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
from torch.nn.functional import one_hot

from simple_shapes_dataset.cli import generate_image
from .utils_train import load_training_params_pickle
from .train import setup_global_workspace, custom_collate_factory

from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def get_training_params(project_name, experiment_name):
    training_params = load_training_params_pickle(project_name,  experiment_name)
    return training_params

def get_global_workspace(project_name, experiment_name, checkpoint_path=None, epoch=0):
    root_path = os.path.abspath(os.path.join('..'))

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

    root_path = os.path.abspath(os.path.join('..'))
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

def get_data_translated(global_workspace, train_samples, n_samples=32, fusion_attr_weight= 1.0, show_results_fusion=False):
    selection_mod = FusionMethod(n_samples, fusion_attr_weight)

    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])
    train_paired_samples = train_samples[frozenset(["v_latents", "attr"])]

    train_images = visual_module.decode_images(train_paired_samples["v_latents"])
    train_attr = torch.cat((train_paired_samples["attr"][0], train_paired_samples["attr"][1]), dim=1)

    unimodal_latents = global_workspace.encode_domains(train_samples)
    gw_latents = global_workspace.encode(unimodal_latents)
    gw_latents_fusion = global_workspace.encode_and_fuse(unimodal_latents, selection_mod)

    gw_latents_decoded = global_workspace.decode(gw_latents[frozenset({'attr'})], ["v_latents", "attr"])
    gw_latents_fusion_decoded = global_workspace.decode(gw_latents_fusion, ["v_latents", "attr"])

    attr_decoded = gw_latents_decoded["attr"]["attr"]
    images_decoded = visual_module.decode_images(gw_latents_decoded['attr']['v_latents'])

    attr_fusion_decoded = gw_latents_fusion_decoded[frozenset({'v_latents', 'attr'})]['attr']
    images_fusion_decoded = visual_module.decode_images(gw_latents_fusion_decoded[frozenset({'v_latents', 'attr'})]['v_latents'])

    result_attr = attr_decoded
    result_images = images_decoded

    if show_results_fusion:
        result_attr = attr_fusion_decoded
        result_images = images_fusion_decoded

    return {
        "train_images": train_images,
        "train_attr": train_attr,
        "images_decoded": result_images,
        "attr_decoded": result_attr
    }


def get_grid_numpy(samples, nrow=8):
    grid = make_grid(samples, nrow=nrow, pad_value=1).permute(1, 2, 0)
    return grid.detach().cpu().numpy()

def plot_original_translated_comparison(original_images, result_images):
    grid_train = get_grid_numpy(original_images)
    grid_decoded = get_grid_numpy(result_images)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)) 

    ax1.imshow(grid_train)
    ax1.set_title("Images originales")
    ax1.axis('off')

    ax2.imshow(grid_decoded)
    ax2.set_title("Images traduites: attr => GW => v_latents")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


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

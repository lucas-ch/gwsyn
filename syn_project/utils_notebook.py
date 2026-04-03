import io
import math
from typing import Mapping, cast
import numpy as np
from shimmer import GlobalWorkspace2Domains
import torch
from PIL import Image
from torch.nn.functional import one_hot
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
import torch.nn.functional as F

from simple_shapes_dataset.cli import generate_image

import matplotlib.pyplot as plt

CAT2IDX = {"Diamond": 0, "Egg": 1, "Triangle": 2}

def get_image_from_interactive_attr(cat, x, y, size, rot, color_r, color_g, color_b):
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

def get_decoded_image_from_interactive_attr(cat, x, y, size, rot, color_r, color_g, color_b, training_params, device, global_workspace):
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
    decoded_latents = global_workspace.gw_mod.decode(attr_gw_latent['attr'])["v_latents"]
    decoded_images = (
        global_workspace.domain_mods["v_latents"]
        .decode_images(decoded_latents)[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    return decoded_images

def get_decoded_images_from_gw_latent(
        gw_latent:Mapping[frozenset[str], torch.Tensor],
        global_workspace:GlobalWorkspace2Domains):
    gw_latents_decoded = global_workspace.decode(gw_latent, ["v_latents", "attr"])
    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])
    images = visual_module.decode_images(gw_latents_decoded["v_latents"]).detach().cpu()

    return images

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

def plot_img_comparison(idx, img_tensor_one: torch.Tensor, image_tensor_two: torch.Tensor):
    original_images_np = img_tensor_one.permute(0, 2, 3, 1).detach().cpu().numpy()
    decoded_images_np = image_tensor_two.permute(0, 2, 3, 1).detach().cpu().numpy()

    # Création d'une figure avec 1 ligne et 2 colonnes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Affichage de la prédiction
    axes[0].imshow(original_images_np[idx])
    axes[0].set_title(f"original (idx {idx})")
    axes[0].axis('off') # Optionnel : enlève les axes gradués

    # Affichage de la cible
    axes[1].imshow(decoded_images_np[idx])
    axes[1].set_title(f"decoded (idx {idx})")
    axes[1].axis('off')

    fig = plt.tight_layout() # Ajuste l'espacement entre les images
    return fig

def split_softmax_category_attributes(concat_tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Sépare un tenseur (N, 8) en deux tenseurs :
    - Les 3 premiers (logits -> probabilités 0-1)
    - Les 5 derniers (inchangés)
    """
    # 1. Extraction des 3 premières colonnes (logits)
    logits = concat_tensor[:, :3]
    
    # 2. Conversion en probabilités (entre 0 et 1)
    # On utilise softmax sur la dimension 1 pour que la somme = 1
    # Si les 3 attributs sont indépendants, utilise torch.sigmoid(logits) à la place
    probs = F.softmax(logits, dim=1)
    
    # 3. Extraction des 5 colonnes restantes
    rest = concat_tensor[:, 3:]
    
    # Retourne une liste de deux tenseurs comme dans ton premier exemple
    return [probs, rest]

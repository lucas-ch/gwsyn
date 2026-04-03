from typing import cast

from shimmer import GlobalWorkspace2Domains, LatentsDomainGroupsT, LossOutput, RawDomainGroupsT, SingleDomainSelection
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule

import torch
import torch.nn.functional as F


def get_mask(images: torch.Tensor, temperature: float = 30.0) -> torch.Tensor:
    """
    Version différentiable de l'extraction de masque.
    
    Args:
        images (torch.Tensor): Tensor de forme (B, C, H, W)
        temperature (float): Contrôle la "dureté" du seuil. 
                             Plus c'est élevé, plus c'est proche d'un masque binaire.
    
    Returns:
        torch.Tensor: Masque "soft" entre 0 et 1.
    """
    grayscale = images.mean(dim=1, keepdim=True)
    return torch.sigmoid((grayscale - 0.1) * temperature)

def dice_loss(pred:torch.Tensor, target:torch.Tensor, smooth=1)-> torch.Tensor:
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice.mean()

def get_area(tensor:torch.Tensor)-> torch.Tensor:
    area = tensor.sum(dim=(1, 2, 3))
    area_norm = (area - 25) / (128 - 25 + 1e-8)
    return torch.clamp(area_norm, 0.0, 1.0)

def get_centroid(mask:torch.Tensor)-> torch.Tensor:
    """
    Calcule le centre de masse avec :
    - x, y normalisés entre -1 et 1
    - (1, 1) correspondant au coin HAUT-DROITE
    """
    B, _, H, W = mask.shape
    device = mask.device
    
    # 1. Création des grilles de -1 à 1
    # Pour x : de gauche (-1) à droite (1)
    # Pour y : de bas (-1) à haut (1) -> on inverse l'ordre de linspace
    grid_x = torch.linspace(-1, 1, W, device=device)
    grid_y = torch.linspace(1, -1, H, device=device) # Inversion ici : le haut est 1
    
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    
    # 2. Calcul de la masse totale
    total_mass = mask.sum(dim=(2, 3)) + 1e-8
    
    # 3. Pondération par les intensités du masque
    # (B, 1, H, W) * (H, W) -> Somme sur H, W -> (B, 1)
    pos_x = (mask * mesh_x).sum(dim=(2, 3)) / total_mass
    pos_y = (mask * mesh_y).sum(dim=(2, 3)) / total_mass
    
    # 4. Retourne (B, 2)
    return torch.cat([pos_x, pos_y], dim=1)

def centroid_loss(mask_pred:torch.Tensor, mask_orig:torch.Tensor)->torch.Tensor:
    """Calcule la distance MSE entre les centres de masse."""
    center_pred = get_centroid(mask_pred)
    center_orig = get_centroid(mask_orig)
    return F.mse_loss(center_pred, center_orig)

def shape_loss(gw_mod: GlobalWorkspace2Domains, domain_latents: LatentsDomainGroupsT, raw_data: RawDomainGroupsT)->LossOutput:
    visual_module = cast(VisualLatentDomainModule, gw_mod.domain_mods["v_latents"])
    visual_module.eval()
    
    with torch.no_grad():
            images_raw = raw_data[frozenset({'attr', 'v_latents'})]['v_latents']
            x_target = visual_module.decode_images(images_raw)
            mask_target = get_mask(x_target)
            target_area = get_area(mask_target).float()
            target_centroid = get_centroid(mask_target)


    latents = domain_latents[frozenset({'attr', 'v_latents'})]
    domain_sources = {'attr': latents['attr']}
    z = gw_mod.encode_and_fuse(domain_sources, SingleDomainSelection())

    gw_decoded_latents = gw_mod.decode(z, domains={'v_latents'})['v_latents']
    x_recons = visual_module.decode_images(gw_decoded_latents)
    mask_pred = get_mask(x_recons)

    pred_area = get_area(mask_pred).float()
    pred_centroid = get_centroid(mask_pred)

    centroid_loss=F.mse_loss(pred_centroid, target_centroid)
    area_loss = F.mse_loss(pred_area, target_area)
    d_loss = dice_loss(mask_pred, mask_target)
    
    loss_output = LossOutput(centroid_loss + area_loss+ d_loss)

    return loss_output


from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import warnings
import cv2
from scipy import stats
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import glob
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import scikit_posthocs as sp
from typing import Dict, Literal, Optional, Tuple

from torchvision.utils import make_grid
import cv2
import numpy as np

import colorsys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Callable


CAT_NAMES = {0:"diamond", 1:"egg", 2:"triangle"}


def get_color_masks(images: torch.Tensor, s_thresh=0, v_min=0, v_max=254) -> np.ndarray:
    images_numpy = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    colors = []
    for i in range(images_numpy.shape[0]):
        img = images_numpy[i]
        color = get_color_mask(img, s_thresh, v_min, v_max)
        colors.append(color)

    return colors


def get_color_mask(image: np.ndarray, s_thresh=50, v_min=30, v_max=225) -> np.ndarray:
    """
    Extrait les pixels colorés en excluant le noir, le blanc et le gris.
    
    Args:
        image: Image RGB (float 0-1 ou uint8 0-255).
        s_thresh: Seuil de saturation (plus c'est haut, plus on ignore les gris).
        v_min: Seuil de noir (ignore ce qui est trop sombre).
        v_max: Seuil de blanc (ignore ce qui est trop clair).
    """
    # Conversion en uint8 si nécessaire
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        
    # Passage en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Création du masque :
    # 1. S > s_thresh : On veut de la couleur (pas de gris)
    # 2. V > v_min    : On ne veut pas de noir
    # 3. V < v_max    : On ne veut pas de blanc pur
    color_mask = (s > s_thresh) & (v > v_min) & (v < v_max)
    
    return color_mask


def get_color_from_attributes(data: torch.Tensor) -> torch.Tensor:
    if data.size(dim=1) < 11:
        raise Exception("no color in attributes") 
    """
    Prend un tenseur [X, 11], extrait les 3 derniers attributs (RGB)
    et les convertit de [-1, 1] vers [0, 255].
    """
    # 1. Extraction des 3 dernières colonnes (indices 8, 9, 10)
    # On utilise : pour toutes les lignes, et -3: pour les 3 dernières colonnes
    rgb_normalized = data[:, -3:] 
    
    # 2. Dénormalisation de [-1, 1] -> [0, 1]
    # .clone() évite de modifier 'data' si on fait des opérations in-place plus tard
    rgb_01 = (rgb_normalized.clone() + 1.0) / 2.0
    
    # 3. Sécurité et conversion
    rgb_01 = torch.clamp(rgb_01, 0.0, 1.0)
    return (rgb_01 * 255).to(torch.uint8)

# return r,g,b between 0 and 255 per entry in tensor
def get_color_from_images(images: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    images_numpy = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    colors = []
    for i in range(images_numpy.shape[0]):
        img = images_numpy[i]
        mask = masks[i]
        color = get_color_from_image(img, mask) * 255
        
        colors.append(color)

    return colors

def get_mask_from_shapes(images: torch.Tensor) -> torch.Tensor:
    masks = []
    images_numpy = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    for i in range(images_numpy.shape[0]):
        img = images_numpy[i]
        mask = get_mask_from_shape(img)
        
        masks.append(mask > 0)

    return masks

def get_mask_from_shape(image: np.ndarray) -> np.ndarray:
    """
    Segment a shape from the background using Otsu thresholding.
    
    Args:
        image: RGB image array.
        
    Returns:
        Binary mask (bool array) where True indicates shape pixels.
    """
    gray_image = np.mean(image, axis=2)
    gray_uint8 = (gray_image * 255).astype(np.uint8)
    _, mask = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0

# return r,g,b between 0 and 1
def get_color_from_image(image: np.ndarray, mask: np.ndarray, barycentre = False) -> np.ndarray:
    """
    Extract the color of a shape using its true barycenter (weighted center of mass).
    Takes into account the geometric properties of the shape.
    
    Args:
        image: RGB image array.
        mask: Binary mask where True indicates shape pixels.
    
    Returns:
        Array of RGB values representing the shape's characteristic color.
    """

    if not barycentre : 
         # Check if mask is valid
        if mask is None or not isinstance(mask, np.ndarray):
            warnings.warn("Invalid mask; using full image for color extraction")
            return np.mean(image, axis=(0, 1))
            
        # Check if mask has enough pixels
        if np.sum(mask) < 3:
            warnings.warn("Empty mask detected; using full image for color extraction")
            return np.mean(image, axis=(0, 1))
            
        mean_color = np.zeros(3)
        for c in range(3):
            pixels = image[:, :, c][mask]
            if len(pixels) > 0:
                mean_color[c] = np.mean(pixels)
            else:
                mean_color[c] = 0.0
        
        # brightest_image_color = np.max(image, axis=(0, 1))
                
        return mean_color


    # Check if mask is valid
    if mask is None or not isinstance(mask, np.ndarray):
        warnings.warn("Invalid mask; using full image for color extraction")
        return np.mean(image, axis=(0, 1))
    
    # Check if mask has enough pixels
    if np.sum(mask) < 10:
        warnings.warn("Empty mask detected; using full image for color extraction")
        return np.mean(image, axis=(0, 1))
    
    # Find all pixels in the mask
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return np.mean(image, axis=(0, 1))
    
    # Calculate moments for the shape
    # For a binary mask, the 0th moment is the area
    M00 = np.sum(mask)
    # 1st order moments give the "mass" distribution
    M10 = np.sum(np.multiply(mask, np.arange(mask.shape[1]).reshape(1, -1)))
    M01 = np.sum(np.multiply(mask, np.arange(mask.shape[0]).reshape(-1, 1)))
    
    # Calculate true barycenter coordinates
    if M00 > 0:  # Avoid division by zero
        centroid_x = int(M10 / M00)
        centroid_y = int(M01 / M00)
    else:
        centroid_y = int(np.mean(y_indices))
        centroid_x = int(np.mean(x_indices))
    
    # Ensure the coordinates are within image bounds
    height, width = mask.shape[:2]
    centroid_y = max(0, min(centroid_y, height - 1))
    centroid_x = max(0, min(centroid_x, width - 1))
    
    # Create a small region around the barycenter to sample colors
    # This is more robust than using a single pixel
    region_size = 3
    min_y = max(0, centroid_y - region_size//2)
    max_y = min(height, centroid_y + region_size//2 + 1)
    min_x = max(0, centroid_x - region_size//2)
    max_x = min(width, centroid_x + region_size//2 + 1)
    
    # Create region mask (intersection of original mask and region)
    region_mask = mask[min_y:max_y, min_x:max_x]
    
    # If we have valid pixels in the region, compute their average color
    if np.sum(region_mask) > 0:
        color = np.zeros(3)
        for c in range(3):
            color_channel = image[min_y:max_y, min_x:max_x, c]
            masked_values = color_channel[region_mask]
            if len(masked_values) > 0:
                color[c] = np.mean(masked_values)
        return color
    
    # Fallback: use average of all pixels in the mask
    color = np.zeros(3)
    for c in range(3):
        pixels = image[:, :, c][mask]
        if len(pixels) > 0:
            color[c] = np.mean(pixels)
        else:
            color[c] = 0.0
    return color

def plot_rgb_distribution(colors_np: np.ndarray, categories_indices: np.ndarray, cat_names=CAT_NAMES, n_bins=40) -> Figure:
    """Displays RGB distribution per category.

    colors_np is an array of RGB: [[0, 0, 255], [255, 255, 0], ...]
    categories indices is an array of categories (diamon, egg, triangle): [0, 0, 2, 1...]
    cat_names is a dictionnary storing the names of the category indices
    n_bins is the bins of histogram
    """
    unique_cats = np.unique(categories_indices)
    n_cats = len(unique_cats)
    channels = ['Red', 'Green', 'Blue']
    colors_palette = ['#e74c3c', '#2ecc71', '#3498db']     

    fig, axes = plt.subplots(3, n_cats, figsize=(n_cats * 2.5, 6), sharex=True, sharey=False)
    
    if n_cats == 1:
        axes = axes.reshape(3, 1)

    for col, cat in enumerate(unique_cats):
        mask = (categories_indices == cat)
        cat_data = colors_np[mask]
        
        name = cat_names.get(cat, f"CAT {cat}") if cat_names else f"CAT {cat}"
        
        for row in range(3):
            ax = axes[row, col]
            
            sns.histplot(
                cat_data[:, row], 
                bins=n_bins, 
                kde=True, 
                ax=ax, 
                color=colors_palette[row],
                element="step",
                alpha=0.6
            )
            
            ax.set_xlim(0, 255)
            ax.tick_params(axis='both', which='major', labelsize=8) # Petites polices
            
            if row == 0:
                ax.set_title(name.upper(), fontweight='bold', fontsize=10, pad=10)
            
            if col == 0:
                ax.set_ylabel(channels[row], fontweight='bold', fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            
            if row == 2:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")

            # Clean design
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=1.0)
    return fig

def categorize_decoded_attr(decoded_attr: torch.Tensor) -> torch.Tensor:
    """Get the category from a result of GW decoded attributes tensor.

    The tensor contains category in the first 3 columns [-1.2, -2.4, 12.5]
    categorize_decoded_attr returns corresponding one hot encoding [0, 0, 1]
    """
    first_three = decoded_attr[:, :3]
    
    indices = torch.argmax(first_three, dim=1)
    binary_categories = F.one_hot(indices, num_classes=3)
    
    return binary_categories.to(torch.float32)

def get_attr_classification_stats(y_true:np.ndarray, y_pred: np.ndarray) -> tuple[int, np.ndarray]:
    correct_mask = (y_true == y_pred)
    accuracy_pct = np.mean(correct_mask) * 100
    incorrect_indices = np.where(~correct_mask)[0]
    
    return accuracy_pct, incorrect_indices

def get_top_img_per_category(results, n = 10):
    attr = results["attr_decoded"]
    categories = torch.argmax(attr[:, :3], dim=1)

    selected_orig = []
    selected_decoded = []

    for cat in range(3):
        indices = (categories == cat).nonzero(as_tuple=True)[0]

        top_indices = indices[:n].to('cpu')

        selected_orig.append(results["train_images"][top_indices])
        selected_decoded.append(results["images_decoded"][top_indices])

    final_orig = torch.cat(selected_orig, dim=0)
    final_decoded = torch.cat(selected_decoded, dim=0)

    return final_orig, final_decoded

def get_grid_numpy(samples, nrow=10):
    grid = make_grid(samples, nrow=nrow, pad_value=1).permute(1, 2, 0)
    return grid.detach().cpu().numpy()


def plot_original_translated_comparison(original_images, result_images, max_images=30, nrow=10):
    num_to_show = min(len(original_images), max_images)
    orig_subset = original_images[:num_to_show]
    res_subset  = result_images[:num_to_show]

    grid_train   = get_grid_numpy(orig_subset, nrow=nrow)
    grid_decoded = get_grid_numpy(res_subset,  nrow=nrow)

    # Taille de figure basée sur les dimensions réelles des grilles en pixels
    dpi          = 100
    grid_h, grid_w = grid_train.shape[:2]
    fig_w        = (grid_w * 2) / dpi        # 2 grilles côte à côte
    fig_h        = (grid_h + 30) / dpi       # +30px pour le titre

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)

    ax1.imshow(grid_train, interpolation='nearest')
    ax1.set_title("Images originales", fontsize=10)
    ax1.axis('off')

    ax2.imshow(grid_decoded, interpolation='nearest')
    ax2.set_title("Images traduites", fontsize=10)
    ax2.axis('off')

    fig.subplots_adjust(top=0.92, bottom=0.0, right=1.0, left=0.0, wspace=0.05)
    return fig

def plot_original_color_comparison(original_images, colors_tensor, max_images=30):
    num_to_show = min(len(original_images), len(colors_tensor), max_images)
    orig_subset = original_images[:num_to_show]
    colors_subset = colors_tensor[:num_to_show]
    
    h, w = orig_subset.shape[2], orig_subset.shape[3]

    color_squares = colors_subset.view(num_to_show, 3, 1, 1).expand(-1, -1, h, w)
    color_squares = color_squares / 255.0
    
    grid_orig = get_grid_numpy(orig_subset)
    grid_color = get_grid_numpy(color_squares)

    # Affichage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) 
    ax1.imshow(grid_orig)
    ax1.set_title("Originales")
    ax1.axis('off')

    ax2.imshow(grid_color)
    ax2.set_title("Couleurs")
    ax2.axis('off')

    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.98, left=0.02, wspace=0.05)
    return fig

def get_samples_rgb(
    data_translated: Dict[str, any], 
    type: Literal['training', 'decoded', 'decoded_edge'] = 'decoded_edge'
) -> np.ndarray:
    """
    Extracts and aggregates color information from different image sources within the dataset.
    
    Args:
        data_translated: Dictionary containing 'train_images' and 'images_decoded' tensors/arrays.
        type: Selection string to determine which source to return:
              - 'training': colors from the original training images.
              - 'decoded': colors from the reconstructed images.
              - 'decoded_edge': colors from the reconstructed images excluding black and white (usually edges).
              
    Returns:
        A vertically stacked NumPy array of all detected pixel colors.
    """
    if type == 'training':
        masks = get_mask_from_shapes(data_translated["train_images"])
        colors_from_training_img = get_color_from_images(data_translated["train_images"], masks)
        colors_np = np.vstack(colors_from_training_img)
        return colors_np

    masks_decoded = get_mask_from_shapes(data_translated["images_decoded"])
    colors_masks = get_color_masks(data_translated["images_decoded"], 0, 0, 254)

    colors_from_decoded_img = get_color_from_images(data_translated["images_decoded"], masks_decoded)
    colors_from_decoded_img_edge = get_color_from_images(data_translated["images_decoded"], colors_masks)

    if type == "decoded":
        colors_np = np.vstack(colors_from_decoded_img)
    elif type == "decoded_edge":
        colors_np = np.vstack(colors_from_decoded_img_edge)

    return colors_np

def get_categories_indices(data_translated: dict[str, torch.Tensor], attr_type = 'attr_decoded') -> np.ndarray:
    """
    Converts raw decoded attributes into discrete category indices.
    
    Args:
        data_translated: Dictionary containing 'attr_decoded' and/or 'train_attr'(tensors of attributes with categories one-hot coded).
        
    Returns:
        A NumPy array of integers representing the class index for each sample.
    """

    # Convert raw attributes (logits) into categorized form (likely via softmax internally)
    categories_from_decoded_attr = categorize_decoded_attr(data_translated[attr_type])
    
    # Retrieve the index of the highest probability (Argmax) and move to CPU/NumPy for analysis
    categories_indices = categories_from_decoded_attr.argmax(dim=1).detach().cpu().numpy()

    return categories_indices

"""
color_analysis.py
─────────────────
Mesures de cohérence colorimétrique par catégorie.

Fonctions publiques
───────────────────
  compute_hue_metrics(colors_np, labels)
      → dict  {'F': float, 'p': float, 'lda': float}

  hue_analysis(colors_np, labels, ...)
      Affiche les métriques + histogramme de teinte.

  run_conditions(conditions, load_fn, ...)
      Itère sur une liste de conditions, appelle load_fn pour chacune,
      retourne un DataFrame de métriques.
"""



# ── Utilitaires internes ───────────────────────────────────────────────────────

def rgb_to_hue(rgb_array: np.ndarray) -> np.ndarray:
    """Array (n, 3) RGB 0-255  →  array (n,) teinte H en degrés 0-360."""
    return np.array([
        colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] * 360
        for r, g, b in rgb_array
    ])

def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Moyenne circulaire en degrés."""
    rad = np.deg2rad(angles_deg)
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)

def circular_std_deg(angles_deg: np.ndarray) -> float:
    """Écart-type circulaire en degrés (Mardia & Jupp)."""
    rad = np.deg2rad(angles_deg)
    R   = np.sqrt(np.sin(rad).mean() ** 2 + np.cos(rad).mean() ** 2)
    return float(np.rad2deg(np.sqrt(-2 * np.log(R))))

def hue_to_sin_cos(hues_deg: np.ndarray) -> np.ndarray:
    """Encode la teinte en (sin, cos) pour respecter la circularité."""
    rad = np.deg2rad(hues_deg)
    return np.stack([np.sin(rad), np.cos(rad)], axis=1)

def boost_color(rgb_0_1: np.ndarray, value: float, saturation_boost: float):
    """Force la luminosité V et booste la saturation en conservant la teinte."""
    h, s, v = colorsys.rgb_to_hsv(*rgb_0_1)
    s = min(1.0, s * saturation_boost)
    return colorsys.hsv_to_rgb(h, s, value)


# ── API publique ───────────────────────────────────────────────────────────────

def compute_hue_metrics(
    colors_np: np.ndarray,
    labels: np.ndarray,
    cat_names: dict | None = None,
) -> dict:
    """
    Calcule les métriques circulaires sur les teintes.
 
    Returns
    -------
    dict avec les clés :
        kruskal_H, kruskal_p, lda_score,
        per_cat : {cat_id: {'mean_hue', 'std_hue', 'name'}}
    """
    cats  = np.unique(labels)
    hues  = rgb_to_hue(colors_np)
    if cat_names is None:
        cat_names = {c: f'Cat. {c}' for c in cats}
 
    # LDA sur (sin H, cos H)
    X         = hue_to_sin_cos(hues)
    lda_score = LinearDiscriminantAnalysis().fit(X, labels).score(X, labels)
 
    # Kruskal-Wallis sur teintes "dépliées" autour de la moyenne globale
    global_mean = circular_mean_deg(hues)
 
    def wrap_around_mean(h: np.ndarray, ref: float) -> np.ndarray:
        return (h - ref + 180) % 360 - 180
 
    groups_wrapped       = [wrap_around_mean(hues[labels == c], global_mean) for c in cats]
    H_stat, p_kruskal    = stats.kruskal(*groups_wrapped)
 
    per_cat = {
        c: {
            'name':     cat_names.get(c, f'Cat. {c}'),
            'mean_hue': circular_mean_deg(hues[labels == c]),
            'std_hue':  circular_std_deg(hues[labels == c]),
        }
        for c in cats
    }
 
    return {
        'kruskal_H': H_stat,
        'kruskal_p': p_kruskal,
        'lda_score': lda_score,
        'per_cat':   per_cat,
        'hues':      hues,          # conservé pour la visualisation
        'labels':    labels,
    }


def hue_analysis(
    colors_np: np.ndarray,
    labels: np.ndarray,
    cat_names: dict = None,
    value: float = 0.85,
    saturation_boost: float = 1.4,
    title: str = None,
    ax=None,
) -> dict:
    """
    Affiche l'histogramme de teinte par catégorie + métriques F et LDA.

    Parameters
    ----------
    colors_np        : array (n, 3) RGB 0-255
    labels           : array (n,) entiers de catégorie
    cat_names        : dict {0: 'Nom', 1: 'Nom', ...}  (optionnel)
    value            : float 0-1, luminosité forcée pour l'affichage
    saturation_boost : float, multiplicateur de saturation
    title            : str, titre du graphe (optionnel)
    ax               : matplotlib Axes existant (optionnel)

    Returns
    -------
    dict {'F': float, 'p': float, 'lda': float}
    """
    cats = np.unique(labels)
    hues = rgb_to_hue(colors_np)

    if cat_names is None:
        cat_names = {c: f'Cat. {c}' for c in cats}

    mean_rgb = {
        c: boost_color(
            colors_np[labels == c].mean(axis=0) / 255,
            value=value,
            saturation_boost=saturation_boost,
        )
        for c in cats
    }

    metrics = compute_hue_metrics(colors_np, labels)
    print(f"Précision LDA (H seul) : {metrics['lda_score']:.2%}")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    for cat in cats:
        h = hues[labels == cat]
        color = mean_rgb[cat]
        label = f"{cat_names[cat]}  (moy. {h.mean():.0f}°)"
        ax.hist(h, bins=36, range=(0, 360), color=color,
                alpha=0.6, label=label, edgecolor='none')
        ax.axvline(h.mean(), color=color, linewidth=2, linestyle='--')

    ax.set_xlabel('Teinte H (°)', fontsize=11)
    ax.set_ylabel('Effectif', fontsize=11)
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.6)

    plot_title = title or f"LDA = {metrics['lda_score']:.1%}"
    ax.set_title(plot_title, fontsize=11)

    if standalone:
        plt.tight_layout()
        plt.show()

    return metrics
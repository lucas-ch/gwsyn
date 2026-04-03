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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from typing import Dict, Literal, Optional, Tuple

from torchvision.utils import make_grid
import cv2
import numpy as np

CAT_NAMES = {0:"diamond", 1:"egg", 2:"triangle"}


def get_color_masks(images: torch.Tensor, s_thresh=50, v_min=30, v_max=225) -> np.ndarray:
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
        if np.sum(mask) < 10:
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

    fig, axes = plt.subplots(3, n_cats, figsize=(n_cats * 3, 7), sharex=True, sharey=False)
    
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

def load_epoch_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "stats_epoch_*.npz")))
    return files

def get_color_data_from_epoch(file: str) -> tuple[int, np.ndarray, np.ndarray]:
    """Extract info from stats_epoch_*.npz file
    epoch: the training epoch from which the data is extracted
    colors: list of RGB of the samples: [[0, 0, 255], [12, 12, 12], ...]
    categories: list of catgories of the samples: [0, 2, 1, 0...]
    """

    data = np.load(file)
    epoch = int(os.path.basename(file).split('_')[-1].replace('.npz', ''))
    colors = data["colors_from_decoded_img"]
    categories_raw = data["categories_from_data_attr"]
    categories = np.argmax(categories_raw, axis=1) if categories_raw.ndim > 1 else categories_raw    

    return epoch, colors, categories

def get_color_data_from_epoch_files(files: list[str]) -> tuple[list[int], dict]:
    """Extract info from stats_epoch_*.npz files
    epochs: list of epochs
    history: for each category, each color, the list of value {0: {'R': [255, 12, 122...], {'G': [...]}, {'B': [...]}, 1: {...}, 2:{...}}
    """
    epochs = []
    history = {}
        
    for f in files:
        epoch, colors, cats = get_color_data_from_epoch(f)
        
        epochs.append(epoch)
        for cat in np.unique(cats):
            if cat not in history:
                history[cat] = {'R': [], 'G': [], 'B': []}
            
            cat_mask = (cats == cat)
            mean_rgb = colors[cat_mask].mean(axis=0)
            
            history[cat]['R'].append(mean_rgb[0])
            history[cat]['G'].append(mean_rgb[1])
            history[cat]['B'].append(mean_rgb[2])
    
    return epochs, history

def plot_color_history(
    ax: Axes, 
    epochs: list[int], 
    cat_id: int, 
    channels: dict[str, np.ndarray], 
    cat_names: Optional[dict[int, str]], 
    vline_epoch: Optional[int]
) -> Axes:
    """
    Trace l'évolution temporelle des intensités moyennes R, G, B pour une catégorie spécifique.

    Args:
        ax: L'objet Axes de matplotlib sur lequel tracer le graphique.
        epochs: Liste ou tableau des indices d'époques (axe X).
        cat_id: Identifiant numérique de la catégorie à tracer.
        channels: Dictionnaire contenant les vecteurs d'intensité {'R': [...], 'G': [...], 'B': [...]}.
        cat_names: Dictionnaire optionnel de correspondance {id: "Nom"}.
        vline_epoch: Époque optionnelle où tracer une ligne verticale (ex: switch d'architecture).

    Returns:
        L'objet Axes configuré et complété.
    """
    
    colors_palette = ['#e74c3c', '#2ecc71', '#3498db']

    name = cat_names.get(cat_id, f"Catégorie {cat_id}") if cat_names else f"Catégorie {cat_id}"
        
    ax.plot(epochs, channels['R'], color=colors_palette[0], label='Red', linewidth=2)
    ax.plot(epochs, channels['G'], color=colors_palette[1], label='Green', linewidth=2)
    ax.plot(epochs, channels['B'], color=colors_palette[2], label='Blue', linewidth=2)
        
    if vline_epoch is not None:
        ax.axvline(x=vline_epoch, color='black', linestyle='--', linewidth=1.5, alpha=0.8, 
                   label=f'Switch @ {vline_epoch}')
        
    ax.set_title(f"Color evolution : {name.upper()}", fontweight='bold')
    ax.set_ylabel("Average intensity (0-255)")
    ax.grid(True, linestyle='--', alpha=0.4)
    
    ax.set_ylim(0, 255)
    ax.legend(loc='upper right', fontsize='small')

    ax.xaxis.set_major_locator(MultipleLocator(50))

    return ax

def plot_color_evolution_per_category(
    data_dir: str, 
    cat_names: dict[int, str] = CAT_NAMES, 
    vline_epoch: Optional[int] = None
) -> Figure:
    files: list[str] = load_epoch_files(data_dir)
    epochs, history = get_color_data_from_epoch_files(files)

    n_cats: int = len(history)
    fig, axes = plt.subplots(n_cats, 1, figsize=(10, 3 * n_cats), sharex=True)
    
    if n_cats == 1: 
        axes = [axes]

    for i, (cat_id, channels) in enumerate(sorted(history.items())):
        ax = axes[i]
        plot_color_history(ax, epochs, cat_id, channels, cat_names, vline_epoch)

    axes[-1].set_xlabel("Epoch", fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_statistical_dominance_evolution(
    results_dict: dict[str, list[dict[str, any]]], 
    cat_names: Optional[dict[int, str]] = None, 
    vline_epoch: Optional[int] = None
) -> Figure:
    """
    Plots the evolution of the Kruskal-Wallis H-statistic across epochs for each category.
    Each subplot represents a category, with markers colored according to the 
    statistically dominant color identified by the Dunn post-hoc test.

    Parameters:
        results_dict: Dictionary containing "kruskal" and "dunn" analysis results.
        cat_names: Optional mapping of category IDs to human-readable names.
        vline_epoch: Optional epoch index to draw a vertical reference line.
    """
    df_k = pd.DataFrame(results_dict["kruskal"]).rename(columns={'f_stat': 'h_stat'})
    df_d = pd.DataFrame(results_dict["dunn"])
    df = pd.merge(df_k, df_d, on=['epoch', 'category']).sort_values(['category', 'epoch'])
    
    unique_cats = sorted(df['category'].unique())
    n_cats = len(unique_cats)
    
    color_map = {'R': '#e74c3c', 'G': '#2ecc71', 'B': '#3498db', 'None': '#95a5a6'}

    fig, axes = plt.subplots(n_cats, 1, figsize=(10, 3 * n_cats), sharex=True)
    if n_cats == 1: axes = [axes]

    for i, cat in enumerate(unique_cats):
        ax = axes[i]
        cat_data = df[df['category'] == cat]
        cat_label = cat_names.get(cat, f"Category {cat}") if cat_names else f"Category {cat}"
        
        ax.plot(cat_data['epoch'], cat_data['h_stat'], 
                color='black', alpha=0.15, linewidth=1, zorder=1)
        
        point_colors = [color_map.get(row['dominant_color'], '#95a5a6') 
                        if row['is_dominant'] else '#95a5a6' 
                        for _, row in cat_data.iterrows()]
        
        ax.scatter(cat_data['epoch'], cat_data['h_stat'], 
                   c=point_colors, s=8, edgecolors='none', zorder=2)

        ax.set_ylim(bottom=0)
        
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, current_ylim[1] * 1.1)

        ax.set_title(f"Statistical bias (H-Stat) : {cat_label.upper()}", fontweight='bold', loc='left')
        ax.set_ylabel("Bias intensity")
        ax.grid(True, linestyle=':', alpha=0.4)
        
        if vline_epoch is not None:
            ax.axvline(x=vline_epoch, color='black', linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Epoch", fontweight='bold')
    axes[-1].xaxis.set_major_locator(MultipleLocator(50))
    
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', lw=0, marker='o', label='Dominant color: red'),
        Line2D([0], [0], color='#2ecc71', lw=0, marker='o', label='Dominant color: green'),
        Line2D([0], [0], color='#3498db', lw=0, marker='o', label='Dominant color: blue'),
        Line2D([0], [0], color='#95a5a6', lw=0, marker='o', label='No dominant color')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def kruskal_colors(
    colors: np.ndarray, 
    category_to_analyze: int, 
    categories: np.ndarray
) -> stats._stats_py.KruskalResult:
    """
    Performs a Kruskal-Wallis H-test as a non-parametric alternative to ANOVA, 
    testing if the median distributions of R, G, and B channels differ.

    Parameters:
        colors: Array of RGB values of the samples (N, 3).
        category: The specific category ID to analyze.
        categories: Array of category labels for the samples (N,).
    """
    cat_colors = colors[categories == category_to_analyze]
    
    r = cat_colors[:, 0]
    g = cat_colors[:, 1]
    b = cat_colors[:, 2]
    
    return stats.kruskal(r, g, b)

def dunn_colors(
    colors: np.ndarray, 
    cat: int, 
    cats: np.ndarray
) -> Tuple[str, bool, pd.DataFrame]:
    """
    Performs a Dunn post-hoc test with Bonferroni correction to identify if 
    one specific color channel statistically dominates the others.
    """ 
    cat_colors = colors[cats == cat]
    if len(cat_colors) < 5:
        return "Insuffisant", False

    data = [cat_colors[:, 0], cat_colors[:, 1], cat_colors[:, 2]]
    
    p_matrix = sp.posthoc_dunn(data, p_adjust='bonferroni')
    p_matrix.columns = ['R', 'G', 'B']
    p_matrix.index = ['R', 'G', 'B']

    means = np.mean(cat_colors, axis=0)
    channels = ['R', 'G', 'B']
    dom_idx = np.argmax(means)
    dom_name = channels[dom_idx]

    others = [c for c in channels if c != dom_name]
    is_dominant = all(p_matrix.loc[dom_name, other] < 0.05 for other in others)

    return dom_name, is_dominant, p_matrix

def color_statisitical_dominance_analysis(
    data_dir: str
) -> dict[str, list[dict[str, any]]]:
    """
    Iterates through epoch files to perform Kruskal-Wallis and Dunn post-hoc tests 
    for every category. Aggregates statistical metrics and dominance data into 
    a structured dictionary for downstream visualization.

    Parameters:
        data_dir: Path to the directory containing the .npz epoch files.

    Returns:
        A dictionary with two keys:
            - "kruskal": List of records containing H-stats and p-values.
            - "dunn": List of records containing dominant color names and p-matrices.
    """
    files = load_epoch_files(data_dir)
    kruskal = []
    dunn = []
    
    for f in files:
        epoch, colors, cats = get_color_data_from_epoch(f)
        
        for cat in np.unique(cats):
            h_stat, p_val = kruskal_colors(colors, cat, cats)
            dom_name, is_dominant, p_matrix = dunn_colors(colors, cat, cats)

            dunn.append({
                "epoch": epoch,
                "category": cat,
                "dominant_color": dom_name,
                "is_dominant": is_dominant,
                "p_matrix": p_matrix}
                )
            
            kruskal.append({
                "epoch": epoch,
                "category": cat,
                "h_stat": h_stat,
                "p_val": p_val,
                "significant": p_val < 0.05
            })

    return {
        "kruskal": kruskal,
        "dunn": dunn
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

def get_samples_rgb(
    data_translated: Dict[str, any], 
    type: Literal['training', 'decoded', 'decoded_edge'] = 'training'
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
    # Generate binary masks for shape localization
    masks = get_mask_from_shapes(data_translated["train_images"])
    masks_decoded = get_mask_from_shapes(data_translated["images_decoded"])

    # Generate a specific mask for colorful pixels (excluding grayscale background/noise)
    colors_masks = get_color_masks(data_translated["images_decoded"], 0, 0, 240)

    # Extract pixel values (RGB) located within the defined masks
    colors_from_training_img = get_color_from_images(data_translated["train_images"], masks)
    colors_from_decoded_img = get_color_from_images(data_translated["images_decoded"], masks_decoded)
    colors_from_decoded_img_edge = get_color_from_images(data_translated["images_decoded"], colors_masks)

    # Default selection: Original training colors
    colors_np = np.vstack(colors_from_training_img)
    
    # Switch output based on the 'type' argument
    if type == "decoded":
        colors_np = np.vstack(colors_from_decoded_img)
    elif type == "decoded_edge":
        colors_np = np.vstack(colors_from_decoded_img_edge)

    return colors_np

def get_categories_indices(data_translated: dict[str, torch.Tensor]) -> np.ndarray:
    """
    Converts raw decoded attributes into discrete category indices.
    
    Args:
        data_translated: Dictionary containing 'attr_decoded' (tensor of attributes with categories one-hot coded).
        
    Returns:
        A NumPy array of integers representing the class index for each sample.
    """
    # Convert raw attributes (logits) into categorized form (likely via softmax internally)
    categories_from_decoded_attr = categorize_decoded_attr(data_translated["attr_decoded"])
    
    # Retrieve the index of the highest probability (Argmax) and move to CPU/NumPy for analysis
    categories_indices = categories_from_decoded_attr.argmax(dim=1).detach().cpu().numpy()

    return categories_indices

import numpy as np
import warnings
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_color_from_attr_batch(data: torch.Tensor) -> torch.Tensor:
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
def get_color_from_img_batch(batch_tensor: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    batch_np = batch_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

    colors = []
    for i in range(batch_np.shape[0]):
        img = batch_np[i]
        mask = masks[i]
        color = get_color_from_img(img, mask) * 255
        
        colors.append(color)

    return colors

def get_mask_from_shape_batch(batch_tensor: torch.Tensor) -> torch.Tensor:
    masks = []
    batch_np = batch_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

    for i in range(batch_np.shape[0]):
        img = batch_np[i]
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
def get_color_from_img(image: np.ndarray, mask: np.ndarray, barycentre = False) -> np.ndarray:
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

def get_color_stats_per_category(colors_np, categories_indices):
    unique_cats = np.unique(categories_indices)
    stats = {}

    for cat in unique_cats:
        cat_mask = (categories_indices == cat)
        cat_colors = colors_np[cat_mask]
        
        if len(cat_colors) == 0: continue

        centroid = np.mean(cat_colors, axis=0)        
        distances = np.linalg.norm(cat_colors - centroid, axis=1)        
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        stats[cat] = {
            'centroid': centroid,
            'avg_dispersion': avg_dist,
            'std_dispersion': std_dist,
            'count': len(cat_colors)
        }

    return stats

def plot_rgb_distribution(colors_np, categories_indices, cat_names={0:"diamond", 1:"egg", 2:"triangle"}, n_bins=40):
    """
    Affiche une grille compacte des distributions RGB par catégorie.
    cat_names: dict {index: "nom"} (ex: {0: "Route", 1: "Trottoir"})
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

def plot_rgb_distribution_3D(categories_indices, colors_np):

    unique_categories = np.unique(categories_indices)
    n_categories = len(unique_categories)

    fig = plt.figure(figsize=(n_categories * 5, 5)) # Adapte la taille selon le nombre de classes
    fig.suptitle("Distribution des couleurs par catégorie (RGB Space)", fontsize=16)

    for i, cat in enumerate(unique_categories):
        ax = fig.add_subplot(1, n_categories, i + 1, projection='3d')
        
        mask = (categories_indices == cat)
        cat_colors = colors_np[mask]
        
        point_colors = cat_colors / 255.0
        point_colors = np.clip(point_colors, 0.0, 1.0)
        
        ax.scatter(cat_colors[:, 0],
                cat_colors[:, 1],
                cat_colors[:, 2],
                c=point_colors,
                s=50,
                marker='o',
                edgecolors='k',
                linewidths=0.5)     

        ax.set_title(f"Catégorie {cat}")
        ax.set_xlabel("Rouge (R)")
        ax.set_ylabel("Vert (G)")
        ax.set_zlabel("Bleu (B)")
        
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)

    plt.tight_layout()
    plt.show()

def categorize_decoded_attr(decoded_attr: torch.Tensor) -> torch.Tensor:
    # First three columns are the category
    first_three = decoded_attr[:, :3]
    
    indices = torch.argmax(first_three, dim=1)
    binary_categories = F.one_hot(indices, num_classes=3)
    
    return binary_categories.to(torch.float32) # Optionnel: convertir en float

def get_attr_classification_stats(y_true, y_pred):
    correct_mask = (y_true == y_pred)
    accuracy_pct = np.mean(correct_mask) * 100
    incorrect_indices = np.where(~correct_mask)[0]
    
    return accuracy_pct, incorrect_indices

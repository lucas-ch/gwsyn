from datetime import datetime
import math
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Dict
import warnings
from tqdm import tqdm
import colorsys
import os 
import pickle

from torchvision.transforms import ToTensor
from PIL import Image
from scipy.stats import ks_2samp
from shimmer.modules.selection import SingleDomainSelection



def extract_shape_parameters(row: Dict) -> Dict[str, Any]:
    """Extract and return basic shape parameters from a data row."""
    return {
        'class': int(row["class"]),
        'position': row["location"],
        'size_px': float(row["size"]),
        'rotation': float(row["rotation"])
    }


def normalize_shape_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize shape parameters for model input."""
    pos_norm = normalize_position(params['position'])
    size_norm = normalize_size(params['size_px'])
    rotx, roty = normalize_rotation(params['rotation'])
    
    return {
        'position_norm': pos_norm,
        'size_norm': size_norm,
        'rotation_norm': (rotx, roty)
    }


def process_color_information(row: Dict, rgb_colors: np.ndarray, 
                             fixed_reference: bool, reference_color_idx: int) -> Dict[str, Any]:
    """Process and normalize color information."""
    if fixed_reference:
        color_rgb = rgb_colors[reference_color_idx]
    else:
        color_idx = int(row["color_index"])
        color_rgb = rgb_colors[color_idx]
    
    # Store original normalized color (0 to 1 range)
    original_color = [
        color_rgb[0] / 255.0,
        color_rgb[1] / 255.0,
        color_rgb[2] / 255.0
    ]
    
    # Normalize RGB values to [-1, 1] range for the model
    color_norm = [
        color_rgb[0] / 127.5 - 1,
        color_rgb[1] / 127.5 - 1,
        color_rgb[2] / 127.5 - 1
    ]
    
    return {
        'original_color': color_rgb,
        'color_norm': color_norm
    }


def create_attribute_vector(shape_params: Dict[str, Any], color_params: Dict[str, Any], 
                           color_mode: bool) -> torch.Tensor:
    """Create attribute vector for model input."""
    pos_norm = shape_params['position_norm']
    size_norm = shape_params['size_norm']
    rotx, roty = shape_params['rotation_norm']
    
    if color_mode:
        color_norm = color_params['color_norm']
        return torch.tensor(
            [[pos_norm[0], pos_norm[1], size_norm, rotx, roty,
              color_norm[0], color_norm[1], color_norm[2]]],
            dtype=torch.float32
        )
    else:
        return torch.tensor(
            [[pos_norm[0], pos_norm[1], size_norm, rotx, roty]],
            dtype=torch.float32
        )


def load_ground_truth_image(idx: int, im_dir: str) -> Image.Image:
    """Load the ground truth image for a sample."""
    image_path = f"{im_dir}/image_{idx:05d}.png"
    with Image.open(image_path) as image:
        return image.convert("RGB")
        

def preprocess_dataset(
    df: Any,
    analysis_attributes: List[str],
    shape_names: List[str],
    color_mode: bool,
    rgb_colors: np.ndarray,
    device: torch.device,
    fixed_reference: bool = False,
    reference_color_idx: int = 0,
    im_dir: str = "./evaluation_set"
) -> List[Dict]:
    """
    Preprocesses the dataset by extracting and normalizing all required parameters.
    
    Args:
        df: Pandas DataFrame containing the dataset.
        analysis_attributes: List of attributes to analyze.
        shape_names: List of shape names.
        color_mode: Whether color information should be used.
        rgb_colors: RGB color mapping as a numpy array.
        device: Torch device.
        fixed_reference: If True, overrides each sample's color with the reference.
        reference_color_idx: Index of the reference color to use if fixed_reference is True.
        im_dir: Directory containing the images.
        
    Returns:
        List of preprocessed samples with all metadata.
    """
    preprocessed_samples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing samples"):
        # Extract basic shape parameters
        shape_params_raw = extract_shape_parameters(row)
        
        # Normalize shape parameters
        shape_params_norm = normalize_shape_parameters(shape_params_raw)
        
        # Process color information
        color_params = process_color_information(
            row, rgb_colors, fixed_reference, reference_color_idx
        )
        
        # Create attribute vector
        attr_vector = create_attribute_vector(shape_params_norm, color_params, color_mode)
        
        # Create one-hot class encoding
        cls = shape_params_raw['class']
        one_hot = F.one_hot(torch.tensor([cls]), num_classes=len(shape_names)).float()
        
        # Load ground truth image
        image = load_ground_truth_image(idx, im_dir)
        if image is None:
            continue
            
        # Create sample dictionary
        sample = {
            'shape': shape_names[cls],
            'position_x': shape_params_raw['position'][0],
            'position_y': shape_params_raw['position'][1],
            'size': shape_params_raw['size_px'],
            'rotation': shape_params_raw['rotation'],
            'original_color': color_params['original_color'],
            'model_inputs': {
                'one_hot': one_hot.to(device),
                'attr_vector': attr_vector.to(device)
            },
            'row_idx': idx,
            'visual_ground_truth': image
        }
        
        preprocessed_samples.append(sample)
    
    return preprocessed_samples

def generate_fixed_colors(
    n_samples: int, max_hue: int = 180
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate colors with a linear distribution across the hue circle.
    """
    # Create evenly spaced hue values, 180 is the max value for hue sur open cv.
    hue_values = np.linspace(0, max_hue, n_samples, endpoint=False).astype(np.uint8)
    
    lightness_values = np.full(n_samples, 180, dtype=np.uint8)
    # Create evenly spaced saturation values (full saturation)
    saturation_values = np.full(n_samples, 230, dtype=np.uint8)
    
    # Create the HLS array
    hls = np.zeros((1, n_samples, 3), dtype=np.uint8)
    hls[0, :, 0] = hue_values
    hls[0, :, 1] = lightness_values
    hls[0, :, 2] = saturation_values
    
    # Convert to RGB
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]
    
    return rgb.astype(int), hls[0].astype(int)

def generate_fixed_colors_original(
    n_samples: int, min_lightness: int = 0, max_lightness: int = 256, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    assert 0 <= max_lightness <= 256
    hls = np.random.randint(
        [0, min_lightness, 0],
        [181, max_lightness, 256],
        size=(1, n_samples, 3),
        dtype=np.uint8,
    )
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]  # type: ignore
    return rgb.astype(int), hls[0].astype(int)


def compare_color_distribution(colors, title, is_hls=False, n_samples=100, seed=0):
    """
    Generate colors and display histograms for each channel (RGB or HLS).
    
    Args:
        colors: Array of color values to plot
        title: Title for the saved plot
        is_hls: If True, interpret colors as HLS; otherwise as RGB
        n_samples: Number of colors to generate
        seed: Random seed for reproducibility
    """
    import matplotlib.pyplot as plt
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set channel names and colors based on the color space
    if is_hls:
        channels = ['Hue', 'Lightness', 'Saturation']
        plot_colors = ['purple', 'gray', 'orange']
        xlims = [(0, 180), (0, 255), (0, 255)]  # HLS ranges
    else:
        channels = ['Red', 'Green', 'Blue']
        plot_colors = ['red', 'green', 'blue']
        xlims = [(0, 255), (0, 255), (0, 255)]  # RGB ranges
    
    # Plot histogram for each channel
    for i in range(3):
        channel_values = colors[:, i]
        axes[i].hist(channel_values, bins=180, color=plot_colors[i], alpha=0.7)
        axes[i].set_title(f'{channels[i]} Channel Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim(*xlims[i])
        
    plt.tight_layout()
    type_text = "HLS" if is_hls else "RGB"
    plt.suptitle(f'{type_text} Color Channel Distributions', y=1.02, fontsize=16)
    plt.savefig(title)
    plt.show()

n_samples = 100
seed = 0
# Generate colors
rgb_colors, hls_colors = generate_fixed_colors_original(n_samples=n_samples)

# Compare color distributions for HLS values
compare_color_distribution(hls_colors, title='hls_distribution', is_hls=True)




def normalize_position(pos: List[float]) -> List[float]:
    """
    Normalize position coordinates from pixel space to [-1, 1] range.
    
    Args:
        pos: Position in pixel coordinates [x, y].
        
    Returns:
        Normalized position in the range [-1, 1].
    """
    return [(pos[0] - 7) / 18 * 2 - 1, (pos[1] - 7) / 18 * 2 - 1]


def normalize_size(size: float) -> float:
    """
    Normalize size from pixel space to [-1, 1] range.
    
    Args:
        size: Size in pixels.
        
    Returns:
        Normalized size in the range [-1, 1].
    """
    s = (size - 7) / 7
    return s * 2 - 1


def normalize_rotation(rot: float) -> Tuple[float, float]:
    """
    Convert rotation angle to cosine and sine components.
    
    Args:
        rot: Rotation angle in radians.
        
    Returns:
        Tuple of (cos(rot), sin(rot)).
    """
    return math.cos(rot), math.sin(rot)


def bin_attribute(value: float, bins: int, attr_range: Tuple[float, float]) -> int:
    """
    Assign a value to a bin index based on the specified range and number of bins.
    
    Args:
        value: The value to bin.
        bins: Number of bins.
        attr_range: (min, max) tuple defining the attribute range.
        
    Returns:
        Bin index from 0 to bins-1.
    """
    min_val, max_val = attr_range
    normalized = (value - min_val) / (max_val - min_val)
    bin_idx = int(normalized * bins + 0.25)
    if bin_idx == bins:
        bin_idx = bins - 1
    return bin_idx


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        p: First distribution as a numpy array.
        q: Second distribution as a numpy array.
        bins: Number of bins for histogram discretization.
        
    Returns:
        KL divergence value.
    """
    if len(p) == 0 or len(q) == 0:
        return float('nan')
    
    # Filter out NaN values
    p = np.array([x for x in p if not np.isnan(x)])
    q = np.array([x for x in q if not np.isnan(x)])
    
    if len(p) == 0 or len(q) == 0:
        return float('nan')

    hist_p, bin_edges = np.histogram(p, bins=bins, range=(0, 255), density=True)
    hist_q, _ = np.histogram(q, bins=bins, range=(0, 255), density=True)

    epsilon = 1e-10
    hist_p = hist_p + epsilon
    hist_q = hist_q + epsilon

    hist_p = hist_p / np.sum(hist_p)
    hist_q = hist_q / np.sum(hist_q)

    return np.sum(hist_p * np.log(hist_p / hist_q))


def segment_shape(image: np.ndarray) -> np.ndarray:
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

def extract_shape_color(image: np.ndarray, mask: np.ndarray, barycentre = False) -> np.ndarray:
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

def _generate_debug_output(sample: Dict) -> Dict:
    """
    Generate random outputs for debugging purposes.
    
    Args:
        sample: The original sample.
        
    Returns:
        Dictionary with random outputs for debugging.
    """
    import numpy as np
    # Generate random outputs for debugging
    random_image = lambda: np.random.random((32, 32, 3))
    random_mask = lambda: np.random.random((32, 32)) > 0.5
    random_color = lambda: np.random.randint(0, 256, size=3)
    
    return {
        'original_sample': sample,  # Keep track of the original data
        # Translation Results
        'translated_image': random_image(),
        'translated_mask': random_mask(),
        'translated_shape_color': random_color(),  # RGB color
        # Half-Cycle Results
        'half_cycle_image': random_image(),
        'half_cycle_mask': random_mask(),
        'half_cycle_shape_color': random_color(),  # RGB color
        # Full-Cycle Results
        'full_cycle_image': random_image(),
        'full_cycle_mask': random_mask(),
        'full_cycle_shape_color': random_color(),  # RGB color
    }

def _prepare_inputs(sample: Dict, device: torch.device) -> Tuple[List, torch.Tensor]:
    """
    Prepare attribute and visual inputs from the sample.
    
    Args:
        sample: The sample dictionary.
        device: Torch device.
        
    Returns:
        Tuple containing attribute inputs and visual ground truth tensor.
    """
    # Attribute inputs
    one_hot = sample['model_inputs']['one_hot'].to(device)
    attr_vector = sample['model_inputs']['attr_vector'].to(device)
    attr_inputs = [one_hot, attr_vector]
    
    # Visual inputs-
    if 'visual_ground_truth' not in sample:
        raise ValueError(f"Sample missing 'visual_ground_truth'. Sample keys: {sample.keys()}")
    
    img = sample["visual_ground_truth"]
    TT = ToTensor()
    visual_ground_truth_tensor = TT(img)[:3].unsqueeze(0).to(device)
    # print ("DEBUG : ", visual_ground_truth_tensor.shape)
    # print("DEBUG : ", img)
    # # Print only non-zero values in the tensor
    # non_zero = visual_ground_truth_tensor.detach().cpu().numpy()
    # non_zero_mask = non_zero != 0
    # non_zero_indices = np.where(non_zero_mask)
    # non_zero_values = non_zero[non_zero_mask]
    # print("DEBUG non-zero values: ", list(zip(zip(*non_zero_indices), non_zero_values)))
    # TT(img)[:3].unsqueeze(0).to(device)
    
    return attr_inputs, visual_ground_truth_tensor

def process_through_global_workspace(
    global_workspace: Any,
    preprocessed_samples: List[Dict],
    device: torch.device,
    debug: bool = False,
    reverb_n : int = 1,
    tilting: int = 1,
    barycentre = False,
) -> List[Dict]:
    """
    Processes samples through the Global Workspace, focusing on visual transformations.
    Calculates:
    1. Translation: Attribute -> GW -> Visual Image
    2. Half-Cycle: Visual -> GW -> Visual Image
    3. Full-Cycle: Visual -> GW -> Attribute -> GW -> Visual Image

    Args:
        global_workspace: The global workspace model instance.
        preprocessed_samples: List of preprocessed samples. Each sample dict
                              is expected to contain sample['model_inputs']
                              with keys like 'attr_vector', 'one_hot', and 'v_latent'.
        device: Torch device.
        debug: If True, generates random outputs instead of processing through model.

    Returns:
        List of processed samples, each containing the original sample info
        and the decoded images (and their extracted colors) for the three paths.
    """
    processed_results = []
    device = torch.device(device)
    global_workspace.to(device)  # Move model to the specified device
    global_workspace.eval()  # Make sure model is in evaluation mode

    # with torch.no_grad():  # Disable gradient calculations for inference
    #     for sample in tqdm(preprocessed_samples, desc="Processing samples through GW model"):
    #         if debug:
    #             processed_results.append(_generate_debug_output(sample))
    #             continue

    #         # Prepare inputs
    #         attr_inputs, visual_ground_truth_tensor = _prepare_inputs(sample, device)
            
    #         # Get visual latent vector from ground truth image
    #         visual_module = global_workspace.domain_mods["v_latents"]
    #         v_latent_vector = visual_module.visual_module.encode(visual_ground_truth_tensor)
            
    #         # Process through different pathways
    #         translated_image_np, translated_mask, translated_shape_color = _process_translation_path(
    #             global_workspace, attr_inputs, device, n=reverb_n, class_value=tilting, barycentre=barycentre)
                
    #         half_cycle_image_np, half_cycle_mask, half_cycle_shape_color = _process_half_cycle_path(
    #             global_workspace, v_latent_vector, device, barycentre=barycentre)
                
    #         full_cycle_image_np, full_cycle_mask, full_cycle_shape_color = _process_full_cycle_path(
    #             global_workspace, v_latent_vector, device, n=reverb_n, barycentre=barycentre)
    #         # weird_cycle_image_np, weird_cycle_mask, weird_cycle_shape_color = _process_weird_cycle_path(
    #         #     global_workspace, attr_inputs, device, n=reverb_n)
            
    #         if debug_ == True :
    #             import matplotlib.pyplot as plt
    #             img = visual_ground_truth_tensor
    #             img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             img = (img * 255).astype('uint8')
                
    #             # plot the 3 images side to side 
    #             plt.figure()
    #             plt.axis('off')
    #             plt.title("Visual Ground Truth")
    #             # Convert to PIL Image or Numpy array if needed
    #             import numpy as np
    #             from PIL import Image
    
    #             translated_image_np = (translated_image_np * 255).astype('uint8')
    #             half_cycle_image_np = (half_cycle_image_np * 255).astype('uint8')
    #             full_cycle_image_np = (full_cycle_image_np * 255).astype('uint8')

    #             if isinstance(translated_image_np, np.ndarray):
    #                 img2 = Image.fromarray(translated_image_np)
    #                 img3 = Image.fromarray(half_cycle_image_np)
    #                 img4 = Image.fromarray(full_cycle_image_np)
    #             # plot the 4 on the same image
    #             plt.subplot(1, 4, 1)
    #             plt.imshow(img)
    #             plt.axis('off')
    #             plt.title("Visual Ground Truth")
    #             plt.subplot(1, 4, 2)
    #             plt.imshow(img2)
    #             plt.axis('off')
    #             plt.title("Translated Image")
    #             plt.subplot(1, 4, 3)
    #             plt.imshow(img3)
    #             plt.axis('off')
    #             plt.title("Half Cycle Image")
    #             plt.subplot(1, 4, 4)
    #             plt.imshow(img4)
    #             plt.axis('off')
    #             plt.title("Full Cycle Image")
    #             # Save the figure
    #             plt.savefig("image.png")
    #             plt.show()
    #             plt.close()
    #         processed_results.append({
    #             'original_sample': sample,  # Keep track of the original data
    #             # Translation Results
    #             'translated_image': translated_image_np,
    #             'translated_mask': translated_mask,
    #             'translated_shape_color': translated_shape_color,  # RGB color
    #             # Half-Cycle Results
    #             'half_cycle_image': half_cycle_image_np,
    #             'half_cycle_mask': half_cycle_mask,
    #             'half_cycle_shape_color': half_cycle_shape_color,  # RGB color
    #             # Full-Cycle Results
    #             'full_cycle_image': full_cycle_image_np,
    #             'full_cycle_mask': full_cycle_mask,
    #             'full_cycle_shape_color': full_cycle_shape_color,  # RGB color
    #         })
            
    # return processed_results
    with torch.no_grad():
        # Use tqdm to track progress by *samples*, not batches
        with tqdm(total=len(preprocessed_samples), desc="Processing samples through GW model", unit="sample") as pbar:
            # Iterate through preprocessed_samples in batches
            for i in range(0, len(preprocessed_samples), BATCH_SIZE):
                # Get a batch of raw/preprocessed samples (list of individual sample objects)
                batch_samples_list_raw = preprocessed_samples[i:i + BATCH_SIZE]

                if not batch_samples_list_raw:
                    continue # Skip if the last batch is empty

                # --- 1. Collate and prepare batch inputs ---
                # This function handles stacking, padding (if needed), moving to device,
                # error handling per sample, and returns valid samples list.
                batched_one_hot, batched_attr_vector, batched_visual_ground_truth, batched_target_indices, valid_batch_samples = _prepare_batch_inputs(batch_samples_list_raw, device)

                # Check if batch preparation was successful and resulted in valid samples
                if batched_one_hot is None or not valid_batch_samples:
                    # If batch preparation failed or no valid samples were found in this chunk,
                    # still update the progress bar for the samples that were intended for this batch chunk.
                    # This keeps the bar progress accurate relative to the original list length.
                    print(f"\nSkipping batch starting at index {i} due to preparation error or no valid samples.")
                    pbar.update(len(batch_samples_list_raw))
                    continue

                batch_size = batched_one_hot.size(0) # Get actual batch size (might be less than BATCH_SIZE)


                # --- 2. Get batched visual latent vector from batched ground truth ---
                visual_module = global_workspace.domain_mods["v_latents"]
                # visual_module.visual_module.encode must accept (B, C, H, W)
                batched_v_latent_vector = visual_module.visual_module.encode(batched_visual_ground_truth) # Should return (B, v_latent_dim)


                # --- 3. Process the batch through different pathways ---
                # These functions now take batched inputs and return batched outputs (tensors on GPU)
                batched_translated_image_tensor = _process_translation_path_batch(
                    global_workspace, batched_one_hot, batched_attr_vector, batched_target_indices, device,
                    class_value=tilting, n=reverb_n
                ) # (B, C, H, W)

                batched_half_cycle_image_tensor = _process_half_cycle_path_batch(
                    global_workspace, batched_v_latent_vector, device
                ) # (B, C, H, W)

                batched_full_cycle_image_tensor = _process_full_cycle_path_batch(
                    global_workspace, batched_v_latent_vector, device, n=reverb_n
                ) # (B, C, H, W)

                # --- 4. Collect and post-process results on CPU ---
                # Move batched tensors back to CPU and convert to NumPy batches
                # Permute dimensions from (B, C, H, W) to (B, H, W, C) for image processing functions
                translated_image_np_batch = batched_translated_image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() # (B, H, W, C)
                half_cycle_image_np_batch = batched_half_cycle_image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() # (B, H, W, C)
                full_cycle_image_np_batch = batched_full_cycle_image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() # (B, H, W, C)


                # Loop through the results of THIS batch (on CPU) to perform
                # per-sample CPU operations (segment_shape, extract_shape_color)
                # and collect the individual results.
                batch_results_list = []
                for j in range(batch_size):
                    # Get individual image NumPy arrays from the batch
                    translated_image_np = translated_image_np_batch[j] # (H, W, C)
                    half_cycle_image_np = half_cycle_image_np_batch[j] # (H, W, C)
                    full_cycle_image_np = full_cycle_image_np_batch[j] # (H, W, C)

                    # Perform CPU post-processing per sample
                    # Make sure segment_shape and extract_shape_color work on (H, W, C) NumPy arrays
                    try:
                        translated_mask = segment_shape(translated_image_np)
                        translated_shape_color = extract_shape_color(translated_image_np, translated_mask, barycentre) * 255

                        half_cycle_mask = segment_shape(half_cycle_image_np)
                        half_cycle_shape_color = extract_shape_color(half_cycle_image_np, half_cycle_mask, barycentre) * 255

                        full_cycle_mask = segment_shape(full_cycle_image_np)
                        full_cycle_shape_color = extract_shape_color(full_cycle_image_np, full_cycle_mask, barycentre) * 255

                        # Store results for this individual sample
                        batch_results_list.append({
                            # Use the original sample from the list of successfully prepared samples
                            'original_sample': valid_batch_samples[j],
                            # Translation Results (already NumPy arrays)
                            'translated_image': translated_image_np,
                            'translated_mask': translated_mask,
                            'translated_shape_color': translated_shape_color,
                            # Half-Cycle Results (already NumPy arrays)
                            'half_cycle_image': half_cycle_image_np,
                            'half_cycle_mask': half_cycle_mask,
                            'half_cycle_shape_color': half_cycle_shape_color,
                            # Full-Cycle Results (already NumPy arrays)
                            'full_cycle_image': full_cycle_image_np,
                            'full_cycle_mask': full_cycle_mask,
                            'full_cycle_shape_color': full_cycle_shape_color,
                        })
                    except Exception as e:
                        # Log error and skip results for this specific sample
                        sample_info = valid_batch_samples[j].get('row_idx', 'N/A')
                        print(f"\nError during CPU post-processing for sample (row_idx {sample_info}) in batch starting at {i}: {e}. Skipping this sample.")
                        # This sample's results will not be added to processed_results

                # Extend the main results list with the results from this batch
                processed_results.extend(batch_results_list)

                # --- Update tqdm progress bar ---
                # Update by the number of *raw* samples intended for this batch chunk,
                # even if some were skipped during _prepare_batch_inputs or post-processing.
                # This ensures the progress bar completes covering all original samples.
                pbar.update(len(batch_samples_list_raw))


    print("\nProcessing complete.")
    return processed_results

selction_mod = SingleDomainSelection()
def _process_half_cycle_path(global_workspace: Any, v_latent_vector: torch.Tensor, device: torch.device, barycentre = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    
    """
    Process the Half-Cycle path: Visual -> GW -> Visual.
    
    Args:
        global_workspace: The global workspace model.
        v_latent_vector: Visual latent vector.
        device: Torch device.
        
    Returns:
        Tuple containing (image, mask, shape_color) for the half-cycle path.
    """
    # Encode Visual Latent -> GW
    v_gw_latent = global_workspace.gw_mod.encode_and_fuse({"v_latents": v_latent_vector}, SingleDomainSelection())
    
    # Decode GW -> Visual Latent
    half_cycle_v_latent = global_workspace.gw_mod.decode(v_gw_latent)["v_latents"]
    
    # Decode Visual Latent -> Image
    half_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(half_cycle_v_latent)[0]
    half_cycle_image_np = half_cycle_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    half_cycle_mask = segment_shape(half_cycle_image_np)
    half_cycle_shape_color = extract_shape_color(half_cycle_image_np, half_cycle_mask, barycentre) * 255
    
    return half_cycle_image_np, half_cycle_mask, half_cycle_shape_color


from shimmer.modules import SingleDomainSelection
def _process_full_cycle_path(global_workspace: Any, v_latent_vector: torch.Tensor, 
                            device: torch.device, n: int = 10, selection_module = SingleDomainSelection(), barycentre = False,
                            )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    

    """
    Process the Full-Cycle path: Visual -> GW -> Attribute -> GW -> Visual.
    Perform the cycle n times.
    
    Args:
        global_workspace: The global workspace model.
        v_latent_vector: Visual latent vector.
        device: Torch device.
        n: Number of cycles to perform (default: 1).
        
    Returns:
        Tuple containing (image, mask, shape_color) for the full-cycle path.
    """
    selection_module.to(device)
    # Start with the original visual latent vector
    current_v_latent = v_latent_vector
    # Save the v_latent tensor to desktop
    # desktop_path = os.path.expanduser("~/Desktop")
    # debug_file_path = os.path.join(desktop_path, "DEBUG.pt")
    # torch.save(current_v_latent.cpu().detach(), debug_file_path)
    # print(f"Saved v_latent tensor to {debug_file_path}")
    # Perform n cycles
    for _ in range(n):
        # Encode Visual Latent -> GW
        gw_latent_from_v = global_workspace.gw_mod.encode_and_fuse({"v_latents": current_v_latent}, selection_module=selection_module)
        
        # Decode GW -> Attribute Latent
        intermediate_attr_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["attr"]
        probas = intermediate_attr_latent[0][0:3]
        # print("DEBUG max probas : ", probas.max())

        # print("DEBUG somme des probas : ", probas.sum())
        # Encode Attribute Latent -> GW
        gw_latent_intermediate = global_workspace.gw_mod.encode_and_fuse({"attr": intermediate_attr_latent}, selection_module=selection_module)
        
        # Decode GW -> Visual Latent
        current_v_latent = global_workspace.gw_mod.decode(gw_latent_intermediate)["v_latents"]
    
    # Decode final Visual Latent -> Image
    full_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(current_v_latent)[0]
    full_cycle_image_np = full_cycle_image_tensor.permute(1, 2, 0).cpu().detach().numpy()
    full_cycle_mask = segment_shape(full_cycle_image_np)
    full_cycle_shape_color = extract_shape_color(full_cycle_image_np, full_cycle_mask, barycentre) * 255
    
    return full_cycle_image_np, full_cycle_mask, full_cycle_shape_color


def _process_translation_path(global_workspace: Any, attr_inputs: List, device: torch.device, n: int = 1, selection_module = SingleDomainSelection(), barycentre = False, noise_scale: float = 0.1, tilting = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the Translation path: Attribute -> GW -> Visual.
    Perform the cycle n times.
    """
    selection_module.to(device)
    
    # Add random noise to the one-hot encoding while maintaining sum of 1
    one_hot = attr_inputs[0].clone()
    
    # Ensure all values are non-negative
    one_hot = torch.clamp(one_hot, min=1e-8)
    
    # Normalize to ensure sum equals 1.0
    one_hot = one_hot / one_hot.sum()
    
    # Verify sum is 1.0
    assert abs(one_hot.sum() - 1.0) < 1e-5, f"One-hot encoding doesn't sum to 1, sum is {one_hot.sum()}"
    
    attr_inputs[0] = one_hot
    current_attr_input = attr_inputs
    
    # Perform n cycles
    for _ in range(n):
        # Encode Attribute -> GW
        intermediate_attr_domain = global_workspace.encode_domain(current_attr_input, "attr")
        attr_gw_latent = global_workspace.gw_mod.encode_and_fuse(
            {"attr": intermediate_attr_domain},
            selection_module=selection_module
        )
        
        # Decode GW -> Visual Latent
        v_latent = global_workspace.gw_mod.decode(attr_gw_latent)["v_latents"]
        
        # For additional cycles, we'd need to go back to attribute domain
        if _ < n - 1:
            # Encode Visual Latent -> GW
            gw_latent_from_v = global_workspace.gw_mod.encode_and_fuse(
                {"v_latents": v_latent},
                selection_module=selection_module
            )
            
            # Decode GW -> Attribute Latent
            intermediate_attr_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["attr"]
            current_attr_input = global_workspace.domain_mods["attr"].decode(intermediate_attr_latent)
    
    # Decode final Visual Latent -> Image
    translated_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(v_latent)[0]
    translated_image_np = translated_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    translated_mask = segment_shape(translated_image_np)
    translated_shape_color = extract_shape_color(translated_image_np, translated_mask, barycentre) * 255
    
    return translated_image_np, translated_mask, translated_shape_color


def safe_extract_channel(processed_sample: dict, channel: str, sample_index: int) -> float:
    """
    Safely extract a channel value from a processed sample.
    If not found at the top level, attempt to extract from 'shape_color'.
    
    Args:
        processed_sample: A dictionary containing processed sample data.
        channel: The color channel to extract (e.g., 'R', 'G', or 'B').
        sample_index: Index of the sample for logging purposes.
        
    Returns:
        The extracted channel value or np.nan if extraction fails.
    """

    shape_color = processed_sample['shape_color']
        
    # Handle numpy arrays
    if isinstance(shape_color, np.ndarray):
        channel_indices = {"R": 0, "G": 1, "B": 2}
        idx = channel_indices.get(channel)
        if idx is not None and idx < shape_color.shape[0]:
            return float(shape_color[idx])
    
    # Handle lists/tuples
    elif isinstance(shape_color, (list, tuple)):
        channel_indices = {"R": 0, "G": 1, "B": 2}
        idx = channel_indices.get(channel)
        if idx is not None and idx < len(shape_color):
            return float(shape_color[idx])
    
    # Debug information
    if 'shape_color' in processed_sample:
        warnings.warn(f"Shape color data type: {type(processed_sample['shape_color'])}")
        warnings.warn(f"Shape color value: {processed_sample['shape_color']}")
    else:
        warnings.warn(f"No shape_color key in processed sample at index {sample_index}")
        
    warnings.warn(f"Could not extract {channel} channel data from processed sample at index {sample_index}")
    return np.nan


def rgb_to_hls(rgb_values):
    """
    Convert RGB values (0-255) to HSV values
    Returns H in range [0,360), S and V in range [0,1]
    """
    # Normalize RGB to [0,1] range
    r, g, b = [x/255.0 for x in rgb_values]
    h, s, v = colorsys.rgb_to_hls(r, g, b)
    # Convert H to degrees [0,360)
    h = h * 360
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return h, r, g, b


# def initialize_h_binning_structures(
#     analysis_attributes,
#     binning_config, 
#     rgb = False
# ):
#     """
#     Initialize data structures for binning samples by attribute for H channel.
#     Creates separate dictionaries for each processing path.
#     """
#     input_colors_by_attr = {}
#     translated_colors_by_attr = {}
#     half_cycle_colors_by_attr = {}
#     full_cycle_colors_by_attr = {}
#     examples_by_attr = {}

#     for attr in analysis_attributes:
#         input_bins = {}
#         translated_bins = {}
#         half_cycle_bins = {}
#         full_cycle_bins = {}
#         example_bins = {}
        
#         if not rgb:
#             for bin_name in binning_config[attr]['bin_names']:
#                 input_bins[bin_name] = {'H': []}
#                 translated_bins[bin_name] = {'H': []}
#                 half_cycle_bins[bin_name] = {'H': []}
#                 full_cycle_bins[bin_name] = {'H': []}
#                 example_bins[bin_name] = []
#         else : 
#             for bin_name in binning_config[attr]['bin_names']:
#                 input_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
#                 translated_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
#                 half_cycle_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
#                 full_cycle_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
#                 example_bins[bin_name] = []

#         input_colors_by_attr[attr] = input_bins
#         translated_colors_by_attr[attr] = translated_bins
#         half_cycle_colors_by_attr[attr] = half_cycle_bins
#         full_cycle_colors_by_attr[attr] = full_cycle_bins
#         examples_by_attr[attr] = example_bins

#     return input_colors_by_attr, translated_colors_by_attr, half_cycle_colors_by_attr, full_cycle_colors_by_attr, examples_by_attr

# def bin_h_processed_samples_with_paths(
#             preprocessed_samples,
#             processed_samples,
#             analysis_attributes,
#             binning_config,
#             input_colors_by_attr,
#             translated_colors_by_attr,
#             half_cycle_colors_by_attr,
#             full_cycle_colors_by_attr,
#             examples_by_attr,
#             display_examples=True, 
#             rgb=False
#         ):
#         """
#         Bins processed samples by attribute values and extracts Hue channel values for all paths.
#         This is an improved version of bin_h_processed_samples that tracks all paths separately.
        
#         Args:
#             preprocessed_samples: List of preprocessed samples
#             processed_samples: List of processed samples from process_through_global_workspace
#             analysis_attributes: List of attributes to analyze
#             binning_config: Configuration for binning
#             input_colors_by_attr: Dict to store input color distributions
#             output_colors_by_attr: Dict to store output color distributions (translated path)
#             half_cycle_colors_by_attr: Dict to store half-cycle color distributions
#             full_cycle_colors_by_attr: Dict to store full-cycle color distributions
#             examples_by_attr: Dict to store example images by attribute and bin
#             display_examples: Whether to store example images
#         """

#         n = len(preprocessed_samples)
#         for idx, (preproc, processed) in enumerate(zip(preprocessed_samples, processed_samples)):
#             # Extract attribute values directly from preprocessed samples
#             shape = preproc.get('model_inputs').get('one_hot').argmax().item()
#             rotation = preproc.get('rotation')
#             size = preproc.get('size')
#             position_x = preproc.get('position_x')
#             position_y = preproc.get('position_y')

#             # Extract RGB channels from colors
#             input_color = preproc.get('original_color')
#             input_h, input_r, input_g, input_b = rgb_to_hls(input_color)
#             # Get output colors from all paths
#             translated_h, translated_r, translated_g, translated_b = rgb_to_hls(processed['translated_shape_color'])
#             half_cycle_h, half_cycle_r, half_cycle_g, half_cycle_b = rgb_to_hls(processed['half_cycle_shape_color'])
#             full_cycle_h, full_cycle_r, full_cycle_g, full_cycle_b = rgb_to_hls(processed['full_cycle_shape_color'])
            
            
            
#             for attr in analysis_attributes:
#                 if attr == 'shape' and shape is not None:
#                     # Shape is categorical, so use name directly
#                     bin_name = binning_config['shape']['bin_names'][shape]
#                     bin_attr_value = shape
#                 elif attr == 'rotation' and rotation is not None:
#                     # Use raw rotation value for binning
#                     bin_idx = bin_attribute(rotation, binning_config['rotation']['n_bins'], 
#                                         binning_config['rotation']['range'])
#                     bin_name = binning_config['rotation']['bin_names'][bin_idx]
#                     bin_attr_value = rotation
                    
#                 elif attr == 'size' and size is not None:
#                     # Use raw size value for binning
                    
                    
#                     if n== 19200: 
#                         bin_idx = bin_attribute(size, binning_config['size']['n_bins'],
#                                             binning_config['size']['range'])
#                     else : 
#                         bin_idx = int(size - 7)
#                     bin_name = binning_config['size']['bin_names'][bin_idx]
#                     bin_attr_value = size
#                 elif attr == 'position_x' and position_x is not None:
#                     # Use raw x-position for binning
#                     bin_idx = bin_attribute(position_x, binning_config['position_x']['n_bins'], 
#                                         binning_config['position_x']['range'])
#                     bin_name = binning_config['position_x']['bin_names'][bin_idx]
#                     bin_attr_value = position_x

#                 elif attr == 'position_y' and position_y is not None:
#                     # Use raw y-position for binning
#                     bin_idx = bin_attribute(position_y, binning_config['position_y']['n_bins'], 
#                                         binning_config['position_y']['range'])
#                     bin_name = binning_config['position_y']['bin_names'][bin_idx]
#                     bin_attr_value = position_y
#                 else:
#                     continue
                
#                 # Add hue values to the appropriate bins for each path
#                 if bin_name in input_colors_by_attr[attr]:
#                     input_colors_by_attr[attr][bin_name]['H'].append(input_h)
#                     translated_colors_by_attr[attr][bin_name]['H'].append(translated_h)
#                     half_cycle_colors_by_attr[attr][bin_name]['H'].append(half_cycle_h)
#                     full_cycle_colors_by_attr[attr][bin_name]['H'].append(full_cycle_h)

#                     if rgb:
#                         input_colors_by_attr[attr][bin_name]['R'].append(input_r)
#                         input_colors_by_attr[attr][bin_name]['G'].append(input_g)
#                         input_colors_by_attr[attr][bin_name]['B'].append(input_b)
                        
#                         translated_colors_by_attr[attr][bin_name]['R'].append(translated_r)
#                         translated_colors_by_attr[attr][bin_name]['G'].append(translated_g)
#                         translated_colors_by_attr[attr][bin_name]['B'].append(translated_b)

#                         half_cycle_colors_by_attr[attr][bin_name]['R'].append(half_cycle_r)
#                         half_cycle_colors_by_attr[attr][bin_name]['G'].append(half_cycle_g)
#                         half_cycle_colors_by_attr[attr][bin_name]['B'].append(half_cycle_b)

#                         full_cycle_colors_by_attr[attr][bin_name]['R'].append(full_cycle_r)
#                         full_cycle_colors_by_attr[attr][bin_name]['G'].append(full_cycle_g)
#                         full_cycle_colors_by_attr[attr][bin_name]['B'].append(full_cycle_b)
                    
#                     # Store example if needed
#                     if display_examples and len(examples_by_attr[attr][bin_name]) < 5:
#                         examples_by_attr[attr][bin_name].append({
#                             'input_image': preproc.get('visual_ground_truth'),
#                             'translated_image': processed.get('translated_image'),
#                             'half_cycle_image': processed.get('half_cycle_image'),
#                             'full_cycle_image': processed.get('full_cycle_image'),
#                             'attr_value': bin_attr_value
#                         })

import numpy as np
import random # Pour le remplacement aléatoire
from typing import List, Dict, Any, Optional, Tuple, Set
# Supposons que ces fonctions existent et fonctionnent comme dans ton code original
# from your_module import rgb_to_hls, bin_attribute

    # --- Début de la fonction modifiée ---
    # def bin_h_processed_samples_with_paths(
    #     preprocessed_samples,
    #     processed_samples,
    #     analysis_attributes,
    #     binning_config,
    #     input_colors_by_attr,
    #     translated_colors_by_attr,
    #     half_cycle_colors_by_attr,
    #     full_cycle_colors_by_attr,
    #     examples_by_attr,  # Dictionnaire de listes d'exemples [{...images...}, ...]
    #     represented_input_hues_by_attr, # Nouveau: Dict de dicts de sets {attr: {bin_name: {h1, h2}}}
    #     max_examples_per_bin: int = 5, # Combien d'exemples variés nous voulons
    #     display_examples: bool = True,
    #     rgb: bool = False
    # ):
    #     """
    #     Bins processed samples by attribute values and extracts Hue channel values for all paths.
    #     Attempts to store varied examples in 'examples_by_attr' based on input Hue.

    #     Args:
    #         preprocessed_samples: List of preprocessed samples
    #         processed_samples: List of processed samples
    #         analysis_attributes: List of attributes to analyze
    #         binning_config: Configuration for binning
    #         input_colors_by_attr, translated_colors_by_attr, ...: Dicts for color distributions
    #         examples_by_attr: Dict to store example images by attribute and bin.
    #                           Structure: {attr_name: {bin_name: [example_dict_1, ...]}}
    #         represented_input_hues_by_attr: Dict to track Hues of stored examples.
    #                                    Structure: {attr_name: {bin_name: {input_h_1, ...}}}
    #         max_examples_per_bin: Max number of varied examples to store per bin.
    #         display_examples: Whether to store example images.
    #         rgb: Whether to store full RGB (not just H) for distributions.
    #     """
    #     if not display_examples: # Si on ne stocke pas d'exemples, rien à faire de plus ici
    #         # La logique originale de remplissage des `*_colors_by_attr` peut continuer
    #         # sans la complexité de la sélection variée.
    #         # Pour la simplicité de cet exemple, je vais assumer que si display_examples est True,
    #         # alors la nouvelle logique s'applique. Si False, la partie 'store example' est sautée.
    #         pass

    #     n_samples_total = len(preprocessed_samples)

    #     for idx, (preproc, processed) in enumerate(zip(preprocessed_samples, processed_samples)):
    #         # Extract attribute values directly from preprocessed samples
    #         # Assurez-vous que .get() est utilisé ou que les clés existent
    #         model_inputs = preproc.get('model_inputs', {})
    #         one_hot_tensor = model_inputs.get('one_hot')
    #         if one_hot_tensor is None or not hasattr(one_hot_tensor, 'argmax'):
    #             # print(f"Warning: Sample {idx} missing 'one_hot' or it's not a tensor. Skipping.")
    #             shape = None # Ou une valeur par défaut / gestion d'erreur
    #         else:
    #             shape = one_hot_tensor.argmax().item()

    #         rotation = preproc.get('rotation')
    #         size = preproc.get('size')
    #         position_x = preproc.get('position_x')
    #         position_y = preproc.get('position_y')

    #         # Extract RGB channels from colors
    #         input_color_rgb_array = preproc.get('original_color')
    #         if input_color_rgb_array is None:
    #             # print(f"Warning: Sample {idx} missing 'original_color'. Skipping color processing for this sample.")
    #             # Mettre des valeurs par défaut ou sauter ce sample pour le binning de couleur
    #             input_h, input_r, input_g, input_b = (0,0,0,0) # Defaults
    #         else:
    #             # S'assurer que input_color est une liste/tuple de 3 valeurs numériques
    #             if not (isinstance(input_color_rgb_array, (list, tuple, np.ndarray)) and len(input_color_rgb_array) == 3):
    #                 # print(f"Warning: Sample {idx} 'original_color' is malformed: {input_color_rgb_array}. Skipping color.")
    #                 input_h, input_r, input_g, input_b = (0,0,0,0)
    #             else:
    #                 input_h, input_r, input_g, input_b = rgb_to_hls(input_color_rgb_array)


    #         # Get output colors from all paths
    #         # Ajouter des .get() pour la robustesse
    #         translated_shape_color = processed.get('translated_shape_color', [0,0,0])
    #         half_cycle_shape_color = processed.get('half_cycle_shape_color', [0,0,0])
    #         full_cycle_shape_color = processed.get('full_cycle_shape_color', [0,0,0])

    #         translated_h, translated_r, translated_g, translated_b = rgb_to_hls(translated_shape_color)
    #         half_cycle_h, half_cycle_r, half_cycle_g, half_cycle_b = rgb_to_hls(half_cycle_shape_color)
    #         full_cycle_h, full_cycle_r, full_cycle_g, full_cycle_b = rgb_to_hls(full_cycle_shape_color)

    #         for attr in analysis_attributes:
    #             bin_name = None
    #             bin_attr_value = None

    #             if attr == 'shape' and shape is not None:

    #                 if shape < len(binning_config['shape']['bin_names']):
    #                     bin_name = binning_config['shape']['bin_names'][shape]
    #                     bin_attr_value = shape # ou bin_name si c'est plus utile
    #                 else:
    #                     # print(f"Warning: Shape index {shape} out of bounds for bin_names. Sample {idx}.")
    #                     continue
    #             elif attr == 'rotation' and rotation is not None:
    #                 bin_idx = bin_attribute(rotation, binning_config['rotation']['n_bins'],
    #                                         binning_config['rotation']['range'])
    #                 bin_name = binning_config['rotation']['bin_names'][bin_idx]
    #                 bin_attr_value = rotation
    #             elif attr == 'size' and size is not None:
    #                 # La logique spécifique pour 'size' et n_samples_total
    #                 if n_samples_total == 19200: # Assumant que c'est le dataset "grand"
    #                     bin_idx = bin_attribute(size, binning_config['size']['n_bins'],
    #                                             binning_config['size']['range'])
    #                 else: # Dataset plus petit ou différent
    #                     # Assurez-vous que `size` est un type qui supporte la soustraction de 7
    #                     # et que le résultat est un index valide pour `bin_names`.
    #                     try:
    #                         bin_idx = int(size - 7)
    #                         if not (0 <= bin_idx < len(binning_config['size']['bin_names'])):
    #                             # print(f"Warning: Calculated size bin_idx {bin_idx} out of range for sample {idx}, size {size}.")
    #                             continue
    #                     except TypeError:
    #                         # print(f"Warning: Size {size} for sample {idx} is not numeric. Skipping.")
    #                         continue
    #                 bin_name = binning_config['size']['bin_names'][bin_idx]
    #                 bin_attr_value = size
    #             elif attr == 'position_x' and position_x is not None:
    #                 bin_idx = bin_attribute(position_x, binning_config['position_x']['n_bins'],
    #                                         binning_config['position_x']['range'])
    #                 bin_name = binning_config['position_x']['bin_names'][bin_idx]
    #                 bin_attr_value = position_x
    #             elif attr == 'position_y' and position_y is not None:
    #                 bin_idx = bin_attribute(position_y, binning_config['position_y']['n_bins'],
    #                                         binning_config['position_y']['range'])
    #                 bin_name = binning_config['position_y']['bin_names'][bin_idx]
    #                 bin_attr_value = position_y
    #             else:
    #                 # print(f"Attribute {attr} not handled or value is None for sample {idx}")
    #                 continue # Skip to next attribute if current one can't be binned

    #             if bin_name is None:
    #                 # print(f"Warning: Could not determine bin_name for attr {attr}, sample {idx}. Skipping.")
    #                 continue

    #             # Assurer que les structures de dictionnaire existent
    #             # (Normalement fait par initialize_h_binning_structures)
    #             if attr not in input_colors_by_attr or bin_name not in input_colors_by_attr[attr]:
    #                 # print(f"Warning: Bin {bin_name} for attribute {attr} not initialized in color dicts. Skipping sample {idx}.")
    #                 continue # Ce bin n'est pas configuré


    #             # --- Remplissage des données de couleur (identique à avant) ---
    #             input_colors_by_attr[attr][bin_name]['H'].append(input_h)
    #             # ... (tout le reste du remplissage des H, R, G, B) ...
    #             translated_colors_by_attr[attr][bin_name]['H'].append(translated_h)
    #             half_cycle_colors_by_attr[attr][bin_name]['H'].append(half_cycle_h)
    #             full_cycle_colors_by_attr[attr][bin_name]['H'].append(full_cycle_h)

    #             if rgb:
    #                 input_colors_by_attr[attr][bin_name]['R'].append(input_r)
    #                 input_colors_by_attr[attr][bin_name]['G'].append(input_g)
    #                 input_colors_by_attr[attr][bin_name]['B'].append(input_b)

    #                 translated_colors_by_attr[attr][bin_name]['R'].append(translated_r)
    #                 translated_colors_by_attr[attr][bin_name]['G'].append(translated_g)
    #                 translated_colors_by_attr[attr][bin_name]['B'].append(translated_b)

    #                 half_cycle_colors_by_attr[attr][bin_name]['R'].append(half_cycle_r)
    #                 half_cycle_colors_by_attr[attr][bin_name]['G'].append(half_cycle_g)
    #                 half_cycle_colors_by_attr[attr][bin_name]['B'].append(half_cycle_b)

    #                 full_cycle_colors_by_attr[attr][bin_name]['R'].append(full_cycle_r)
    #                 full_cycle_colors_by_attr[attr][bin_name]['G'].append(full_cycle_g)
    #                 full_cycle_colors_by_attr[attr][bin_name]['B'].append(full_cycle_b)


    #             # --- Logique de stockage d'exemples variés ---
    #             if display_examples:
    #                 # Assurer que les structures pour les exemples et les teintes représentées existent
    #                 if attr not in examples_by_attr or bin_name not in examples_by_attr[attr]:
    #                     # print(f"Warning: Bin {bin_name} for attribute {attr} not initialized in examples_by_attr. Skipping example storage for sample {idx}.")
    #                     continue
    #                 if attr not in represented_input_hues_by_attr or bin_name not in represented_input_hues_by_attr[attr]:
    #                     # print(f"Warning: Bin {bin_name} for attribute {attr} not initialized in represented_input_hues_by_attr. Skipping example storage for sample {idx}.")
    #                     continue


    #                 current_examples_list = examples_by_attr[attr][bin_name]
    #                 current_represented_hues_set = represented_input_hues_by_attr[attr][bin_name]
                    
    #                 new_example_dict = {
    #                     'input_image': preproc.get('visual_ground_truth'),
    #                     'translated_image': processed.get('translated_image'),
    #                     'half_cycle_image': processed.get('half_cycle_image'),
    #                     'full_cycle_image': processed.get('full_cycle_image'),
    #                     'attr_value': bin_attr_value, # La valeur de l'attribut qui a causé ce bin
    #                     'original_input_h': input_h   # Stocker la teinte d'origine pour la stratégie de variété
    #                 }

    #                 if len(current_examples_list) < max_examples_per_bin:
    #                     # Le bin n'est pas plein, on ajoute directement
    #                     current_examples_list.append(new_example_dict)
    #                     current_represented_hues_set.add(input_h)
    #                 else:
    #                     # Le bin est plein. On remplace si le nouvel exemple apporte une nouvelle teinte d'input.
    #                     if input_h not in current_represented_hues_set:
    #                         # Cette teinte n'est pas encore représentée, on veut cet exemple.
    #                         # Remplacer un exemple existant au hasard.
    #                         idx_to_replace = random.randint(0, max_examples_per_bin - 1)
                            
    #                         # Retirer la teinte de l'ancien exemple du set des teintes représentées
    #                         old_example_h = current_examples_list[idx_to_replace]['original_input_h']
    #                         current_represented_hues_set.discard(old_example_h) # Utiliser discard pour éviter KeyError
                            
    #                         # Remplacer l'exemple
    #                         current_examples_list[idx_to_replace] = new_example_dict
    #                         # Ajouter la nouvelle teinte
    #                         current_represented_hues_set.add(input_h)
    #                     # else: Si la teinte est déjà représentée et le bin est plein, on ignore ce nouvel exemple
    #                     #       pour conserver la diversité des teintes qu'on a déjà.
    #                     #       Une stratégie plus avancée pourrait aussi regarder d'autres attributs.
    # # --- Fin de la fonction modifiée ---
import random
import numpy as np
from collections import defaultdict
def bin_h_processed_samples_with_paths(
        preprocessed_samples,
        processed_samples,
        analysis_attributes,
        binning_config,
        input_colors_by_attr,
        translated_colors_by_attr,
        half_cycle_colors_by_attr,
        full_cycle_colors_by_attr,
        examples_by_attr,
        display_examples=True, 
        rgb=False
    ):
    """
    Bins processed samples by attribute values and extracts Hue channel values for all paths.
    This is an improved version of bin_h_processed_samples that tracks all paths separately.
    
    Args:
        preprocessed_samples: List of preprocessed samples
        processed_samples: List of processed samples from process_through_global_workspace
        analysis_attributes: List of attributes to analyze
        binning_config: Configuration for binning
        input_colors_by_attr: Dict to store input color distributions
        output_colors_by_attr: Dict to store output color distributions (translated path)
        half_cycle_colors_by_attr: Dict to store half-cycle color distributions
        full_cycle_colors_by_attr: Dict to store full-cycle color distributions
        examples_by_attr: Dict to store example images by attribute and bin
        display_examples: Whether to store example images
    """
    
    # Dictionnaire pour collecter les exemples diversifiés
    diverse_examples_collector = {
        'diamond': [],
        'egg': [],  
        'triangle': []
    }
    shape_names = ['diamond', 'egg', 'triangle']

    n = len(preprocessed_samples)
    for idx, (preproc, processed) in enumerate(zip(preprocessed_samples, processed_samples)):
        # Extract attribute values directly from preprocessed samples
        shape = preproc.get('model_inputs').get('one_hot').argmax().item()
        rotation = preproc.get('rotation')
        size = preproc.get('size')
        position_x = preproc.get('position_x')
        position_y = preproc.get('position_y')

        # Extract RGB channels from colors
        input_color = preproc.get('original_color')
        input_h, input_r, input_g, input_b = rgb_to_hls(input_color)
        # Get output colors from all paths
        translated_h, translated_r, translated_g, translated_b = rgb_to_hls(processed['translated_shape_color'])
        half_cycle_h, half_cycle_r, half_cycle_g, half_cycle_b = rgb_to_hls(processed['half_cycle_shape_color'])
        full_cycle_h, full_cycle_r, full_cycle_g, full_cycle_b = rgb_to_hls(processed['full_cycle_shape_color'])
        
        # Collecter les exemples diversifiés pour chaque forme
        if shape < len(shape_names):
            shape_name = shape_names[shape]
            if shape == 0:
                rotation_min = 3.6
            elif shape == 1:
                rotation_min = 0
            elif shape == 2:
                rotation_min = 1.6

            current_examples = diverse_examples_collector[shape_name]
            
            # Vérifier si on a besoin de plus d'exemples pour cette forme
            if len(current_examples) < 5:
                
                is_diverse = True
                # Check diversity across all attributes in a single loop
                if (rotation < rotation_min):
                    is_diverse = False
                if current_examples:  # Only check if we have existing examples
                    # is_diverse = True
                    thresholds = {
                        'input_h': 1,
                        'rotation': 0.01,
                        'size': 0.1,
                        'position_x': 1,
                        'position_y': 1  # Added position_y check
                    }
                    
                    for existing_example in current_examples:
                        # Check all attributes in one pass
                        if (abs(input_h - existing_example.get('input_h', 0)) < thresholds['input_h'] or
                            (abs(rotation - existing_example.get('rotation', 0)) < thresholds['rotation']) or
                            abs(size - existing_example.get('size', 0)) < thresholds['size'] or
                            abs(position_x - existing_example.get('position_x', 0)) + abs(position_y - existing_example.get('position_y', 0)) < thresholds['position_x']) :
                            # If any attribute is too similar, this example is not diverse
                            is_diverse = False
                            break
                # vérifier la diversité en position y
                
                if is_diverse :
                    example_data = {
                        'input_image': preproc.get('visual_ground_truth'),
                        'translated_image': processed.get('translated_image'),
                        'half_cycle_image': processed.get('half_cycle_image'),
                        'full_cycle_image': processed.get('full_cycle_image'),
                        'input_h': input_h,
                        'translated_h': translated_h,
                        'half_cycle_h': half_cycle_h,
                        'full_cycle_h': full_cycle_h,
                        'shape': shape,
                        'shape_name': shape_name,
                        'rotation': rotation,
                        'size': size,
                        'position_x': position_x,
                        'position_y': position_y,
                        'attr_value': shape
                    }
                    current_examples.append(example_data)
        
        for attr in analysis_attributes:
            if attr == 'shape' and shape is not None:
                # Shape is categorical, so use name directly
                bin_name = binning_config['shape']['bin_names'][shape]
                bin_attr_value = shape
            elif attr == 'rotation' and rotation is not None:
                # Use raw rotation value for binning
                bin_idx = bin_attribute(rotation, binning_config['rotation']['n_bins'], 
                                    binning_config['rotation']['range'])
                bin_name = binning_config['rotation']['bin_names'][bin_idx]
                bin_attr_value = rotation
                
            elif attr == 'size' and size is not None:
                # Use raw size value for binning
                if n== 19200: 
                    bin_idx = bin_attribute(size, binning_config['size']['n_bins'],
                                        binning_config['size']['range'])
                else : 
                    bin_idx = int(size - 7)
                bin_name = binning_config['size']['bin_names'][bin_idx]
                bin_attr_value = size
            elif attr == 'position_x' and position_x is not None:
                # Use raw x-position for binning
                bin_idx = bin_attribute(position_x, binning_config['position_x']['n_bins'], 
                                    binning_config['position_x']['range'])
                bin_name = binning_config['position_x']['bin_names'][bin_idx]
                bin_attr_value = position_x

            elif attr == 'position_y' and position_y is not None:
                # Use raw y-position for binning
                bin_idx = bin_attribute(position_y, binning_config['position_y']['n_bins'], 
                                    binning_config['position_y']['range'])
                bin_name = binning_config['position_y']['bin_names'][bin_idx]
                bin_attr_value = position_y
            else:
                continue
            
            # Add hue values to the appropriate bins for each path
            if bin_name in input_colors_by_attr[attr]:
                input_colors_by_attr[attr][bin_name]['H'].append(input_h)
                translated_colors_by_attr[attr][bin_name]['H'].append(translated_h)
                half_cycle_colors_by_attr[attr][bin_name]['H'].append(half_cycle_h)
                full_cycle_colors_by_attr[attr][bin_name]['H'].append(full_cycle_h)

                if rgb:
                    input_colors_by_attr[attr][bin_name]['R'].append(input_r)
                    input_colors_by_attr[attr][bin_name]['G'].append(input_g)
                    input_colors_by_attr[attr][bin_name]['B'].append(input_b)
                    
                    translated_colors_by_attr[attr][bin_name]['R'].append(translated_r)
                    translated_colors_by_attr[attr][bin_name]['G'].append(translated_g)
                    translated_colors_by_attr[attr][bin_name]['B'].append(translated_b)

                    half_cycle_colors_by_attr[attr][bin_name]['R'].append(half_cycle_r)
                    half_cycle_colors_by_attr[attr][bin_name]['G'].append(half_cycle_g)
                    half_cycle_colors_by_attr[attr][bin_name]['B'].append(half_cycle_b)

                    full_cycle_colors_by_attr[attr][bin_name]['R'].append(full_cycle_r)
                    full_cycle_colors_by_attr[attr][bin_name]['G'].append(full_cycle_g)
                    full_cycle_colors_by_attr[attr][bin_name]['B'].append(full_cycle_b)
                
                # Store example if needed
                if display_examples and len(examples_by_attr[attr][bin_name]) < 5:
                    examples_by_attr[attr][bin_name].append({
                        'input_image': preproc.get('visual_ground_truth'),
                        'translated_image': processed.get('translated_image'),
                        'half_cycle_image': processed.get('half_cycle_image'),
                        'full_cycle_image': processed.get('full_cycle_image'),
                        'attr_value': bin_attr_value
                    })
    
    # Ajouter les exemples diversifiés au dictionnaire final
    examples_by_attr['diverse_examples'] = diverse_examples_collector
    print_diverse_examples_stats(examples_by_attr)
    create_diverse_example_plate(examples_by_attr, save_path='diverse_examples_plate.png')


def create_diverse_example_plate(
    examples_by_attr: dict,
    save_path: str,
    plate_name: str = "diverse_examples",
    figsize: tuple = (18, 9)  # Reduced height since we removed title
) -> None:
    """
    Crée une plaquette d'exemples diversifiés avec 4 lignes et 15 colonnes :
    - Input (départ côté images)
    - Translation (départ côté attributs) 
    - Half-cycle (reconstruction - départ côté images)
    - Full-cycle (reconstruction - départ côté images)
    
    Args:
        examples_by_attr: Dictionnaire contenant les exemples avec 'diverse_examples'
        save_path: Chemin de sauvegarde
        plate_name: Nom de la plaquette
        figsize: Taille de la figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Récupérer les exemples diversifiés
    if 'diverse_examples' not in examples_by_attr:
        print("Aucun exemple diversifié trouvé dans examples_by_attr")
        return
    
    diverse_examples = examples_by_attr['diverse_examples']
    
    # Aplatir les exemples (5 de chaque forme)
    all_examples = []
    shape_order = ['diamond', 'egg', 'triangle']
    
    for shape_name in shape_order:
        if shape_name in diverse_examples:
            examples = diverse_examples[shape_name][:5]  # Max 5 par forme
            all_examples.extend(examples)
    
    if not all_examples:
        print("Aucun exemple diversifié trouvé")
        return
    
    num_examples = len(all_examples)
    print(f"Affichage de {num_examples} exemples diversifiés")
    
    # Définir les chemins et leurs labels
    paths_info = [
        ('input', 'Input', 'black'),
        ('translated', 'Traduction attributs ', 'green'), 
        ('half_cycle', 'Demi-cycle visuel', 'orange'),
        ('full_cycle', 'Cyccle-Complet visuel', 'red')
    ]
    
    # Créer la figure
    fig, axes = plt.subplots(len(paths_info), num_examples, figsize=figsize)
    
    # Gérer le cas où il n'y a qu'une seule ligne ou colonne
    if num_examples == 1:
        axes = axes.reshape(-1, 1)
    elif len(paths_info) == 1:
        axes = axes.reshape(1, -1)
    
    # Pour chaque ligne (chemin)
    for row_idx, (path_key, path_label, path_color) in enumerate(paths_info):
        # Label de la ligne à gauche
        if num_examples > 0:
            axes[row_idx, 0].text(-0.15, 0.5, path_label, 
                                 rotation=90, verticalalignment='center', 
                                 horizontalalignment='center', fontsize=12, 
                                 fontweight='bold', color=path_color,
                                 transform=axes[row_idx, 0].transAxes)
        
        # Pour chaque exemple
        for col_idx, example in enumerate(all_examples):
            if col_idx >= num_examples:
                break
            
            ax = axes[row_idx, col_idx]
            
            # Récupérer l'image pour ce chemin
            img_key = f'{path_key}_image'
            img = None
            
            if isinstance(example, dict) and img_key in example:
                img = example[img_key]
            
            # Traitement de l'image
            if hasattr(img, 'mode') and hasattr(img, 'size'):
                # Image PIL vers numpy
                img = np.array(img)
            
            # Conversion grayscale vers RGB si nécessaire
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            
            # Normalisation si nécessaire
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1.0:
                img = img / img.max()
            
            # Afficher l'image
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                       
            ax.axis('off')
    
    # Ajuster l'espacement pour rapprocher les exemples
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.08, right=0.98, top=0.98, bottom=0.02)
    
    # Sauvegarder
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{plate_name}_plate.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plaquette sauvegardée : {save_file}")


# Fonction utilitaire pour afficher les statistiques des exemples diversifiés
def print_diverse_examples_stats(examples_by_attr):
    """Affiche les statistiques des exemples diversifiés collectés"""
    if 'diverse_examples' not in examples_by_attr:
        print("Aucun exemple diversifié trouvé")
        return
    
    diverse_examples = examples_by_attr['diverse_examples']
    
    print("\n=== Statistiques des exemples diversifiés ===")
    total_examples = 0
    
    for shape_name, examples in diverse_examples.items():
        num_examples = len(examples)
        total_examples += num_examples
        print(f"{shape_name}: {num_examples} exemples")
        
        if examples:
            h_values = [ex.get('input_h', 0) for ex in examples]
            print(f"  Valeurs H: {[f'{h:.2f}' for h in h_values]}")
            
            rotations = [ex.get('rotation', 0) for ex in examples if ex.get('rotation') is not None]
            if rotations:
                print(f"  Rotations: {[f'{r:.1f}' for r in rotations]}")
            
            sizes = [ex.get('size', 0) for ex in examples if ex.get('size') is not None]
            if sizes:
                print(f"  Tailles: {[f'{s:.1f}' for s in sizes]}")
    
    print(f"\nTotal: {total_examples} exemples diversifiés collectés")


# def calculate_diversity_score(existing_examples, new_example):
#     """
#     Calcule un score de diversité pour un nouvel exemple par rapport aux exemples existants.
#     Plus le score est élevé, plus l'exemple est diversifié.
#     """
#     if not existing_examples:
#         return float('inf')  # Premier exemple, score maximum
    
#     diversity_score = 0
    
#     for existing in existing_examples:
#         # Distance pour les attributs numériques (normalisées)
#         rotation_diff = abs(new_example['rotation'] - existing['rotation']) / 360.0 if new_example['rotation'] is not None and existing['rotation'] is not None else 0
#         size_diff = abs(new_example['size'] - existing['size']) / 10.0 if new_example['size'] is not None and existing['size'] is not None else 0  # Assumant une plage de taille de ~10
#         pos_x_diff = abs(new_example['position_x'] - existing['position_x']) / 100.0 if new_example['position_x'] is not None and existing['position_x'] is not None else 0  # Assumant une plage de position de ~100
#         pos_y_diff = abs(new_example['position_y'] - existing['position_y']) / 100.0 if new_example['position_y'] is not None and existing['position_y'] is not None else 0
        
#         # Distance pour la teinte (circulaire, 0-360°)
#         h_diff = min(abs(new_example['input_h'] - existing['input_h']), 
#                     360 - abs(new_example['input_h'] - existing['input_h'])) / 180.0
        
#         # Score de diversité combiné (moyenne des différences)
#         example_diversity = (rotation_diff + size_diff + pos_x_diff + pos_y_diff + h_diff) / 5.0
#         diversity_score += example_diversity
    
#     # Score moyen de diversité par rapport à tous les exemples existants
#     return diversity_score / len(existing_examples)

# def _extract_sample_data(preproc, processed, idx):
#     """Extract and validate sample data."""
#     try:
#         model_inputs = preproc.get('model_inputs', {})
#         one_hot_tensor = model_inputs.get('one_hot')
        
#         if one_hot_tensor is None or not hasattr(one_hot_tensor, 'argmax'):
#             shape = None
#         else:
#             shape = one_hot_tensor.argmax().item()
        
#         rotation = preproc.get('rotation')
#         size = preproc.get('size')
#         position_x = preproc.get('position_x')
#         position_y = preproc.get('position_y')
        
#         input_color_rgb_array = preproc.get('original_color')
#         if input_color_rgb_array is None or not _is_valid_rgb(input_color_rgb_array):
#             input_h, input_r, input_g, input_b = (0, 0, 0, 0)
#         else:
#             input_h, input_r, input_g, input_b = rgb_to_hls(input_color_rgb_array)
        
#         # Get output colors with defaults
#         translated_shape_color = processed.get('translated_shape_color', [0, 0, 0])
#         half_cycle_shape_color = processed.get('half_cycle_shape_color', [0, 0, 0])  
#         full_cycle_shape_color = processed.get('full_cycle_shape_color', [0, 0, 0])
        
#         translated_h, translated_r, translated_g, translated_b = rgb_to_hls(translated_shape_color)
#         half_cycle_h, half_cycle_r, half_cycle_g, half_cycle_b = rgb_to_hls(half_cycle_shape_color)
#         full_cycle_h, full_cycle_r, full_cycle_g, full_cycle_b = rgb_to_hls(full_cycle_shape_color)
        
#         return {
#             'shape': shape,
#             'rotation': rotation,
#             'size': size,
#             'position_x': position_x,
#             'position_y': position_y,
#             'input_color_rgb_array': input_color_rgb_array,
#             'input_h': input_h, 'input_r': input_r, 'input_g': input_g, 'input_b': input_b,
#             'translated_h': translated_h, 'translated_r': translated_r, 'translated_g': translated_g, 'translated_b': translated_b,
#             'half_cycle_h': half_cycle_h, 'half_cycle_r': half_cycle_r, 'half_cycle_g': half_cycle_g, 'half_cycle_b': half_cycle_b,
#             'full_cycle_h': full_cycle_h, 'full_cycle_r': full_cycle_r, 'full_cycle_g': full_cycle_g, 'full_cycle_b': full_cycle_b,
#         }
#     except Exception as e:
#         print(f"Warning: Error processing sample {idx}: {e}")
#         return None


# def _is_valid_rgb(color):
#     """Check if color is a valid RGB array."""
#     return (isinstance(color, (list, tuple, np.ndarray)) and 
#             len(color) == 3 and 
#             all(isinstance(c, (int, float)) for c in color))


# def _get_bin_info(attr, sample_data, binning_config, n_samples_total, idx):
#     """Get bin name and attribute value for a given attribute."""
#     try:
#         if attr == 'shape' and sample_data['shape'] is not None:
#             shape = sample_data['shape']
#             if shape < len(binning_config['shape']['bin_names']):
#                 bin_name = binning_config['shape']['bin_names'][shape]
#                 return bin_name, shape
                
#         elif attr == 'rotation' and sample_data['rotation'] is not None:
#             rotation = sample_data['rotation']
#             bin_idx = bin_attribute(rotation, binning_config['rotation']['n_bins'],
#                                   binning_config['rotation']['range'])
#             bin_name = binning_config['rotation']['bin_names'][bin_idx]
#             return bin_name, rotation
            
#         elif attr == 'size' and sample_data['size'] is not None:
#             size = sample_data['size']
#             if n_samples_total == 19200:
#                 bin_idx = bin_attribute(size, binning_config['size']['n_bins'],
#                                       binning_config['size']['range'])
#             else:
#                 bin_idx = int(size - 7)
#                 if not (0 <= bin_idx < len(binning_config['size']['bin_names'])):
#                     return None
#             bin_name = binning_config['size']['bin_names'][bin_idx]
#             return bin_name, size
            
#         elif attr == 'position_x' and sample_data['position_x'] is not None:
#             position_x = sample_data['position_x']
#             bin_idx = bin_attribute(position_x, binning_config['position_x']['n_bins'],
#                                   binning_config['position_x']['range'])
#             bin_name = binning_config['position_x']['bin_names'][bin_idx]
#             return bin_name, position_x
            
#         elif attr == 'position_y' and sample_data['position_y'] is not None:
#             position_y = sample_data['position_y']
#             bin_idx = bin_attribute(position_y, binning_config['position_y']['n_bins'],
#                                   binning_config['position_y']['range'])
#             bin_name = binning_config['position_y']['bin_names'][bin_idx]
#             return bin_name, position_y
            
#     except (TypeError, ValueError, IndexError) as e:
#         print(f"Warning: Error binning {attr} for sample {idx}: {e}")
    
#     return None


# def _ensure_structures_exist(attr, bin_name, input_colors_by_attr, examples_by_attr, represented_input_hues_by_attr):
#     """Ensure all required data structures exist for the attribute and bin."""
#     return (attr in input_colors_by_attr and bin_name in input_colors_by_attr[attr] and
#             attr in examples_by_attr and bin_name in examples_by_attr[attr] and
#             attr in represented_input_hues_by_attr and bin_name in represented_input_hues_by_attr[attr])


# def _add_color_data(attr, bin_name, sample_data, input_colors_by_attr, translated_colors_by_attr,
#                    half_cycle_colors_by_attr, full_cycle_colors_by_attr, rgb):
#     """Add color data to the appropriate bins."""
#     # Add H values
#     input_colors_by_attr[attr][bin_name]['H'].append(sample_data['input_h'])
#     translated_colors_by_attr[attr][bin_name]['H'].append(sample_data['translated_h'])
#     half_cycle_colors_by_attr[attr][bin_name]['H'].append(sample_data['half_cycle_h'])
#     full_cycle_colors_by_attr[attr][bin_name]['H'].append(sample_data['full_cycle_h'])
    
#     # Add RGB if requested
#     if rgb:
#         for channel, key in [('R', 'r'), ('G', 'g'), ('B', 'b')]:
#             input_colors_by_attr[attr][bin_name][channel].append(sample_data[f'input_{key}'])
#             translated_colors_by_attr[attr][bin_name][channel].append(sample_data[f'translated_{key}'])
#             half_cycle_colors_by_attr[attr][bin_name][channel].append(sample_data[f'half_cycle_{key}'])
#             full_cycle_colors_by_attr[attr][bin_name][channel].append(sample_data[f'full_cycle_{key}'])


# def _calculate_diversity_scores(candidates, hue_tolerance):
#     """Calculate diversity scores for candidates based on hue uniqueness."""
#     for i, candidate in enumerate(candidates):
#         # Count how many other candidates have similar hues
#         similar_count = 0
#         for j, other in enumerate(candidates):
#             if i != j:
#                 hue_diff = min(abs(candidate['input_h'] - other['input_h']),
#                              1.0 - abs(candidate['input_h'] - other['input_h']))  # Circular distance
#                 if hue_diff <= hue_tolerance:
#                     similar_count += 1
        
#         # Higher score for more unique hues (fewer similar candidates)
#         candidate['diversity_score'] = 1.0 / (1.0 + similar_count)


# def _select_diverse_examples(candidates, max_examples, min_examples, diversity_weight, hue_tolerance):
#     """Select examples using a combination of diversity and randomness."""
#     if len(candidates) <= max_examples:
#         return candidates
    
#     selected = []
#     remaining = candidates.copy()
    
#     # First, ensure we get the most diverse examples
#     diversity_selections = min(max_examples, max(min_examples, int(max_examples * diversity_weight)))
    
#     # Sort by diversity score and select top diverse candidates
#     remaining.sort(key=lambda x: x['diversity_score'], reverse=True)
    
#     # Select diverse candidates, but avoid too many similar hues
#     for candidate in remaining:
#         if len(selected) >= diversity_selections:
#             break
            
#         # Check if this hue is already well represented
#         is_unique_enough = True
#         for selected_candidate in selected:
#             hue_diff = min(abs(candidate['input_h'] - selected_candidate['input_h']),
#                          1.0 - abs(candidate['input_h'] - selected_candidate['input_h']))
#             if hue_diff <= hue_tolerance:
#                 is_unique_enough = False
#                 break
        
#         if is_unique_enough or len(selected) < min_examples:
#             selected.append(candidate)
#             remaining.remove(candidate)
    
#     # Fill remaining slots randomly from remaining candidates
#     remaining_slots = max_examples - len(selected)
#     if remaining_slots > 0 and remaining:
#         random_selections = random.sample(remaining, min(remaining_slots, len(remaining)))
#         selected.extend(random_selections)
    
#     return selected


# def _process_samples_basic(preprocessed_samples, processed_samples, analysis_attributes,
#                          binning_config, input_colors_by_attr, translated_colors_by_attr,
#                          half_cycle_colors_by_attr, full_cycle_colors_by_attr, rgb):
#     """Basic processing without example storage for when display_examples=False."""
#     n_samples_total = len(preprocessed_samples)
    
#     for idx, (preproc, processed) in enumerate(zip(preprocessed_samples, processed_samples)):
#         sample_data = _extract_sample_data(preproc, processed, idx)
#         if sample_data is None:
#             continue
            
#         for attr in analysis_attributes:
#             bin_info = _get_bin_info(attr, sample_data, binning_config, n_samples_total, idx)
#             if bin_info is None:
#                 continue
                
#             bin_name, _ = bin_info
            
#             if (attr not in input_colors_by_attr or 
#                 bin_name not in input_colors_by_attr[attr]):
#                 continue
            
#             _add_color_data(attr, bin_name, sample_data, input_colors_by_attr,
#                           translated_colors_by_attr, half_cycle_colors_by_attr,
#                           full_cycle_colors_by_attr, rgb)

# Tu auras besoin d'initialiser `represented_input_hues_by_attr` de la même manière que `examples_by_attr`
# avant d'appeler cette fonction. Par exemple, dans `initialize_h_binning_structures`:

def initialize_h_binning_structures(
    analysis_attributes,
    binning_config,
    rgb=False
):
    input_colors_by_attr = {}
    translated_colors_by_attr = {}
    half_cycle_colors_by_attr = {}
    full_cycle_colors_by_attr = {}
    examples_by_attr = {}
    represented_input_hues_by_attr = {} # NOUVEAU

    for attr in analysis_attributes:
        input_bins = {}
        translated_bins = {}
        half_cycle_bins = {}
        full_cycle_bins = {}
        example_bins = {} # Sera une liste d'exemples par bin
        represented_hues_bins = {} # Sera un set de teintes par bin

        for bin_name in binning_config[attr]['bin_names']:
            # Structure pour les couleurs
            if not rgb:
                input_bins[bin_name] = {'H': []}
                translated_bins[bin_name] = {'H': []}
                half_cycle_bins[bin_name] = {'H': []}
                full_cycle_bins[bin_name] = {'H': []}
            else:
                input_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                translated_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                half_cycle_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                full_cycle_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
            
            # Structure pour les exemples et les teintes représentées
            example_bins[bin_name] = [] # Initialiser comme liste vide
            represented_hues_bins[bin_name] = set() # Initialiser comme set vide

        input_colors_by_attr[attr] = input_bins
        translated_colors_by_attr[attr] = translated_bins
        half_cycle_colors_by_attr[attr] = half_cycle_bins
        full_cycle_colors_by_attr[attr] = full_cycle_bins
        examples_by_attr[attr] = example_bins
        represented_input_hues_by_attr[attr] = represented_hues_bins # NOUVEAU

    return (input_colors_by_attr, translated_colors_by_attr, half_cycle_colors_by_attr,
            full_cycle_colors_by_attr, examples_by_attr, represented_input_hues_by_attr)




def save_binned_results(output_dir, input_colors_by_attr, translated_colors_by_attr, 
                        half_cycle_colors_by_attr, full_cycle_colors_by_attr, 
                        examples_by_attr, binning_config, analysis_attributes):
    """
    Save binned results to disk for later reuse.
    
    Args:
        output_dir: Directory to save results to
        input_colors_by_attr: Dict of input color distributions
        translated_colors_by_attr: Dict of translated color distributions
        half_cycle_colors_by_attr: Dict of half-cycle color distributions
        full_cycle_colors_by_attr: Dict of full-cycle color distributions
        examples_by_attr: Dict of example images by attribute and bin
        binning_config: Configuration used for binning
        analysis_attributes: List of attributes that were analyzed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dictionaries containing the binning results
    result_data = {
        'input_colors_by_attr': input_colors_by_attr,
        'translated_colors_by_attr': translated_colors_by_attr,
        'half_cycle_colors_by_attr': half_cycle_colors_by_attr,
        'full_cycle_colors_by_attr': full_cycle_colors_by_attr,
        'examples_by_attr': examples_by_attr,
        'binning_config': binning_config,
        'analysis_attributes': analysis_attributes
    }
    
    save_path = os.path.join(output_dir, 'binned_results.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(result_data, f)
    

def load_binned_results(output_dir):
    """
    Load previously saved binned results from disk.
    
    Args:
        output_dir: Directory where results were saved
        
    Returns:
        Tuple of (input_colors_by_attr, translated_colors_by_attr, 
                 half_cycle_colors_by_attr, full_cycle_colors_by_attr, 
                 examples_by_attr, binning_config, analysis_attributes)
        Returns None if no saved results are found
    """
    save_path = os.path.join(output_dir, 'binned_results.pkl')
    
    if not os.path.exists(save_path):
        print(f"No saved binned results found at {save_path}")
        return None
    
    
    

    with open(save_path, 'rb') as f:
        result_data = pickle.load(f)
    
    
    
    # Extract and return the individual components
    return (
        result_data['input_colors_by_attr'],
        result_data['translated_colors_by_attr'],
        result_data['half_cycle_colors_by_attr'],
        result_data['full_cycle_colors_by_attr'],
        result_data['examples_by_attr'],
        result_data['binning_config'],
        result_data['analysis_attributes']
    )

def comparison_metrics(values1, values2, num_bins):

        # KL(Bin1 || Bin2) - Order matters!
        kl_h_12 = kl_divergence(values1, values2)
        # KL(Bin2 || Bin1)
        kl_h_21 = kl_divergence(values2, values1)
        # Symmetric KL (Average)
        kl_h_sym = (kl_h_12 + kl_h_21) / 2.0 if np.isfinite(kl_h_12) and np.isfinite(kl_h_21) else np.inf

        # Determine the range for histograms
        min_val = min(np.min(values1) if len(values1) > 0 else 0, np.min(values2) if len(values2) > 0 else 0)
        max_val = max(np.max(values1) if len(values1) > 0 else 255, np.max(values2) if len(values2) > 0 else 255)
        
        # Create histograms and normalize to get probability distributions
        hist1, bin_edges = np.histogram(values1, bins=num_bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(values2, bins=num_bins, range=(min_val, max_val), density=True)
        
        # Get bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create arrays with bin values repeated according to frequencies
        # Multiply by 1000 and round to get integer counts
        binned_dist1 = np.repeat(bin_centers, np.round(hist1 * 1000).astype(int))
        binned_dist2 = np.repeat(bin_centers, np.round(hist2 * 1000).astype(int))
        
        # Normalize values to 0-1 range for KS test
        range_size = max_val - min_val
        if range_size > 0:
            binned_dist1_norm = (binned_dist1 - min_val) / range_size
            binned_dist2_norm = (binned_dist2 - min_val) / range_size
            
            # Perform KS test on the binned distributions
            ks_result = ks_2samp(binned_dist1_norm, binned_dist2_norm)
            ks_stat = ks_result.statistic
            ks_pval = ks_result.pvalue
        else:
            LOGGER.warning(f"Cannot normalize distributions - all values are the same: {min_val}")
            ks_stat = 0.0 if np.array_equal(values1, values2) else 1.0
            ks_pval = 1.0 if np.array_equal(values1, values2) else 0.0

        return kl_h_12, kl_h_21, kl_h_sym, ks_stat, ks_pval





binning_config_6144 = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 16,
        'range': (0, 2 * np.pi),
        'bin_names': [f'{i}x2pi_16' for i in range(0, 16)]
    },
    'size': {
        'n_bins': 8,
        'range': (7, 14),
        'bin_names': [f"{i}" for i in range(0, 8)]
    },
    'position_x': {
        'n_bins': 4,
        'range': (7, 25), # Assuming 32x32 images
        'bin_names': ['Left', 'Middle-Left', 'Middle-Right', 'Right']
    },
    'position_y': {
        'n_bins': 4,
        'range': (7, 25), # Assuming 32x32 images
        'bin_names': ['Bottom', 'Low-Middle', 'High-Middle', 'Top']
    }
}


default_binning_config = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 4,
        'range': (0, 2 * np.pi),
        'bin_names': ['0-90', '90-180', '180-270', '270-360'] # Adjusted for clarity
    },
    'size': {
        'n_bins': 4,
        'range': (7, 14),
        'bin_names': ['Very Small', 'Small', 'Medium', 'Large']
    },
    'position_x': {
        'n_bins': 2,
        'range': (0, 32), # Assuming 32x32 images
        'bin_names': ['Left', 'Right']
    },
    'position_y': {
        'n_bins': 2,
        'range': (0, 32), # Assuming 32x32 images
        'bin_names': ['Bottom', 'Top']
    }
}


debug_ = False






















BATCH_SIZE = 512 # Adjust this based on your GPU memory and model size


# --- Helper functions (adapted for batching) ---
def _prepare_batch_inputs(batch_samples: List[Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
    """
    Collates a list of individual sample data into batched tensors,
    adapting to the observed sample structure and original _prepare_inputs logic.

    Args:
        batch_samples: A list where each item is a single preprocessed sample
                       with keys like 'model_inputs' -> {'one_hot', 'attr_vector'}
                       and 'visual_ground_truth'.
        device: The target device (cuda or cpu).

    Returns:
        A tuple containing:
        - batched_one_hot: torch.Tensor (B, num_classes)
        - batched_attr_vector: torch.Tensor (B, attr_dim)
        - batched_visual_ground_truth: torch.Tensor (B, C, H, W)
        - batched_target_indices: torch.Tensor (B,) - for tilting logic on one_hot
        - valid_batch_samples: List[Any] - list of original sample dicts successfully processed

        Returns (None, None, None, None, None) if batch is empty or preparation fails for any sample.
        Note: Returning None for any failure simplifies the main loop error handling.
        If you need to process partial batches, the logic inside the main loop needs to be more complex.
    """
    if not batch_samples:
        return None, None, None, None, None

    batched_one_hot_list = []
    batched_attr_vector_list = []
    batched_visual_gt_list = []
    batched_target_indices_list = []
    valid_batch_samples = [] # Keep track of samples successfully processed

    for sample in batch_samples:
        try:
            # --- Access and process attribute tensors ---
            if 'model_inputs' not in sample or 'one_hot' not in sample['model_inputs'] or 'attr_vector' not in sample['model_inputs']:
                 raise KeyError("Missing 'model_inputs', 'one_hot', or 'attr_vector' key in sample['model_inputs']")

            # These might already be tensors on GPU based on your error message
            one_hot_tensor_one_sample = sample['model_inputs']['one_hot'] # Expected shape (1, num_classes)
            attr_vector_tensor_one_sample = sample['model_inputs']['attr_vector'] # Expected shape (1, attr_dim)

            if one_hot_tensor_one_sample.ndim != 2 or one_hot_tensor_one_sample.shape[0] != 1:
                print(f"Warning: one_hot tensor shape unexpected: {one_hot_tensor_one_sample.shape}. Expected (1, N). Skipping sample.")
                continue # Skip this sample

            if attr_vector_tensor_one_sample.ndim != 2 or attr_vector_tensor_one_sample.shape[0] != 1:
                 print(f"Warning: attr_vector tensor shape unexpected: {attr_vector_tensor_one_sample.shape}. Expected (1, M). Skipping sample.")
                 continue # Skip this sample


            # --- Convert PIL Image 'visual_ground_truth' to Tensor ---
            if 'visual_ground_truth' not in sample:
                 raise KeyError("Missing 'visual_ground_truth' key in sample")

            visual_gt_pil = sample['visual_ground_truth']
            if not isinstance(visual_gt_pil, Image.Image):
                 # Handle cases where visual_ground_truth might already be a tensor or array
                 print(f"Warning: visual_ground_truth is not a PIL Image for sample. Type: {type(visual_gt_pil)}. Skipping sample.")
                 continue # Skip this sample
            TT = ToTensor()
            # Use the global TT = ToTensor() instance
            visual_gt_tensor_one_sample = TT(visual_gt_pil)[:3].unsqueeze(0) # Converts to (1, C, H, W) float [0, 1]


            # --- Find the target index for tilting from the one_hot tensor ---
            # Assumes the one_hot tensor has a value of 1.0 at the target class index
            # We look for 1.0 in the (1, num_classes) tensor. .nonzero()[0] gives row, col indices.
            # We need the column index (dim=1).
            one_indices = (one_hot_tensor_one_sample == 1.0).nonzero(as_tuple=True) # Returns tuple of tensors for indices per dim
            if len(one_indices) == 0 or one_indices[1].numel() == 0:
                 # Handle case where original attr_input might not be a pure one-hot (e.g., all zeros)
                 print(f"Warning: Could not find target index (value 1.0) in one_hot input for sample: {sample}. Skipping sample.")
                 continue # Skip this sample
            # The column index of the 1.0 value is the target index
            target_index_one_sample = one_indices[1][0] # Should be a 0-dim tensor (scalar tensor)


            # Append the tensors/indices for this single sample to the lists
            # Ensure tensors are on the correct device (inputs might already be on GPU)
            batched_one_hot_list.append(one_hot_tensor_one_sample.to(device)) # (1, num_classes) -> to(device)
            batched_attr_vector_list.append(attr_vector_tensor_one_sample.to(device)) # (1, attr_dim) -> to(device)
            batched_visual_gt_list.append(visual_gt_tensor_one_sample.to(device)) # (1, C, H, W) -> to(device)
            batched_target_indices_list.append(target_index_one_sample.unsqueeze(0).to(device)) # 0-dim -> 1-dim (1,) -> to(device)

            # Add the original sample to the list of samples successfully prepared
            valid_batch_samples.append(sample)


        except KeyError as e:
             print(f"Error preparing sample for batching: {sample}. Missing key: {e}. Skipping sample.")
             continue # Skip this sample
        except Exception as e:
             print(f"Error preparing sample for batching: {sample}. Error: {e}. Skipping sample.")
             continue # Skip this sample

    # If no samples were valid in this batch
    if not valid_batch_samples:
         print(f"Warning: No valid samples prepared in batch starting at index {batch_samples[0].get('row_idx', 'N/A')}. Skipping batch.")
         return None, None, None, None, None # Return None for the whole batch

    # Stack the lists of tensors into batches
    try:
        # Resulting tensors will have shape (B, ...)
        batched_one_hot = torch.cat(batched_one_hot_list, dim=0) # (B, num_classes)
        batched_attr_vector = torch.cat(batched_attr_vector_list, dim=0) # (B, attr_dim)
        batched_visual_ground_truth = torch.cat(batched_visual_gt_list, dim=0) # (B, C, H, W)
        batched_target_indices = torch.cat(batched_target_indices_list, dim=0) # (B,)

    except Exception as e:
        print(f"Error stacking batch tensors: {e}. Check consistency of sample data shapes. Skipping batch.")
        return None, None, None, None, None # Return None for the whole batch on stacking error


    # All tensors are already on the device because we moved them per-sample inside the loop
    # but calling .to(device) again here is harmless and ensures it.
    # batched_one_hot = batched_one_hot.to(device)
    # batched_attr_vector = batched_attr_vector.to(device)
    # batched_visual_ground_truth = batched_visual_ground_truth.to(device)
    # batched_target_indices = batched_target_indices.to(device)


    # Return the batched tensors and the list of samples that were successfully processed
    return batched_one_hot, batched_attr_vector, batched_visual_ground_truth, batched_target_indices, valid_batch_samples
# Adapt process functions to accept and return BATched tensors (or NumPy arrays)
# Segment and extract color will be done *after* moving results back to CPU, batch-wise.

def _process_translation_path_batch(global_workspace: Any,
                                    batched_one_hot: torch.Tensor, # (B, num_classes)
                                    batched_attr_vector: torch.Tensor, # (B, attr_dim)
                                    batched_target_indices: torch.Tensor, # (B,)
                                    device: torch.device,
                                    class_value: float = 1.0,
                                    n: int = 1,
                                    selection_module = SingleDomainSelection(), # Assume SingleDomainSelection is defined
                                   ) -> torch.Tensor: # Return batched image tensor (B, C, H, W)

    # Ensure selection module is on device if it has parameters
    if hasattr(selection_module, 'to'):
        selection_module.to(device)

    # Clone the input one-hot batch to avoid modifying the original
    current_one_hot_batch = batched_one_hot.clone()  # (B, num_classes)
    current_attr_vector_batch = batched_attr_vector  # (B, attr_dim)
    batch_size, num_classes = current_one_hot_batch.shape

    # Add same random noise pattern to all one-hot encodings in the batch
    noise_scale = 0.01
    generator = torch.Generator()  # Garder sur CPU
    generator.manual_seed(42)

    # Générer le bruit sur CPU puis le déplacer sur le bon device
    noise = torch.randn(num_classes, generator=generator).unsqueeze(0) * noise_scale
    noise = noise.to(current_one_hot_batch.device)  # Déplacer sur le même device
    current_one_hot_batch += noise  # Broadcasting automatique

    # Ensure all values are non-negative
    current_one_hot_batch = torch.clamp(current_one_hot_batch, min=1e-8)
    
    # Normalize each one-hot encoding to ensure sum equals 1.0
    # Using sum along dim=1 to normalize per sample
    sums = current_one_hot_batch.sum(dim=1, keepdim=True)
    current_one_hot_batch = current_one_hot_batch / sums


    # Apply tilting if needed - vectorized on the one_hot batch
    if class_value != 1.0:
        modified_one_hot_batch = torch.zeros_like(current_one_hot_batch, device=device) # (B, num_classes)

        # Place class_value at the target index for each sample in the batch
        scatter_indices = batched_target_indices.unsqueeze(-1) # (B, 1)
        scatter_values = torch.full((batch_size, 1), class_value, device=device) # (B, 1)
        modified_one_hot_batch.scatter_(1, scatter_indices, scatter_values)

        # Calculate remaining probability per sample
        remaining = 1.0 - class_value

        if num_classes > 1:
            # Distribute remaining probability to non-target columns for each sample
            # Create a mask (B, num_classes) where True indicates non-target column for that sample
            col_indices = torch.arange(num_classes, device=device).unsqueeze(0).expand(batch_size, -1) # (B, num_classes)
            target_indices_expanded = batched_target_indices.unsqueeze(-1) # (B, 1)
            non_target_mask = (col_indices != target_indices_expanded) # (B, num_classes)

            # Calculate the value to distribute to each non-target class per sample
            # This assumes the number of non-target classes is constant across samples, which is true for fixed num_classes
            non_target_value_per_class = remaining / (num_classes - 1) # Scalar

            # Add the non-target value where the mask is True
            modified_one_hot_batch[non_target_mask] += non_target_value_per_class

        # Update the current one_hot batch
        current_one_hot_batch = modified_one_hot_batch # (B, num_classes)


    # Perform n cycles (Attribute Input -> GW -> Visual Latent -> GW -> Attribute Latent -> Attribute Input)
    # The "Attribute Input" structure is [one_hot_batch, attr_vector_batch]
    # All internal tensors/lists should have the batch dimension (B, ...)
    for _ in range(n):
        # Prepare the attribute input structure for global_workspace.encode_domain
        # This list structure [tensor(B,N), tensor(B,M)] is assumed to be handled by encode_domain
        current_attr_input_batch_list = [current_one_hot_batch, current_attr_vector_batch]

        # Encode Attribute Input -> GW
        # global_workspace.encode_domain must accept [tensor(B,N), tensor(B,M)] and return (B, attr_latent_dim)
        intermediate_attr_domain_batch = global_workspace.encode_domain(current_attr_input_batch_list, "attr") # Should be (B, attr_latent_dim)

        # encode_and_fuse must accept {"domain_name": (B, latent_dim)} and return (B, gw_latent_dim)
        attr_gw_latent_batch = global_workspace.gw_mod.encode_and_fuse(
            {"attr": intermediate_attr_domain_batch},
            selection_module=selection_module
        ) # Should be (B, gw_latent_dim)

        # Decode GW -> Visual Latent
        # decode must accept (B, gw_latent_dim) and return {"domain_name": (B, latent_dim), ...}
        v_latent_batch = global_workspace.gw_mod.decode(attr_gw_latent_batch)["v_latents"] # Should be (B, v_latent_dim)

        # For additional cycles, go back to attribute domain
        if _ < n - 1:
            # Encode Visual Latent -> GW
            gw_latent_from_v_batch = global_workspace.gw_mod.encode_and_fuse(
                {"v_latents": v_latent_batch},
                selection_module=selection_module
            ) # Should be (B, gw_latent_dim)

            # Decode GW -> Attribute Latent
            intermediate_attr_latent_batch = global_workspace.gw_mod.decode(gw_latent_from_v_batch)["attr"] # Should be (B, attr_latent_dim)

            # Decode Attribute Latent -> Attribute Input Representation
            # global_workspace.domain_mods["attr"].decode must accept (B, attr_latent_dim)
            # and return the attribute input structure, assumed to be [tensor(B,N), tensor(B,M)]
            decoded_attr_input_batch_list = global_workspace.domain_mods["attr"].decode(intermediate_attr_latent_batch)

            # Update the current attribute input tensors for the next cycle
            if isinstance(decoded_attr_input_batch_list, list) and len(decoded_attr_input_batch_list) == 2 and \
               isinstance(decoded_attr_input_batch_list[0], torch.Tensor) and isinstance(decoded_attr_input_batch_list[1], torch.Tensor):
                current_one_hot_batch = decoded_attr_input_batch_list[0]
                current_attr_vector_batch = decoded_attr_input_batch_list[1]
            else:
                # Handle unexpected output format from decode if necessary
                print(f"Warning: domain_mods['attr'].decode did not return expected [tensor, tensor] structure. Cycle {_ + 1} might fail or give incorrect results.")
                # You might need to break the loop or handle this case

    # Decode final Visual Latent -> Image
    # global_workspace.domain_mods["v_latents"].decode_images must accept (B, v_latent_dim) and return (B, C, H, W)
    translated_image_tensor_batch = global_workspace.domain_mods["v_latents"].decode_images(v_latent_batch) # (B, C, H, W)

    # Return batched tensor (still on GPU)
    return translated_image_tensor_batch


# --- process_half_cycle_path_batch (No changes needed from previous version if it only uses visual and GW) ---
# This function signature and logic should be correct as it only takes a single batched tensor input.
def _process_half_cycle_path_batch(global_workspace: Any,
                                   batched_v_latent_vector: torch.Tensor, # (B, v_latent_dim)
                                   device: torch.device,
                                  ) -> torch.Tensor: # Return batched image tensor (B, C, H, W)

    # Ensure selection module is on device (if it has parameters and is used within encode/decode)
    # If SingleDomainSelection is stateless, this might not be strictly necessary,
    # but it's safer to call .to(device) if there's any doubt.
    selection_module = SingleDomainSelection().to(device)


    # Encode Visual Latent -> GW
    # encode_and_fuse must handle {"v_latents": (B, v_latent_dim)} -> (B, gw_latent_dim)
    v_gw_latent_batch = global_workspace.gw_mod.encode_and_fuse({"v_latents": batched_v_latent_vector}, selection_module=selection_module)

    # Decode GW -> Visual Latent
    # decode must handle (B, gw_latent_dim) -> {"v_latents": (B, v_latent_dim)}
    half_cycle_v_latent_batch = global_workspace.gw_mod.decode(v_gw_latent_batch)["v_latents"] # Should be (B, v_latent_dim)

    # Decode Visual Latent -> Image
    # global_workspace.domain_mods["v_latents"].decode_images must accept (B, v_latent_dim) and return (B, C, H, W)
    half_cycle_image_tensor_batch = global_workspace.domain_mods["v_latents"].decode_images(half_cycle_v_latent_batch) # (B, C, H, W)

    # Return batched tensor (still on GPU)
    return half_cycle_image_tensor_batch


# --- process_full_cycle_path_batch (No changes needed from previous version if it only uses visual/GW/attribute LATENT) ---
# This function's logic assumes the attribute representation decoded from GW is a single tensor (B, attr_latent_dim),
# which seems consistent with how GW decode usually works for a domain.
def _process_full_cycle_path_batch(global_workspace: Any,
                                   batched_v_latent_vector: torch.Tensor, # (B, v_latent_dim)
                                   device: torch.device,
                                   n: int = 10,
                                   selection_module = SingleDomainSelection(), # Assume SingleDomainSelection is defined
                                  ) -> torch.Tensor: # Return batched image tensor (B, C, H, W)

    # Ensure selection module is on device
    if hasattr(selection_module, 'to'):
        selection_module.to(device)

    # Start with the original batched visual latent vector
    current_v_latent_batch = batched_v_latent_vector # (B, v_latent_dim)

    # Perform n cycles (Visual -> GW -> Attribute Latent -> GW -> Visual)
    # All internal tensors should have the batch dimension (B, ...)
    for _ in range(n):
        # Encode Visual Latent -> GW
        # encode_and_fuse must handle {"v_latents": (B, v_latent_dim)} -> (B, gw_latent_dim)
        gw_latent_from_v_batch = global_workspace.gw_mod.encode_and_fuse({"v_latents": current_v_latent_batch}, selection_module=selection_module) # Should be (B, gw_latent_dim)

        # Decode GW -> Attribute Latent
        # decode must handle (B, gw_latent_dim) -> {"attr": (B, attr_latent_dim), ...}
        intermediate_attr_latent_batch = global_workspace.gw_mod.decode(gw_latent_from_v_batch)["attr"] # Should be (B, attr_latent_dim)

        # Encode Attribute Latent -> GW
        # encode_and_fuse must handle {"attr": (B, attr_latent_dim)} -> (B, gw_latent_dim)
        gw_latent_intermediate_batch = global_workspace.gw_mod.encode_and_fuse({"attr": intermediate_attr_latent_batch}, selection_module=selection_module) # Should be (B, gw_latent_dim)

        # Decode GW -> Visual Latent
        # decode must handle (B, gw_latent_dim) -> {"v_latents": (B, v_latent_dim)}
        current_v_latent_batch = global_workspace.gw_mod.decode(gw_latent_intermediate_batch)["v_latents"] # Should be (B, v_latent_dim)

    # Decode final Visual Latent -> Image
    # global_workspace.domain_mods["v_latents"].decode_images must accept (B, v_latent_dim) and return (B, C, H, W)
    full_cycle_image_tensor_batch = global_workspace.domain_mods["v_latents"].decode_images(current_v_latent_batch) # (B, C, H, W)

    # Return batched tensor (still on GPU)
    return full_cycle_image_tensor_batch

# --- How to call the new batched function ---
# Make sure preprocessed_samples, global_workspace, etc. are defined
# final_results = process_samples_batched(
#     preprocessed_samples=preprocessed_samples,
#     global_workspace=global_workspace,
#     device=device,
#     BATCH_SIZE=BATCH_SIZE,
#     reverb_n=reverb_n,
#     tilting=tilting,
#     barycentre=barycentre,
#     debug=debug,
#     debug_=debug_ # Will be ignored if debug is False
# )

# You would then use final_results which is the list of dictionaries.

def save_training_params_pickle(config, project_name, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    root_dir = Path.cwd()
    log_dir = root_dir / "checkpoints" / project_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = log_dir / f"config_{timestamp}.pkl"

    with open(file_path, 'wb') as f:
        pickle.dump(config, f)
    
    return file_path

def get_project_root():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

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













from typing import Dict, List, Optional, Tuple
import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any, cast
from collections.abc import Sequence
from PIL import Image   
import numpy as np
import matplotlib.pyplot as plt
import warnings

def visualize_examples_by_attribute(
    # The list contains DICTIONARIES
    examples: Dict[str, List[Dict[str, Any]]],
    attribute_name: str,
    bin_names: List[str],
    output_path: str,
    max_per_bin: int = 5,
    # *** List of keys for images to show per example ***
    image_keys_to_display: List[str] = ['input_image', 'translated_image', 'half_cycle_image', 'full_cycle_image']
) -> None:
    """
    Ultra-minimal function to display multiple example images per instance,
    stored in dictionaries. Images defined by 'image_keys_to_display'.
    WARNING: Lacks robust error handling and type conversion beyond basics.
    """
    # --- 1. Filter and Setup ---
    # Get list of dictionaries for valid bins
    valid_examples = {bn: examples.get(bn, [])
                      for bn in bin_names if examples.get(bn)}
    valid_bin_names = list(valid_examples.keys())

    if not valid_bin_names: return

    # Determine max examples based on *requested* max_per_bin, not actual count yet
    max_examples_to_show = max_per_bin

    num_img_keys = len(image_keys_to_display)
    if num_img_keys == 0:
         warnings.warn("No image keys provided in 'image_keys_to_display'. Cannot visualize examples.", stacklevel=2)
         return

    n_bins = len(valid_bin_names)
    # *** Adjust columns based on examples * keys ***
    total_cols = max_examples_to_show * num_img_keys
    fig, axes = plt.subplots(n_bins, total_cols,
                             figsize=(total_cols * 1.5, n_bins * 1.7), # Adjusted size
                             squeeze=False)

    # --- 2. Iterate and Display ---
    for r, bin_name in enumerate(valid_bin_names):
        list_of_example_dicts = valid_examples[bin_name]
        num_actual_examples_in_bin = len(list_of_example_dicts)

        # Set Y-label for the row (aligned with the first image column)
        axes[r, 0].set_ylabel(bin_name, fontsize=10)

        for c_example in range(max_examples_to_show): # Iterate up to max examples requested
            if c_example < num_actual_examples_in_bin:
                example_dict = list_of_example_dicts[c_example]

                # Iterate through the image keys for this example dict
                for c_img_key, img_key in enumerate(image_keys_to_display):
                    actual_col_index = c_example * num_img_keys + c_img_key
                    ax = axes[r, actual_col_index]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # Set a title for each individual image (e.g., "Input", "Trans")
                    ax.set_title(img_key.replace('_',' ').rsplit(' ',1)[0].title(), fontsize=8) # Basic title generation

                    processed_img = None
                    error_msg = None

                    try:
                        img_data_raw = example_dict.get(img_key)

                        if img_data_raw is None:
                            error_msg = f"Key\nMissing"
                        elif isinstance(img_data_raw, np.ndarray):
                            if img_data_raw.dtype == object:
                                processed_img = np.array(img_data_raw.tolist(), dtype=np.float32)
                            elif np.issubdtype(img_data_raw.dtype, np.number):
                                processed_img = img_data_raw
                        elif isinstance(img_data_raw, Image.Image):
                            processed_img = np.array(img_data_raw)

                        # Basic Checks & Display
                        if processed_img is not None:
                            if processed_img.ndim < 2 :
                                 error_msg = f"Invalid Dim\n{processed_img.ndim}D"
                            else:
                                if np.issubdtype(processed_img.dtype, np.floating) and processed_img.size > 0 and processed_img.max() > 1.1:
                                     processed_img = np.clip(processed_img / 255.0, 0, 1)
                                ax.imshow(processed_img)
                        else:
                            # If image data was missing or couldn't be processed minimally
                            if error_msg is None: error_msg = f"Type Err:\n{type(img_data_raw).__name__}"
                            ax.text(0.5, 0.5, error_msg, ha='center', va='center', fontsize=8, color='red')
                            # Log once per failed image display attempt
                            warnings.warn(f"Cannot display image for {attribute_name}/{bin_name}[{c_example}][{img_key}]: {error_msg.replace(' ','')}", stacklevel=2)

                    except Exception as e:
                        err_type = type(e).__name__
                        warnings.warn(f"Error displaying image {attribute_name}/{bin_name}[{c_example}][{img_key}]: {err_type}", stacklevel=2)
                        ax.text(0.5, 0.5, f"Error:\n{err_type}", ha='center', va='center', fontsize=8, color='red')

            else:
                 # Handle empty slots for examples beyond what the current bin has
                 for c_img_key in range(num_img_keys):
                    actual_col_index = c_example * num_img_keys + c_img_key
                    if actual_col_index < total_cols: # Boundary check
                         axes[r, actual_col_index].axis('off')

    # --- 3. Finalize and Save ---
    # Construct a title indicating the keys shown
    keys_str = ', '.join([k.split('_')[0] for k in image_keys_to_display])
    plt.suptitle(f"{attribute_name.title()} Examples ({keys_str})", fontsize=12)
    # Adjust layout more aggressively if needed
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=0.8, w_pad=0.5) # Add padding adjustment

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100)
        # Optional logging: LOGGER.info(f"Saved examples: {output_path}")
    except Exception as e:
        warnings.warn(f"Save failed for {output_path}: {e}", stacklevel=2)
    finally:
        plt.close(fig)

def visualize_color_distributions_by_attribute(
    data: Dict[str, Dict[str, List[float]]],
    attribute_name: str,
    bin_names: List[str],
    output_path: str, 
    channels = ["R", "G", "B"], 
    num_bins: int = 50, 
) -> None:
    """
    Create a visualization of color distributions grouped by attribute bins.
    """
    valid_bins = []
    valid_bin_names = []
    for bin_idx, bin_name in enumerate(bin_names):
        if bin_name in data and any(channel in data[bin_name] and len(data[bin_name][channel]) > 0 
                                   for channel in channels):
            valid_bins.append(bin_idx)
            valid_bin_names.append(bin_name)

    n_valid_bins = len(valid_bins)
    if n_valid_bins == 0:
        warnings.warn(f"No valid data for {attribute_name} color distributions")
        return

    fig, axes = plt.subplots(n_valid_bins, len(channels), figsize=(len(channels) * 3, n_valid_bins * 2))
    
    # Handle different axes shapes based on dimensions
    if n_valid_bins == 1 and len(channels) == 1:
        axes = np.array([[axes]])  # Make it a 2D array with a single element
    elif n_valid_bins == 1:
        axes = np.array([axes])  # Make it a 2D array with one row
    elif len(channels) == 1:
        axes = axes.reshape(-1, 1)  # Make it a 2D array with one column

    for out_idx, bin_idx in enumerate(valid_bins):
        bin_name = bin_names[bin_idx]
        for ch_idx, channel in enumerate(channels):
            ax = axes[out_idx][ch_idx]
            if bin_name in data and channel in data[bin_name] and len(data[bin_name][channel]) > 0:
                filtered_data = [x for x in data[bin_name][channel] if not np.isnan(x)]
                if filtered_data:
                    ax.hist(filtered_data, bins=num_bins, alpha=0.7)
                    if ch_idx == 0:
                        ax.set_ylabel(bin_name)
                    ax.set_title(f"Channel {channel}")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_xlim(0, 255)

    plt.suptitle(f"Color distributions by {attribute_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_input_output_distributions(
    input_data: Dict[str, List[float]],
    output_data: Dict[str, List[float]],
    bin_name: str,
    attribute_name: str,
    output_path: str, 
    channels= ["R", "G", "B"]
) -> None:
    """
    Create a visualization comparing input and output color distributions for a specific bin.
    """

    has_valid_data = any(
        (channel in input_data and any(not np.isnan(x) for x in input_data[channel])) or
        (channel in output_data and any(not np.isnan(x) for x in output_data[channel]))
        for channel in channels
    )
    if not has_valid_data:
        warnings.warn(f"No valid data for {attribute_name} - {bin_name}")
        return

    fig, axes = plt.subplots(1, len(channels), figsize=(len(channels) * 4, 3))
    
    # Handle case with single channel
    if len(channels) == 1:
        axes = [axes]
        
    for ch_idx, channel in enumerate(channels):
        ax = axes[ch_idx]
        if channel in input_data and input_data[channel]:
            filtered_input = np.array([x for x in input_data[channel] if not np.isnan(x)])
            if filtered_input.size:
                hist_in, bin_edges = np.histogram(filtered_input, bins=30, range=(0, 255), density=False)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.bar(bin_centers, hist_in, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, label="Input", color='blue')
        if channel in output_data and output_data[channel]:
            filtered_output = np.array([x for x in output_data[channel] if not np.isnan(x)])
            if filtered_output.size:
                hist_out, bin_edges = np.histogram(filtered_output, bins=30, range=(0, 255), density=False)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.bar(bin_centers, hist_out, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, label="Output", color='red')
        ax.set_title(f"Channel {channel}")
        ax.set_xlim(0, 255)
        ax.legend()

    plt.suptitle(f"{attribute_name} - {bin_name}: Input vs Output Color Distributions")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_kl_heatmap(
    kl_values: Dict[str, Dict[str, float]],
    attribute_name: str,
    bin_names: List[str],
    output_path: str,
    channels: List[str] = ["R", "G", "B"],
    title: str = None
) -> None:
    """
    Create a heatmap visualization of KL divergence between input and output distributions.
    
    Args:
        kl_values: Dictionary of KL divergence values by bin and channel
        attribute_name: Name of the attribute being visualized
        bin_names: List of bin names
        output_path: Path to save the output image
        channels: List of channel names to visualize
        title: Custom title for the plot (if None, a default title is generated)
    """
    valid_bins = []
    valid_bin_names = []
    for bin_idx, bin_name in enumerate(bin_names):
        if bin_name in kl_values and any(channel in kl_values[bin_name] and not np.isnan(kl_values[bin_name][channel])
                                         for channel in channels):
            valid_bins.append(bin_idx)
            valid_bin_names.append(bin_name)

    n_valid_bins = len(valid_bins)
    if n_valid_bins == 0:
        warnings.warn(f"No valid data for {attribute_name} KL divergence heatmap")
        return

    kl_matrix = np.full((n_valid_bins, len(channels)), np.nan)
    for out_idx, bin_idx in enumerate(valid_bins):
        bin_name = bin_names[bin_idx]
        if bin_name in kl_values:
            for ch_idx, channel in enumerate(channels):
                if channel in kl_values[bin_name] and not np.isnan(kl_values[bin_name][channel]):
                    kl_matrix[out_idx, ch_idx] = kl_values[bin_name][channel]

    # Adjust figure size based on number of channels
    width = max(4, len(channels) * 2)
    fig, ax = plt.subplots(figsize=(width, n_valid_bins * 0.8 + 2))
    
    valid_data = kl_matrix[~np.isnan(kl_matrix)]
    if valid_data.size:
        vmin, vmax = np.min(valid_data), np.max(valid_data)
        im = ax.imshow(kl_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(np.zeros((n_valid_bins, len(channels))), cmap='viridis')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("KL Divergence")
    ax.set_xticks(np.arange(len(channels)))
    ax.set_yticks(np.arange(n_valid_bins))
    ax.set_xticklabels(channels)
    ax.set_yticklabels(valid_bin_names)
    
    # Use custom title if provided, otherwise generate default title
    if title is None:
        if len(channels) == 1:
            title = f"KL Divergence: Input vs Output for {channels[0]} Channel - {attribute_name}"
        else:
            title = f"KL Divergence: Input vs Output Color Distributions for {attribute_name}"
    
    ax.set_title(title)

    for i in range(n_valid_bins):
        for j in range(len(channels)):
            if not np.isnan(kl_matrix[i, j]):
                mean_val = np.nanmean(valid_data) if valid_data.size else 0
                text_color = "w" if kl_matrix[i, j] > mean_val / 2 else "black"
                ax.text(j, i, f"{kl_matrix[i, j]:.2f}", ha="center", va="center", color=text_color)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


# TODO : TEST function 
def visualize_distribution_comparison(
    dist1_data: Dict[str, List[float]],
    dist2_data: Dict[str, List[float]],
    dist1_label: str,
    dist2_label: str,
    bin_name: str, # Can represent the single bin or the pair being compared
    attr: str,
    save_path: str,
    channels: List[str] = ["H"], # Currently focuses on H, allows future extension
    title: Optional[str] = None,
    num_bins: int = 50, # Number of bins for the histogram
    hist_range: Optional[Tuple[float, float]] = (0, 360) # Default range for Hue (degrees)
):
    """
    Generates and saves a histogram plot comparing distributions for specified channels.

    Currently designed primarily for comparing the 'H' channel between two distributions.

    Args:
        dist1_data: Dictionary containing data for the first distribution.
                    Expected format: {'H': [val1, val2,...], ...}
        dist2_data: Dictionary containing data for the second distribution.
                    Expected format: {'H': [val1, val2,...], ...}
        dist1_label: Label for the first distribution (e.g., "Input H", "Translated H (Bin A)").
        dist2_label: Label for the second distribution (e.g., "Translated H", "Translated H (Bin B)").
        bin_name: Name of the bin or description of the comparison (e.g., "Diamond", "Bin A vs Bin B").
        attr: Name of the attribute being analyzed (e.g., "shape").
        save_path: Full path where the plot image will be saved.
        channels: List of channel keys to plot (currently only 'H' is implemented effectively).
        title: Optional custom title for the plot. If None, a default title is generated.
        num_bins: Number of bins to use in the histograms.
        hist_range: Tuple specifying the (min, max) range for the histogram X-axis.
                    Defaults to (0, 360) suitable for Hue in degrees.
    """
    
    output_path = Path(save_path)
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Attempting to create directory: {output_path}") 
    # --- Data Preparation ---
    for channel in channels:
        
        # Extract H channel data, removing None/NaN
        values1 = np.array([x for x in dist1_data.get(channel, []) if x is not None and not np.isnan(x)])
        values2 = np.array([x for x in dist2_data.get(channel, []) if x is not None and not np.isnan(x)])

        if len(values1) == 0 and len(values2) == 0:
            warnings.warn.warning(f"No valid {channel} data for either distribution ({dist1_label}, {dist2_label}) "
                           f"for attribute '{attr}', bin '{bin_name}'. Skipping plot.")
            return
        elif len(values1) == 0:
             warnings.warn.warning(f"No valid {channel} data for distribution 1 ({dist1_label}) "
                            f"for attribute '{attr}', bin '{bin_name}'. Plotting only distribution 2.")
        elif len(values2) == 0:
             warnings.warn.warning(f"No valid {channel} data for distribution 2 ({dist2_label}) "
                            f"for attribute '{attr}', bin '{bin_name}'. Plotting only distribution 1.")


        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        if channel == "H":
            # Set histogram range for Hue channel
            hist_range = (0, 360)
            # Set histogram range for RGB channels
        else:
            hist_range = (0, 255)
        # Plot histograms using density=True to compare shapes independent of sample size
        if len(values1) > 0:
            ax.hist(values1, bins=num_bins, range=hist_range, density=True,
                    alpha=0.7, label=f"{dist1_label} (N={len(values1)})")

        if len(values2) > 0:
            ax.hist(values2, bins=num_bins, range=hist_range, density=True,
                    alpha=0.7, label=f"{dist2_label} (N={len(values2)})")


        # --- Formatting ---
        ax.set_xlabel(f"{channel} Value")
        ax.set_ylabel(f"Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if title is None:
            # Generate a default title if none provided
            title = f"{channel}-Channel Distribution Comparison: {dist1_label} vs {dist2_label}\nAttribute: {attr.title()}, Bin(s): {bin_name}"
        ax.set_title(title, wrap=True)

        # --- Saving ---
        file_name = f'{channel}_comparison.png'
        output_file = os.path.join(output_path, file_name)  # Use a different variable name
        # Ensure output directory exists (optional, could be handled by caller)
        output_file_str = str(output_file) 
        plt.tight_layout()
        plt.savefig(output_file_str, bbox_inches='tight')
        if channel == "H":
            # print(f"DEBUG: Saved histogram for {channel} channel to {output_file_str}")
            print(f"Saved distribution comparison plot to {output_file}")
        # Close the figure to free memory, especially important in loops
        plt.close(fig)


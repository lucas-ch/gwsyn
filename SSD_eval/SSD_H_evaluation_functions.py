import collections.abc
from pathlib import Path
import os
import math
import ast
import io
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union, cast, Literal

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import patches
import cv2
import pandas as pd
from PIL import Image

from tqdm import tqdm
from itertools import combinations

from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from SSD_utils import (
    generate_fixed_colors,
    normalize_position,
    normalize_size,
    normalize_rotation,
    kl_divergence,
    bin_attribute,
    segment_shape,
    initialize_h_binning_structures,
    bin_h_processed_samples_with_paths, # Assuming this exists and handles 3 paths
    preprocess_dataset, # Assuming this exists
    process_through_global_workspace,
    save_binned_results,
    load_binned_results,
    comparison_metrics,
    default_binning_config,
    binning_config_6144,
)
from SSD_visualize_functions import (
    visualize_color_distributions_by_attribute,
    visualize_input_output_distributions, # Should be adaptable or ,
    visualize_examples_by_attribute, 
    visualize_distribution_comparison,
)

# Mock LOGGER if not imported
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add json import at the top of the file if not already present
import json

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper function to perform comparison for a single bin and path
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def extract_valid_values(data_dict, channels=None):
    """Extract valid values from distribution dictionary for specified channels."""
    
    if channels is None:
        print("No channels specified, defaulting to H channel.")
        channels = ["H"]
    
    result = {}
    for channel in channels:
        values = np.array([x for x in data_dict.get(channel, []) 
                         if x is not None and not np.isnan(x)])
        result[channel] = values
    
    return result

def calculate_distribution_metrics(values1, values2, channels=None):
    """Calculate KL divergence and KS test for the given distributions."""
    if channels is None:
        print( "No channels specified, defaulting to H channel.")
        channels = ["H"]
    
    results = {}
    
    try:
        # Calculate metrics for each channel
        for channel in channels:
            vals1 = values1.get(channel, np.array([]))
            vals2 = values2.get(channel, np.array([]))
            
            if len(vals1) <= 1 or len(vals2) <= 1:
                continue
                
            # KL divergence
            results[channel] = kl_divergence(vals1, vals2)
            
            # KS test
            ks_result = ks_2samp(vals1, vals2)
            results[f'ks_stat_{channel.lower()}'] = ks_result.statistic
            
            # Only store pval once since we mainly care about H channel
            if channel == "H":
                results['ks_pval'] = ks_result.pvalue
    
    except Exception as e:
        LOGGER.error(f"Failed to calculate metrics: {e}", exc_info=True)
    
    return results

def visualize_distributions(dist1, dist2, label1, label2, bin_name, attr, 
                           save_path, channels=None, num_bins=50, title=None):
    """Wrapper for visualization function with error handling."""
    try:
        if 'visualize_distribution_comparison' in globals():
            visualize_distribution_comparison(
                dist1_data=dist1,
                dist2_data=dist2,
                dist1_label=label1,
                dist2_label=label2,
                bin_name=bin_name,
                attr=attr,
                save_path=save_path,
                channels=channels or ["H"],
                num_bins=num_bins,
                title=title
            )
        else:
            LOGGER.warning("`visualize_distribution_comparison` function not found. Skipping plot generation.")
    except Exception as e:
        LOGGER.error(f"Failed to generate comparison plot: {e}", exc_info=True)

def _compare_distributions_for_bin(
    dist1,
    dist2,
    bin_name,
    attr,
    comparison_label,
    output_dir_for_comparison,
    rgb=True
):
    """
    Compares two distributions for a single bin, visualizing and calculating metrics.
    
    Args:
        dist1: Dictionary containing data for the FIRST distribution
        dist2: Dictionary containing data for the SECOND distribution
        bin_name: Name of the current bin
        attr: Name of the attribute being analyzed
        comparison_label: String identifying the comparison
        output_dir_for_comparison: Directory to save visualizations
        rgb: Whether to process RGB channels in addition to H
        
    Returns:
        Dictionary with metrics or None if data is insufficient
    """
    # 1. Determine labels for visualization
    label1_str, label2_str = "Dist 1", "Dist 2"  # Defaults
    
    if '_vs_' in comparison_label:
        try:
            part1, part2 = comparison_label.split('_vs_')
            label1_str = part1.replace('_', ' ').title()
            label2_str = part2.replace('_', ' ').title()
        except ValueError:
            LOGGER.warning(f"Could not parse comparison label '{comparison_label}'. Using default labels.")
    else:
        label1_str = "Input"
        label2_str = comparison_label.replace('_', ' ').title()
    
    LOGGER.debug(f"Comparing '{label1_str}' vs '{label2_str}' for attr '{attr}', bin '{bin_name}'")
    
    # 2. Extract valid data
    channels = ["H", "R", "G", "B"] if rgb else ["H"]
    values1_dict = extract_valid_values(dist1, channels)
    values2_dict = extract_valid_values(dist2, channels)
    
    # Check if we have enough data in the H channel
    if len(values1_dict.get("H", [])) <= 1 or len(values2_dict.get("H", [])) <= 1:
        LOGGER.warning(f"Skipping bin '{bin_name}' for {attr} ({comparison_label}) - No valid H data "
                      f"({label1_str}: {len(values1_dict.get('H', []))}, "
                      f"{label2_str}: {len(values2_dict.get('H', []))})")
        return None
    
    LOGGER.info(f"  - Comparing {comparison_label} for bin '{bin_name}': "
               f"{len(values1_dict['H'])} '{label1_str}' vs {len(values2_dict['H'])} '{label2_str}' H values.")
    
    # 3. Visualize comparison
    os.makedirs(output_dir_for_comparison, exist_ok=True)
    safe_bin_name = bin_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    didier = os.path.join(output_dir_for_comparison, f'{comparison_label}_{safe_bin_name}')
    
    visualize_distributions(
        dist1=dist1,
        dist2=dist2,
        label1=label1_str,
        label2=label2_str,
        bin_name=bin_name,
        attr=attr,
        save_path=didier,
        channels=channels
    )
    
    # 4. Calculate metrics
    metrics = calculate_distribution_metrics(values1_dict, values2_dict, channels)
    
    # Log results
    if rgb:
        LOGGER.info(f"    KL Div ({label1_str} || {label2_str}) "
                   f"R: {metrics.get('R', np.inf):.4f}, "
                   f"G: {metrics.get('G', np.inf):.4f}, "
                   f"B: {metrics.get('B', np.inf):.4f}")
        LOGGER.info(f"    K-S test ({label1_str} vs {label2_str}) "
                   f"R: stat={metrics.get('ks_stat_r', np.nan):.4f}, "
                   f"G: stat={metrics.get('ks_stat_g', np.nan):.4f}, "
                   f"B: stat={metrics.get('ks_stat_b', np.nan):.4f}, "
                   f"p={metrics.get('ks_pval', np.nan):.4f}")
    
    LOGGER.info(f"    KL Div ({label1_str} || {label2_str}) H: {metrics.get('H', np.inf):.4f}")
    LOGGER.info(f"    K-S test ({label1_str} vs {label2_str}) H: "
               f"stat={metrics.get('ks_stat_h', np.nan):.4f}, p={metrics.get('ks_pval', np.nan):.4f}")
    
    return metrics

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Updated HueShapeAnalyzer class using the modified functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HueShapeAnalyzer:
    """Class to analyze shape data across multiple attribute dimensions using Hue channel."""

    def __init__(
        self,
        global_workspace, # Assume this is the trained model/pipeline
        device: torch.device,
        shape_names: List[str] = ["diamond", "egg", "triangle"],
        color: bool = True,
        output_dir: str = ".",
        seed=0,
        debug = False, 
        num_bins = 50,
        reverb_n = 1, # Number of reverberation cycles for full cycle and translation path
        binning_config: Dict = default_binning_config, # Use the default config
        rgb = True,
        barycentre = False,
    ):
        """
        Initialize the hue shape analyzer.
        """
        self.global_workspace = global_workspace
        self.device = device
        self.shape_names = shape_names
        self.color = color # Controls if color attributes are used in input vectors
        self.output_dir = output_dir
        self.seed = seed
        self.debug = debug
        self.num_bins = num_bins # Number of bins for histogram visualization adb statistical tests
        self.reverb_n = reverb_n # Number of reverberation cycles for full cycle and translation path
        os.makedirs(self.output_dir, exist_ok=True)
        # Use a fixed set of colors for reproducibility if needed
        self.rgb_colors, self.hls_colors = generate_fixed_colors(100)
        self.binning_config = binning_config # Use the default config
        self.rgb = rgb # Flag to indicate if RGB colors are used
        self.barycentre = barycentre # Flag to indicate if barycentre is used

    def _process_csv(
        self,
        csv_path: str,
        attributes: List[str], # analysis_attributes
        use_fixed_reference: bool = False,
        reference_color_idx: int = 0,
        im_dir: str = "./evaluation_set",  # Added default
        tilting = 1
    ) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
        """
        Load the CSV file, preprocess all samples, and process them through the global workspace.
        """
        df = pd.read_csv(csv_path)
        # Handle potential string representation of lists/tuples
        if isinstance(df["location"].iloc[0], str):
             df["location"] = df["location"].apply(ast.literal_eval)
        if use_fixed_reference:
            fixed_color = self.rgb_colors[reference_color_idx]
            # Ensure it's a list of lists/tuples if needed by downstream code
            df["fixed_color"] = [list(fixed_color)] * len(df) # Example: store as list

        LOGGER.info(f"Preprocessing data from {csv_path}...")

        
        preprocessed_samples = preprocess_dataset(
            df,
            attributes,
            self.shape_names,
            self.color, # Use instance attribute
            self.rgb_colors,
            self.device,
            fixed_reference=use_fixed_reference,
            reference_color_idx=reference_color_idx,
            im_dir=im_dir
        )
        LOGGER.info(f"Processing {len(preprocessed_samples)} samples through global workspace...")
        # Assuming process_through_global_workspace returns dicts with keys like:
        # 'translated_shape_color', 'half_cycle_shape_color', 'full_cycle_shape_color',
        # 'translated_image', 'half_cycle_image', 'full_cycle_image' etc.
        processed_samples = process_through_global_workspace(
            self.global_workspace,
            preprocessed_samples,
            self.device,
            debug = self.debug, 
            reverb_n=self.reverb_n,
            tilting=tilting,
            barycentre=self.barycentre
        )
        return df, preprocessed_samples, processed_samples

    def analyze_dataset(
        self,
        csv_path: str,
        analysis_attributes: List[str] = None,
        display_examples: bool = True,
        seed=None,
        binning_config=None,
        im_dir: str = "./evaluation_set",  # Added default image dir
        tilting = 1, 
        parrallel = False, 
        conditioning = False # Added parallel processing flag
    ) -> Dict[str, Any]:
        """
        Analyze a dataset of shape images using Hue channel across multiple processing paths.

        Args:
            csv_path (str): Path to the dataset CSV file.
            analysis_attributes (List[str], optional): List of attributes to analyze.
                Defaults to ['shape', 'rotation', 'size', 'position_x', 'position_y'].
            display_examples (bool, optional): Whether to store and visualize example images per bin.
                Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to instance seed.
            binning_config (Dict, optional): Configuration for binning attributes.
                Defaults to instance default config.
            im_dir (str, optional): Directory containing the images referenced by the CSV.
                Defaults to "./evaluation_set".


        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, including
                            KL divergences for each path (input vs translated, vs half-cycle,
                            vs full-cycle) and the binned color distributions.
        """
        if analysis_attributes is None:
            analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        if binning_config is None:
            binning_config = self.binning_config # Use instance default
        if seed is None:
            seed = self.seed
        seed_everything(seed) # Set seed if using PyTorch Lightning

        results = {attr: {} for attr in analysis_attributes} # Initialize results dict

        LOGGER.info(f"Starting analysis for {csv_path}")

        save_path = os.path.join(self.output_dir, 'binned_results.pkl')

        LOGGER.info("Initializing binning structures...")
            # Initialize binning structures for H channel, now expecting structures for all paths
            # (Make sure initialize_h_binning_structures creates dicts for all 4 sets of colors)
        (
            input_colors_by_attr,
            translated_colors_by_attr, # Renamed for clarity
            half_cycle_colors_by_attr,
            full_cycle_colors_by_attr,
            examples_by_attr, 
            represented_hues_bins,
        ) = initialize_h_binning_structures(
            analysis_attributes,
            binning_config, self.rgb
        )
        if os.path.exists(save_path):
            LOGGER.warning(f"Loading existing binned results from {save_path}")


            (
                input_colors_by_attr,
                translated_colors_by_attr,
                half_cycle_colors_by_attr,
                full_cycle_colors_by_attr,
                examples_by_attr,
                binning_config,
                analysis_attributes
            ) = load_binned_results(self.output_dir)
            

        else : 
            df, preprocessed_samples, processed_samples = self._process_csv(
                csv_path,
                analysis_attributes, # Pass attributes for preprocessing reference
                use_fixed_reference=False, # Standard analysis uses original colors
                im_dir=im_dir, 
                tilting=tilting
            )

            

            LOGGER.info("Binning processed samples...")
            # Bin processed samples for H channel analysis across all paths
            # (Make sure bin_h_processed_samples_with_paths populates all 4 color dicts correctly)
            bin_h_processed_samples_with_paths(
                preprocessed_samples=preprocessed_samples,
                processed_samples=processed_samples,
                analysis_attributes=analysis_attributes,
                binning_config=binning_config,
                input_colors_by_attr=input_colors_by_attr,
                translated_colors_by_attr=translated_colors_by_attr, # Pass the correct target dict
                half_cycle_colors_by_attr=half_cycle_colors_by_attr, # Pass the correct target dict
                full_cycle_colors_by_attr=full_cycle_colors_by_attr, # Pass the correct target dict
                examples_by_attr=examples_by_attr,
                display_examples=display_examples,
                # represented_input_hues_by_attr=represented_hues_bins, # Pass the bins for H channel
                rgb=self.rgb # Pass the conditioning flag
            )
            LOGGER.info("Binning complete, saving the results")
            # Save the results to the output directory
            save_binned_results(self.output_dir, input_colors_by_attr, translated_colors_by_attr, 
                            half_cycle_colors_by_attr, full_cycle_colors_by_attr, 
                            examples_by_attr, binning_config, analysis_attributes)


        LOGGER.info("Processing analysis attributes and comparing paths...")
        # Process analysis attributes using the new multi-path function
        if parrallel:
            results = process_analysis_attributes_parallel(
                analysis_attributes=analysis_attributes,
                output_dir=self.output_dir,
                color=self.color, # Pass color flag for directory naming
                binning_config=binning_config,
                input_colors_by_attr=input_colors_by_attr,
                translated_colors_by_attr=translated_colors_by_attr, # Pass renamed dict
                half_cycle_colors_by_attr=half_cycle_colors_by_attr,
                full_cycle_colors_by_attr=full_cycle_colors_by_attr,
                examples_by_attr=examples_by_attr,
                results=results,  # Pass the results dict to be updated
                num_bins = self.num_bins, # Pass the number of bins for visualization
                significance_alpha=0.05, # Pass the significance threshold
                rgb=self.rgb,
                conditioning = conditioning # Pass the RGB flag, 
            )
        else : 
            results = process_analysis_attributes(
                analysis_attributes=analysis_attributes,
                output_dir=self.output_dir,
                color=self.color, # Pass color flag for directory naming
                binning_config=binning_config,
                input_colors_by_attr=input_colors_by_attr,
                translated_colors_by_attr=translated_colors_by_attr, # Pass renamed dict
                half_cycle_colors_by_attr=half_cycle_colors_by_attr,
                full_cycle_colors_by_attr=full_cycle_colors_by_attr,
                examples_by_attr=examples_by_attr,
                results=results,  # Pass the results dict to be updated
                num_bins = self.num_bins, # Pass the number of bins for visualization
                significance_alpha=0.05, # Pass the significance threshold
                rgb=self.rgb, # Pass the RGB flag, 
                conditioning=conditioning
            )

        LOGGER.info(f"Analysis complete. Results saved in {self.output_dir}")
        return results


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW Heatmap function for bin-pair comparisons (KL or KS)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def visualize_metric_heatmap_bin_pairs(
    metric_results: Dict[Tuple[str, str], Dict[str, float]], # e.g., {('bin1', 'bin2'): {'kl_symmetric': val, ...}}
    metric_key: str, # e.g., 'kl_symmetric', 'ks_pval'
    bin_names: List[str],
    attribute: str,
    path_name: str,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    invert_cmap: bool = False, # e.g., for p-values, smaller is more significant
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotation_fmt: str = ".2f" # Format for annotations in cells
):
    """
    Visualizes a heatmap of a specific metric (KL, KS p-value, etc.)
    calculated between pairs of bins for a given attribute and path.

    Args:
        metric_results: Dict mapping (bin1_name, bin2_name) tuples to metric dicts.
        metric_key: The key within the inner metric dict to plot (e.g., 'kl_symmetric').
        bin_names: Ordered list of bin names for the axes.
        attribute: Name of the attribute.
        path_name: Name of the path.
        save_path: Full path to save the heatmap image.
        title: Optional title for the plot.
        cmap: Matplotlib colormap name.
        invert_cmap: If True, use the reversed version of the colormap.
        vmin, vmax: Optional min/max values for the color scale.
        annotation_fmt: String format for cell annotations.
    """
    n_bins = len(bin_names)
    if n_bins < 2:
        LOGGER.warning(f"Skipping heatmap for {attribute} ({path_name}) - requires at least 2 bins.")
        return

    matrix = np.full((n_bins, n_bins), np.nan) # Initialize with NaNs

    for i in range(n_bins):
        for j in range(n_bins):
            if i == j:
                # Set diagonal based on metric (e.g., 0 for KL/distance, 1 for p-value)
                matrix[i, j] = 0.0 if 'kl' in metric_key or 'ks_stat' in metric_key else 1.0 if 'pval' in metric_key else np.nan
                continue

            # Try fetching result for (bin_i, bin_j) or (bin_j, bin_i)
            bin_i = bin_names[i]
            bin_j = bin_names[j]
            result_dict = metric_results.get((bin_i, bin_j), metric_results.get((bin_j, bin_i), None))

            if result_dict and metric_key in result_dict:
                value = result_dict[metric_key]
                if value is not None and np.isfinite(value):
                    matrix[i, j] = value
                # else: keep as NaN if metric is missing, None, or non-finite

    if np.isnan(matrix).all():
        LOGGER.warning(f"Skipping heatmap for {attribute} ({path_name}), metric '{metric_key}' - no valid data found.")
        return

    # Determine vmin/vmax if not provided
    valid_values = matrix[~np.isnan(matrix)]
    if vmin is None:
        vmin = np.min(valid_values) if len(valid_values) > 0 else 0
    if vmax is None:
        # Special handling for p-values often capped at 0.05 or 1.0
        if 'pval' in metric_key:
             vmax = 1.0 # Typically p-values range 0-1
             # Or maybe focus on significance: vmax = 0.1
        else:
             vmax = np.max(valid_values) if len(valid_values) > 0 else 1

    # Handle colormap inversion
    if invert_cmap:
        try:
            if not cmap.endswith("_r"):
                cmap = cmap + "_r"
        except Exception: # Catch potential issues with cmap name manipulation
             pass # Use original cmap if inversion fails

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 0.8), max(5, n_bins * 0.7)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    # Configure axes
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels(bin_names)
    ax.set_yticklabels(bin_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add annotations
    for i in range(n_bins):
        for j in range(n_bins):
            value = matrix[i, j]
            if not np.isnan(value):
                # Adjust text color based on background intensity
                bg_color_val = (value - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else 0.5
                text_color = "white" if bg_color_val < 0.5 else "black"
                # Special highlight for significant p-values?
                if 'pval' in metric_key and value < 0.05:
                     text_color = "red" # Example: highlight significant p-values
                ax.text(j, i, f"{value:{annotation_fmt}}", ha="center", va="center", color=text_color)
            else:
                 ax.text(j, i, "N/A", ha="center", va="center", color="grey")


    # Add colorbar and title
    fig.colorbar(im, ax=ax, label=metric_key.replace('_', ' ').title())
    if title is None:
        title = f"{metric_key.replace('_', ' ').title()} between Bins\nAttribute: {attribute.title()}, Path: {path_name.title()}"
    ax.set_title(title, wrap=True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        # LOGGER.info(f"Saved heatmap to {save_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save heatmap {save_path}: {e}")
    plt.close(fig)

def process_analysis_attributes(
    analysis_attributes: List[str],
    output_dir: str,
    color: bool,
    binning_config: Dict,
    input_colors_by_attr: Dict,
    translated_colors_by_attr: Dict,
    half_cycle_colors_by_attr: Dict,
    full_cycle_colors_by_attr: Dict,
    examples_by_attr: Dict,
    results: Dict,
    num_bins: int = 50,
    significance_alpha: float = 0.05,
    rgb: bool = True, 
    conditioning = False
) -> Dict:
    """
    Processes analysis attributes by comparing pairs of bins within each attribute
    for each processing path (translated, half-cycle, full-cycle).

    Calculates KL divergence and KS tests between H channel distributions of bin pairs.
    Visualizes bin-pair distribution comparisons and generates metric heatmaps.
    Generates and saves a global summary of comparison statistics.

    Args:
        analysis_attributes: List of attributes to analyze (e.g., ['shape', 'rotation']).
        output_dir: Base directory for saving results.
        color: Boolean indicating if color was used (affects directory naming).
        binning_config: Dictionary defining bins for each attribute.
        input_colors_by_attr: Dict mapping attr -> bin_name -> {'H': [values...]} (Stored for context).
        translated_colors_by_attr: Dict for translated path results.
        half_cycle_colors_by_attr: Dict for half-cycle path results.
        full_cycle_colors_by_attr: Dict for full-cycle path results.
        examples_by_attr: Dict mapping attr -> bin_name -> [example_dict].
        results: Dictionary to store the detailed analysis results.
        num_bins: Number of bins for histograms and metric calculations.
        significance_alpha: P-value threshold for counting significant comparisons.
        rgb: Whether to process RGB channels in addition to H.

    Returns:
        The updated results dictionary containing detailed comparisons.
    """
    # --- Initialize Global Summary Statistics ---
    total_comparisons_overall = 0
    total_significant_comparisons_overall = 0
    attribute_stats_summary = {}
    path_stats_summary = {}

    # Define the path data sources and their names
    path_data_sources = {
        "translated": translated_colors_by_attr,
        "half_cycle": half_cycle_colors_by_attr,
        "full_cycle": full_cycle_colors_by_attr,
    }

    # Initialize path stats summary structure
    for path_name in path_data_sources.keys():
        path_stats_summary[path_name] = {'total': 0, 'significant': 0}

    for attr in analysis_attributes:
        LOGGER.info(f"Processing attribute for bin-pair comparisons: {attr}")
        attr_base_dir = os.path.join(output_dir, f"{attr}")
        os.makedirs(attr_base_dir, exist_ok=True)

        # --- Initialize results storage and summary stats for this attribute ---
        results[attr] = results.get(attr, {})
        attribute_stats_summary[attr] = {'total': 0, 'significant': 0}
        
        # Store original distributions for context
        results[attr]['input_distributions'] = input_colors_by_attr.get(attr, {})
        for path_name, source_dict in path_data_sources.items():
            results[attr][f'{path_name}_distributions'] = source_dict.get(attr, {})
            # Initialize storage for bin-pair metrics for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = {}

        # Get bins and pairs for this attribute
        bin_names = binning_config.get(attr, {}).get('bin_names', [])
        if len(bin_names) < 2:
            LOGGER.warning(f"Attribute '{attr}' has fewer than 2 bins ({bin_names}). Skipping bin-pair comparisons.")
            continue
        bin_pairs = list(combinations(bin_names, 2))

        # --- Iterate through each PATH ---
        for path_name, path_data in path_data_sources.items():
            # Directory for this path's comparison plots and heatmaps
            path_comparison_dir = os.path.join(attr_base_dir, f"{path_name}")

            if attr not in path_data:
                LOGGER.warning(f"    No data found for attribute '{attr}' in path '{path_name}'. Skipping.")
                continue
            path_data_for_attr = path_data[attr]

            # Dictionary to store results for this specific path and attribute
            bin_comparison_results_for_path = {}

            # --- Iterate through BIN PAIRS ---
            for bin1_name, bin2_name in bin_pairs:
                total_comparisons_overall += 1
                attribute_stats_summary[attr]['total'] += 1
                path_stats_summary[path_name]['total'] += 1

                # Use our refactored _compare_distributions_between_bins function
                metrics, flag = _compare_distributions_between_bins(
                    path_name=path_name,
                    attribute=attr,
                    bin1_name=bin1_name,
                    bin2_name=bin2_name,
                    path_data_for_attribute=path_data_for_attr,
                    output_dir_for_path_comparison=path_comparison_dir,
                    num_bins=num_bins,
                    rgb=rgb,   # Pass the rgb paramete
                    conditioning=conditioning
                )

                bin_comparison_results_for_path[(bin1_name, bin2_name)] = metrics
                
                # Check significance for summary
                ks_pval = metrics.get('ks_pval')
                if ks_pval is not None and np.isfinite(ks_pval) and ks_pval < significance_alpha and flag:
                    total_significant_comparisons_overall += 1
                    attribute_stats_summary[attr]['significant'] += 1
                    path_stats_summary[path_name]['significant'] += 1

            # Store the detailed bin-pair results for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = bin_comparison_results_for_path

            # --- Generate Heatmaps for the current path ---
            if bin_comparison_results_for_path:
                os.makedirs(path_comparison_dir, exist_ok=True)
                
                # Generate heatmaps using helper function
                generate_metric_heatmaps(
                    bin_comparison_results_for_path,
                    bin_names,
                    attr,
                    path_name,
                    path_comparison_dir,
                    rgb=rgb
                )
            else:
                LOGGER.warning(f"No valid bin comparison results found for path '{path_name}', attribute '{attr}'. Skipping heatmaps.")

        # --- Visualize examples (outside path loop) ---
        visualize_attribute_examples(examples_by_attr, attr, bin_names, attr_base_dir)

    # --- Assemble and Save Global Summary ---
    global_summary = {
        "total_comparisons": total_comparisons_overall,
        "significant_comparisons": total_significant_comparisons_overall,
        "significance_alpha": significance_alpha,
        "attribute_stats": attribute_stats_summary,
        "path_stats": path_stats_summary,
    }

    save_global_summary(global_summary, output_dir)

    return results

# Helper functions to keep the main function clean and modular

def generate_metric_heatmaps(bin_comparison_results, bin_names, attribute, path_name, output_dir, rgb=False):
    """Generate heatmaps for various metrics of bin comparisons."""
    if 'visualize_metric_heatmap_bin_pairs' not in globals():
        LOGGER.warning("`visualize_metric_heatmap_bin_pairs` function not found. Skipping heatmap generation.")
        return
    
    try:
        # Symmetric KL Heatmap
        visualize_metric_heatmap_bin_pairs(
            metric_results=bin_comparison_results,
            metric_key='kl_symmetric',
            bin_names=bin_names,
            attribute=attribute,
            path_name=path_name,
            save_path=os.path.join(output_dir, f'{path_name}_{attribute}_heatmap_KL_symmetric.png'),
            vmin=0  # KL >= 0
        )
        
        # KS p-value Heatmap
        visualize_metric_heatmap_bin_pairs(
            metric_results=bin_comparison_results,
            metric_key='ks_pval',
            bin_names=bin_names,
            attribute=attribute,
            path_name=path_name,
            save_path=os.path.join(output_dir, f'{path_name}_{attribute}_heatmap_KS_pvalue.png'),
            cmap='viridis_r',  # Invert cmap for p-values (low = significant)
            vmin=0, vmax=1.0,
            annotation_fmt=".3f"  # More precision for p-values
        )
        
        # Generate RGB heatmaps if RGB mode is enabled
        if rgb:
            for channel in ['R', 'G', 'B']:
                visualize_metric_heatmap_bin_pairs(
                    metric_results=bin_comparison_results,
                    metric_key=f'kl_symmetric_{channel.lower()}',
                    bin_names=bin_names,
                    attribute=attribute,
                    path_name=path_name,
                    save_path=os.path.join(output_dir, f'{path_name}_{attribute}_heatmap_KL_symmetric_{channel}.png'),
                    vmin=0,
                    title=f"Symmetric KL Divergence - {channel} Channel"
                )
                
                visualize_metric_heatmap_bin_pairs(
                    metric_results=bin_comparison_results,
                    metric_key=f'ks_stat_{channel.lower()}',
                    bin_names=bin_names,
                    attribute=attribute,
                    path_name=path_name,
                    save_path=os.path.join(output_dir, f'{path_name}_{attribute}_heatmap_KS_stat_{channel}.png'),
                    vmin=0, vmax=1.0,
                    title=f"KS Statistic - {channel} Channel"
                )
                
    except Exception as e:
        LOGGER.error(f"Failed to generate heatmaps for {attribute}, path {path_name}: {e}", exc_info=True)

def visualize_attribute_examples(examples_by_attr, attr, bin_names, output_dir):
    """Visualize examples for a given attribute if available."""
    if 'visualize_examples_by_attribute' not in globals():
        LOGGER.warning("`visualize_examples_by_attribute` not found. Skipping examples.")
        return
        
    if attr in examples_by_attr and any(examples_by_attr[attr].values()):
        try:
            visualize_examples_by_attribute(
                examples_by_attr[attr],
                attr,
                bin_names,
                os.path.join(output_dir, f'examples_by_{attr}.png')
            )
        except Exception as e:
            LOGGER.error(f"Failed to visualize examples for {attr}: {e}", exc_info=True)
    else:
        LOGGER.warning(f"Skipping examples visualization for {attr} due to missing data.")

def save_global_summary(global_summary, output_dir):
    """Save the global summary statistics to a JSON file."""
    summary_file_path = os.path.join(output_dir, "within_model_comparison_summary.json")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(global_summary, f, indent=2)
        LOGGER.info(f"Within-model comparison summary saved to: {summary_file_path}")
    except TypeError as e:
        LOGGER.error(f"Failed to serialize within-model summary to JSON: {e}. Summary was: {global_summary}")
    except Exception as e:
        LOGGER.error(f"Failed to save within-model comparison summary: {e}", exc_info=True)


import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations
from typing import List, Dict
import concurrent.futures
from functools import partial
import logging

# Set up logger
LOGGER = logging.getLogger(__name__)

def compute_kl_divergence(P, Q, eps=1e-10):
    """
    Calculate KL divergence using NumPy for compatibility.
    
    Args:
        P: First probability distribution array
        Q: Second probability distribution array
        eps: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    # Convert tensors to numpy if needed
    if isinstance(P, torch.Tensor):
        P = P.cpu().numpy()
    if isinstance(Q, torch.Tensor):
        Q = Q.cpu().numpy()
        
    # Make sure both distributions sum to 1
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Add small epsilon to avoid log(0)
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)
    
    # Calculate KL divergence: sum(P * log(P / Q))
    kl_div = np.sum(P * np.log(P / Q))
    
    return kl_div

def _compare_distributions_between_bins(
    path_name,
    attribute,
    bin1_name,
    bin2_name,
    path_data_for_attribute,
    output_dir_for_path_comparison,
    num_bins=50,
    rgb=True, 
    conditioning=False
):
    """
    Compares distributions between two specified bins for a single attribute.
    
    Args:
        path_name: Name of the processing path
        attribute: Name of the attribute being analyzed
        bin1_name: Name of the first bin for comparison
        bin2_name: Name of the second bin for comparison
        path_data_for_attribute: Dictionary containing binned data for the attribute
        output_dir_for_path_comparison: Directory to save visualizations
        num_bins: Number of bins for histogram
        rgb: Whether to process RGB channels in addition to H
        
    Returns:
        Dictionary with comparison metrics or None if data is insufficient
    """
    flag = True
    # 1. Extract data for the two bins
    dist1_data = path_data_for_attribute.get(bin1_name, {})
    dist2_data = path_data_for_attribute.get(bin2_name, {})
    
    # Define which channels to process
    channels = ["H", "R", "G", "B"] if rgb else ["H"]
    values1_dict = extract_valid_values(dist1_data, channels)
    values2_dict = extract_valid_values(dist2_data, channels)
    
    # Check if we have enough data in the H channel
    if len(values1_dict.get("H", [])) <= 1 or len(values2_dict.get("H", [])) <= 1:
        LOGGER.warning(f"Skipping comparison between bins '{bin1_name}' and '{bin2_name}' for "
                      f"attribute '{attribute}', path '{path_name}' - No valid H data "
                      f"({bin1_name}: {len(values1_dict.get('H', []))}, {bin2_name}: {len(values2_dict.get('H', []))})")
        return None
    
    comparison_label = f"{attribute}_{bin1_name}_vs_{bin2_name}"
    LOGGER.debug(f"  Comparing bins for path '{path_name}', {comparison_label}")
    
    # 2. Visualize comparison
    os.makedirs(output_dir_for_path_comparison, exist_ok=True)
    safe_bin1 = bin1_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    safe_bin2 = bin2_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    plot_filename = os.path.join(
        output_dir_for_path_comparison,
        f'{attribute}_{safe_bin1}_vs_{safe_bin2}'
    )
    
    title = f"Distribution Comparison ({path_name.title()} Path)\n{attribute.title()}: {bin1_name} vs {bin2_name}"
    
    
    # 3. Calculate metrics
    results = {}
    
    # For H channel, use the existing comparison_metrics function that returns multiple metrics
    values1_h = values1_dict.get("H", np.array([]))
    values2_h = values2_dict.get("H", np.array([]))
    kl_h_12, kl_h_21, kl_h_sym, ks_stat, ks_pval = comparison_metrics(values1_h, values2_h, num_bins)
    
    results = {
        'kl_1_vs_2': kl_h_12,
        'kl_2_vs_1': kl_h_21,
        'kl_symmetric': kl_h_sym,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval
    }
    
    # If RGB is enabled, calculate metrics for R, G, B channels
    if rgb:
        flag = not conditioning
        for channel in ["R", "G", "B"]:
            values1 = values1_dict.get(channel, np.array([]))
            values2 = values2_dict.get(channel, np.array([]))
            
            if len(values1) <= 1 or len(values2) <= 1:
                continue
                
            # Use the same comparison_metrics function
            kl_12, kl_21, kl_sym, channel_ks_stat, channel_ks_pval = comparison_metrics(values1, values2, num_bins)
            
            # Add to results with channel-specific keys
            results[f'kl_1_vs_2_{channel.lower()}'] = kl_12
            results[f'kl_2_vs_1_{channel.lower()}'] = kl_21
            results[f'kl_symmetric_{channel.lower()}'] = kl_sym
            results[f'ks_stat_{channel.lower()}'] = channel_ks_stat
            results[f'ks_pval_{channel.lower()}'] = channel_ks_pval
            
            # Log RGB channel results if they're significant
            if channel_ks_pval < 0.05:
                flag = True
                LOGGER.info(f"    Path '{path_name}', {attribute} ({channel}): '{bin1_name}' vs '{bin2_name}' -> "
                           f"KL({bin1_name}||{bin2_name}): {kl_12:.4f}, "
                           f"KL({bin2_name}||{bin1_name}): {kl_21:.4f}, "
                           f"Sym KL: {kl_sym:.4f}, "
                           f"KS Stat ({num_bins} bins): {channel_ks_stat:.4f}, KS p-val: {channel_ks_pval:.4f}")
    
    # Log H channel results if they're significant
    if ks_pval < 0.05 and flag:
        LOGGER.info(f"    Path '{path_name}', {attribute} (H): '{bin1_name}' vs '{bin2_name}' -> "
                   f"KL({bin1_name}||{bin2_name}): {kl_h_12:.4f}, "
                   f"KL({bin2_name}||{bin1_name}): {kl_h_21:.4f}, "
                   f"Sym KL: {kl_h_sym:.4f}, "
                   f"KS Stat ({num_bins} bins): {ks_stat:.4f}, KS p-val: {ks_pval:.4f}")
        visualize_distributions(
            dist1=dist1_data,
            dist2=dist2_data,
            label1=f"{path_name.title()} ({bin1_name})",
            label2=f"{path_name.title()} ({bin2_name})",
            bin_name=f"{bin1_name} vs {bin2_name}",
            attr=attribute,
            save_path=plot_filename,
            channels=channels,
            num_bins=num_bins,
            title=title
        )
    
    if not np.isfinite(kl_h_sym):
        LOGGER.warning(f"Symmetric KL divergence calculation resulted in non-finite value for "
                      f"{path_name}, {comparison_label}.")
    
    return results, flag

def _compare_distributions_between_bins_cuda(
    path_name: str,
    attribute: str,
    bin1_name: str,
    bin2_name: str,
    path_data_for_attribute: Dict,
    output_dir_for_path_comparison: str,
    num_bins: int = 50,
    rgb: bool = True,
    device: torch.device = None, 
    conditioning = False
) -> Dict:
    """
    Compare distributions between two bins using GPU-accelerated operations when available.
    
    Args:
        path_name: Name of the processing path
        attribute: Name of the attribute being compared
        bin1_name, bin2_name: Names of the two bins to compare
        path_data_for_attribute: Dict containing data for this attribute
        output_dir_for_path_comparison: Directory for saving comparison plots
        num_bins: Number of bins for histograms
        rgb: Whether to process RGB channels in addition to H
        device: torch.device to use for computations
        
    Returns:
        Dictionary of comparison metrics
    """
    
    metrics = {}
    
    # 1. Extract data for the two bins
    dist1_data = path_data_for_attribute.get(bin1_name, {})
    dist2_data = path_data_for_attribute.get(bin2_name, {})
    
    # Define which channels to process
    channels = ["H", "R", "G", "B"] if rgb else ["H"]
    values1_dict = extract_valid_values(dist1_data, channels)
    values2_dict = extract_valid_values(dist2_data, channels)
    
    # Check if we have enough data in the H channel
    if len(values1_dict.get("H", [])) <= 1 or len(values2_dict.get("H", [])) <= 1:
        LOGGER.warning(f"Skipping comparison between bins '{bin1_name}' and '{bin2_name}' for "
                      f"attribute '{attribute}', path '{path_name}' - No valid H data "
                      f"({bin1_name}: {len(values1_dict.get('H', []))}, {bin2_name}: {len(values2_dict.get('H', []))})")
        return None
    
    comparison_label = f"{attribute}_{bin1_name}_vs_{bin2_name}"
    LOGGER.debug(f"  Comparing bins for path '{path_name}', {comparison_label}")
    
    # 2. Visualize comparison
    
    safe_bin1 = bin1_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    safe_bin2 = bin2_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    plot_filename = os.path.join(
        output_dir_for_path_comparison,
        f'{attribute}_{safe_bin1}_vs_{safe_bin2}'
    )
    
    title = f"Distribution Comparison ({path_name.title()} Path)\n{attribute.title()}: {bin1_name} vs {bin2_name}"
    
    
    # 3. Calculate metrics
    results = {}
    
    # For H channel, use the existing comparison_metrics function that returns multiple metrics
    h_values1 = values1_dict.get("H", np.array([]))
    h_values2 = values2_dict.get("H", np.array([]))
    
    # Move data to device if possible
    if device is not None and device.type == 'cuda':
        try:
            # Convert to tensors if they're not already
            if not isinstance(h_values1, torch.Tensor):
                h_values1 = torch.tensor(h_values1, device=device)
            else:
                h_values1 = h_values1.to(device)
                
            if not isinstance(h_values2, torch.Tensor):
                h_values2 = torch.tensor(h_values2, device=device)
            else:
                h_values2 = h_values2.to(device)
        except Exception as e:
            LOGGER.warning(f"Error moving data to GPU: {str(e)}. Falling back to CPU.")
            # Ensure data is numpy arrays for CPU processing
            if isinstance(h_values1, torch.Tensor):
                h_values1 = h_values1.cpu().numpy()
            if isinstance(h_values2, torch.Tensor):
                h_values2 = h_values2.cpu().numpy()
    
    # Convert tensors to numpy if needed for calculations
    h_values1_np = h_values1.cpu().numpy() if isinstance(h_values1, torch.Tensor) else h_values1
    h_values2_np = h_values2.cpu().numpy() if isinstance(h_values2, torch.Tensor) else h_values2
    
    # Determine the range for histograms
    min_val = min(np.min(h_values1_np) if len(h_values1_np) > 0 else 0, 
                 np.min(h_values2_np) if len(h_values2_np) > 0 else 0)
    max_val = max(np.max(h_values1_np) if len(h_values1_np) > 0 else 255, 
                 np.max(h_values2_np) if len(h_values2_np) > 0 else 255)
    
    # Create histograms and normalize to get probability distributions
    hist1, bin_edges = np.histogram(h_values1_np, bins=num_bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(h_values2_np, bins=num_bins, range=(min_val, max_val), density=True)
    
    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate KL divergence
    kl_div_1_to_2 = compute_kl_divergence(hist1, hist2)
    kl_div_2_to_1 = compute_kl_divergence(hist2, hist1)
    
    # Symmetric KL (Average)
    kl_div_sym = (kl_div_1_to_2 + kl_div_2_to_1) / 2.0 if np.isfinite(kl_div_1_to_2) and np.isfinite(kl_div_2_to_1) else np.inf
    
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
        from scipy import stats
        ks_stat, ks_pval = stats.ks_2samp(binned_dist1_norm, binned_dist2_norm)
    else:
        LOGGER.warning(f"Cannot normalize distributions - all values are the same: {min_val}")
        ks_stat = 0.0 if np.array_equal(h_values1_np, h_values2_np) else 1.0
        # ks_pval = 1.0 if np.array_equal(h_values1_np, h_values2_np) else 0.0
    
    # Store metrics
    metrics = {
        'kl_div_1_to_2': kl_div_1_to_2,
        'kl_div_2_to_1': kl_div_2_to_1,
        'kl_div_sym': kl_div_sym,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
    }
    flag1 = ks_pval < 0.05
    flag2 = False
    # Process RGB channels if requested
    if rgb:
        for channel in ['R', 'G', 'B']:
            if (channel in path_data_for_attribute[bin1_name] and 
                channel in path_data_for_attribute[bin2_name]):
                
                values1 = path_data_for_attribute[bin1_name][channel]
                values2 = path_data_for_attribute[bin2_name][channel]
                
                if len(values1) >= 5 and len(values2) >= 5:
                    # Move data to device if possible
                    if device is not None and device.type == 'cuda':
                        try:
                            if not isinstance(values1, torch.Tensor):
                                values1 = torch.tensor(values1, device=device)
                            else:
                                values1 = values1.to(device)
                                
                            if not isinstance(values2, torch.Tensor):
                                values2 = torch.tensor(values2, device=device)
                            else:
                                values2 = values2.to(device)
                        except Exception as e:
                            LOGGER.warning(f"Error moving {channel} data to GPU: {str(e)}. Falling back to CPU.")
                            # Ensure data is numpy arrays for CPU processing
                            if isinstance(values1, torch.Tensor):
                                values1 = values1.cpu().numpy()
                            if isinstance(values2, torch.Tensor):
                                values2 = values2.cpu().numpy()
                    
                    # Convert tensors to numpy if needed for calculations
                    values1_np = values1.cpu().numpy() if isinstance(values1, torch.Tensor) else values1  
                    values2_np = values2.cpu().numpy() if isinstance(values2, torch.Tensor) else values2
                    
                    # Determine the range for histograms
                    ch_min_val = min(np.min(values1_np) if len(values1_np) > 0 else 0,
                                    np.min(values2_np) if len(values2_np) > 0 else 0)
                    ch_max_val = max(np.max(values1_np) if len(values1_np) > 0 else 255,
                                    np.max(values2_np) if len(values2_np) > 0 else 255)
                    
                    # Create histograms and normalize
                    ch_hist1, ch_bin_edges = np.histogram(values1_np, bins=num_bins, 
                                                        range=(ch_min_val, ch_max_val), density=True)
                    ch_hist2, _ = np.histogram(values2_np, bins=num_bins, 
                                             range=(ch_min_val, ch_max_val), density=True)
                    
                    # Get bin centers
                    ch_bin_centers = (ch_bin_edges[:-1] + ch_bin_edges[1:]) / 2
                    
                    # Calculate KL divergence
                    kl_1_to_2 = compute_kl_divergence(ch_hist1, ch_hist2)
                    kl_2_to_1 = compute_kl_divergence(ch_hist2, ch_hist1)
                    
                    # Symmetric KL
                    kl_sym = (kl_1_to_2 + kl_2_to_1) / 2.0 if np.isfinite(kl_1_to_2) and np.isfinite(kl_2_to_1) else np.inf
                    
                    # Create binned distributions for KS test
                    ch_binned_dist1 = np.repeat(ch_bin_centers, np.round(ch_hist1 * 1000).astype(int))
                    ch_binned_dist2 = np.repeat(ch_bin_centers, np.round(ch_hist2 * 1000).astype(int))
                    
                    # Normalize for KS test
                    ch_range_size = ch_max_val - ch_min_val
                    if ch_range_size > 0:
                        ch_binned_dist1_norm = (ch_binned_dist1 - ch_min_val) / ch_range_size  
                        ch_binned_dist2_norm = (ch_binned_dist2 - ch_min_val) / ch_range_size
                        
                        # KS test
                        from scipy import stats
                        ch_ks_stat, ch_ks_pval = stats.ks_2samp(ch_binned_dist1_norm, ch_binned_dist2_norm)
                        if ch_ks_pval < 0.05:
                            flag2 = True
                            if flag1 : 
                                LOGGER.info(f"    Path '{path_name}', {attribute} ({channel}): '{bin1_name}' vs '{bin2_name}' -> "
                                        f"KL({bin1_name}||{bin2_name}): {kl_1_to_2:.4f}, "
                                        f"KL({bin2_name}||{bin1_name}): {kl_2_to_1:.4f}, "
                                        f"Sym KL: {kl_sym:.4f}, "
                                        f"KS Stat ({num_bins} bins): {ch_ks_stat:.4f}, KS p-val: {ch_ks_pval:.4f}")
                    else:
                        LOGGER.warning(f"Cannot normalize {channel} distributions - all values are the same: {ch_min_val}")
                        # ch_ks_stat = 0.0 if np.array_equal(values1_np, values2_np) else 1.0
                        # ch_ks_pval = 1.0 if np.array_equal(values1_np, values2_np) else 0.0
                    
                    # Store metrics for this channel
                    metrics[f'kl_div_1_to_2_{channel}'] = kl_1_to_2
                    metrics[f'kl_div_2_to_1_{channel}'] = kl_2_to_1  
                    metrics[f'kl_div_sym_{channel}'] = kl_sym
                    metrics[f'ks_stat_{channel}'] = ch_ks_stat
                    metrics[f'ks_pval_{channel}'] = ch_ks_pval
    
    # Create visualization directory if needed
    if flag1 and flag2:
        os.makedirs(output_dir_for_path_comparison, exist_ok=True)
        visualize_distributions(
            dist1=dist1_data,
            dist2=dist2_data,
            label1=f"{path_name.title()} ({bin1_name})",
            label2=f"{path_name.title()} ({bin2_name})",
            bin_name=f"{bin1_name} vs {bin2_name}",
            attr=attribute,
            save_path=plot_filename,
            channels=channels,
            num_bins=num_bins,
            title=title
        )
    
    return metrics

def generate_metric_heatmaps_batch(
    bin_comparison_results: Dict,
    bin_names: List[str],
    attribute: str,
    path_name: str,
    output_dir: str,
    rgb: bool = True,
    batch_size: int = 8
):
    """
    Generate heatmaps for all metrics using matplotlib with batched processing.
    
    Args:
        bin_comparison_results: Dictionary of bin comparison metrics
        bin_names: List of bin names for labeling
        attribute: Attribute name for titling
        path_name: Processing path name for titling
        output_dir: Directory to save heatmaps
        rgb: Whether to include RGB metrics
        batch_size: Number of heatmaps to process in a batch
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    from matplotlib.figure import Figure
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to visualize
    metrics = ['kl_div_sym', 'ks_stat', 'ks_pval']
    
    # Add RGB metrics if needed
    if rgb:
        for channel in ['R', 'G', 'B']:
            metrics.extend([f'kl_div_sym_{channel}', 
                           f'ks_stat_{channel}', f'ks_pval_{channel}'])
    
    # Filter metrics to those that actually exist in the results
    available_metrics = []
    for metric_name in metrics:
        if any(metric_name in result for result in bin_comparison_results.values()):
            available_metrics.append(metric_name)
    
    # Process metrics in batches
    for i in range(0, len(available_metrics), batch_size):
        batch_metrics = available_metrics[i:i+batch_size]
        
        # Process each metric in this batch
        for metric_name in batch_metrics:
            n_bins = len(bin_names)
            matrix = np.zeros((n_bins, n_bins))
            matrix.fill(np.nan)  # Fill with NaN for bins not compared
            
            # Populate matrix
            for (bin1, bin2), metrics_dict in bin_comparison_results.items():
                if metric_name in metrics_dict:
                    i = bin_names.index(bin1)
                    j = bin_names.index(bin2)
                    matrix[i, j] = metrics_dict[metric_name]
                    # For symmetric metrics, fill the other half
                    if 'kl_div' not in metric_name:  # KL div is asymmetric
                        matrix[j, i] = metrics_dict[metric_name]
            
            # Create heatmap
            fig = plt.figure(figsize=(10, 8))
            mask = np.isnan(matrix)
            
            # Choose colormap based on metric
            if 'pval' in metric_name:
                cmap = 'viridis'
                vmin, vmax = 0, 0.05  # p-values typically significant below 0.05
            else:
                cmap = 'magma'
                vmin, vmax = None, None
            
            sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                       annot=True, fmt=".3f", xticklabels=bin_names, yticklabels=bin_names)
            
            plt.title(f"{attribute} - {path_name} - {metric_name}")
            plt.tight_layout()
            
            # Save figure
            filename = f"{attribute}_{path_name}_{metric_name}_heatmap.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=100)
            plt.close(fig)
        
        # Clear memory between batches
        plt.close('all')
    
    return

def process_analysis_attributes_parallel(
    analysis_attributes: List[str],
    output_dir: str,
    color: bool,
    binning_config: Dict,
    input_colors_by_attr: Dict,
    translated_colors_by_attr: Dict,
    half_cycle_colors_by_attr: Dict,
    full_cycle_colors_by_attr: Dict,
    examples_by_attr: Dict,
    results: Dict,
    num_bins: int = 50,
    significance_alpha: float = 0.05,
    rgb: bool = True,
    max_workers: int = None,
    batch_size: int = 32, 
    conditioning : bool = False
) -> Dict:
    """
    Parallelized version of process_analysis_attributes using CUDA for computation.
    
    Args:
        analysis_attributes: List of attributes to analyze (e.g., ['shape', 'rotation']).
        output_dir: Base directory for saving results.
        color: Boolean indicating if color was used (affects directory naming).
        binning_config: Dictionary defining bins for each attribute.
        input_colors_by_attr: Dict mapping attr -> bin_name -> {'H': [values...]} (Stored for context).
        translated_colors_by_attr: Dict for translated path results.
        half_cycle_colors_by_attr: Dict for half-cycle path results.
        full_cycle_colors_by_attr: Dict for full-cycle path results.
        examples_by_attr: Dict mapping attr -> bin_name -> [example_dict].
        results: Dictionary to store the detailed analysis results.
        num_bins: Number of bins for histograms and metric calculations.
        significance_alpha: P-value threshold for counting significant comparisons.
        rgb: Whether to process RGB channels in addition to H.
        max_workers: Maximum number of parallel workers (defaults to CPU count)
        batch_size: Number of comparisons to batch together for GPU processing

    Returns:
        The updated results dictionary containing detailed comparisons.
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")
    
    # If no GPU is available, fall back to CPU parallelism
    if device.type == "cpu":
        LOGGER.warning("CUDA not available, falling back to CPU parallelism")
    
    # --- Initialize Global Summary Statistics ---
    total_comparisons_overall = 0
    total_significant_comparisons_overall = 0
    attribute_stats_summary = {}
    path_stats_summary = {}

    # Define the path data sources and their names
    path_data_sources = {
        "translated": translated_colors_by_attr,
        "half_cycle": half_cycle_colors_by_attr,
        "full_cycle": full_cycle_colors_by_attr,
    }

    # Initialize path stats summary structure
    for path_name in path_data_sources.keys():
        path_stats_summary[path_name] = {'total': 0, 'significant': 0}

    # Process each attribute (could be parallelized if needed)
    for attr in analysis_attributes:
        LOGGER.info(f"Processing attribute for bin-pair comparisons: {attr}")
        attr_base_dir = os.path.join(output_dir, f"{attr}")
        os.makedirs(attr_base_dir, exist_ok=True)

        # --- Initialize results storage and summary stats for this attribute ---
        results[attr] = results.get(attr, {})
        attribute_stats_summary[attr] = {'total': 0, 'significant': 0}
        
        # Store original distributions for context
        results[attr]['input_distributions'] = input_colors_by_attr.get(attr, {})
        for path_name, source_dict in path_data_sources.items():
            results[attr][f'{path_name}_distributions'] = source_dict.get(attr, {})
            # Initialize storage for bin-pair metrics for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = {}

        # Get bins and pairs for this attribute
        bin_names = binning_config.get(attr, {}).get('bin_names', [])
        if len(bin_names) < 2:
            LOGGER.warning(f"Attribute '{attr}' has fewer than 2 bins ({bin_names}). Skipping bin-pair comparisons.")
            continue
        bin_pairs = list(combinations(bin_names, 2))

        # --- Iterate through each PATH ---
        for path_name, path_data in path_data_sources.items():
            # Directory for this path's comparison plots and heatmaps
            path_comparison_dir = os.path.join(attr_base_dir, f"{path_name}")

            if attr not in path_data:
                LOGGER.warning(f"    No data found for attribute '{attr}' in path '{path_name}'. Skipping.")
                continue
            path_data_for_attr = path_data[attr]

            # Dictionary to store results for this specific path and attribute
            bin_comparison_results_for_path = {}
            
            # --- Parallelize BIN PAIR comparisons ---
            # Create a partial function with fixed parameters
            comparison_func = partial(
                _compare_distributions_between_bins_cuda,
                path_name=path_name,
                attribute=attr,
                path_data_for_attribute=path_data_for_attr,
                output_dir_for_path_comparison=path_comparison_dir,
                num_bins=num_bins,
                rgb=rgb,
                device=device
            )
            
            # Process bin pairs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks and collect futures
                futures = {
                    executor.submit(comparison_func, bin1_name=bin1, bin2_name=bin2): (bin1, bin2)
                    for bin1, bin2 in bin_pairs
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    bin1, bin2 = futures[future]
                    try:
                        metrics = future.result()
                        bin_comparison_results_for_path[(bin1, bin2)] = metrics
                        
                        # Update summary statistics
                        total_comparisons_overall += 1
                        attribute_stats_summary[attr]['total'] += 1
                        path_stats_summary[path_name]['total'] += 1
                        
                        # Check significance for summary
                        ks_pval = metrics.get('ks_pval')
                        
                        
                        rgb_significant = True
                        
                        
                        if rgb:
                            if conditioning:
                                rgb_significant = False

                                for channel in ['R', 'G', 'B']:
                                    channel_pval = metrics.get(f'ks_pval_{channel}')
                                    if channel_pval is not None and np.isfinite(channel_pval) and channel_pval < significance_alpha:
                                        rgb_significant = True
                                        print(f"Significant for {channel} channel")
                                        break

                        if ks_pval is not None and np.isfinite(ks_pval) and ks_pval < significance_alpha and rgb_significant:
                            total_significant_comparisons_overall += 1
                            attribute_stats_summary[attr]['significant'] += 1
                            path_stats_summary[path_name]['significant'] += 1
                    
                    except Exception as e:
                        LOGGER.error(f"Error processing comparison for bins {bin1} and {bin2}: {str(e)}")

            # Store the detailed bin-pair results for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = bin_comparison_results_for_path

            # --- Generate Heatmaps for the current path ---
            if bin_comparison_results_for_path:
                os.makedirs(path_comparison_dir, exist_ok=True)
                
                # Generate heatmaps using helper function
                generate_metric_heatmaps_batch(
                    bin_comparison_results_for_path,
                    bin_names,
                    attr,
                    path_name,
                    path_comparison_dir,
                    rgb=rgb,
                    batch_size=batch_size
                )
            else:
                LOGGER.warning(f"No valid bin comparison results found for path '{path_name}', attribute '{attr}'. Skipping heatmaps.")

        # --- Visualize examples (outside path loop) ---
        visualize_attribute_examples(examples_by_attr, attr, bin_names, attr_base_dir)

    # --- Assemble and Save Global Summary ---
    global_summary = {
        "total_comparisons": total_comparisons_overall,
        "significant_comparisons": total_significant_comparisons_overall,
        "significance_alpha": significance_alpha,
        "attribute_stats": attribute_stats_summary,
        "path_stats": path_stats_summary,
    }

    save_global_summary_parrallel(global_summary, output_dir)

    return results

def save_global_summary_parrallel(global_summary, output_dir):
    """
    Save global summary of comparison statistics.
    
    Args:
        global_summary: Dictionary with global statistics
        output_dir: Directory to save summary
    """
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, "within_model_comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(global_summary, f, indent=4)
    
    # Create visualizations of the summary
    
    # 1. Bar chart of significant comparisons by attribute
    attr_stats = global_summary.get('attribute_stats', {})
    if attr_stats:
        attrs = list(attr_stats.keys())
        sig_counts = [attr_stats[a].get('significant', 0) for a in attrs]
        total_counts = [attr_stats[a].get('total', 0) for a in attrs]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(attrs))
        width = 0.35
        
        plt.bar(x - width/2, sig_counts, width, label='Significant')
        plt.bar(x + width/2, total_counts, width, label='Total')
        
        plt.xlabel('Attributes')
        plt.ylabel('Count')
        plt.title('Significant vs Total Comparisons by Attribute')
        plt.xticks(x, attrs, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "attribute_significance_summary.png"), dpi=100)
        plt.close()
    
    # 2. Bar chart of significant comparisons by path
    path_stats = global_summary.get('path_stats', {})
    if path_stats:
        paths = list(path_stats.keys())
        sig_counts = [path_stats[p].get('significant', 0) for p in paths]
        total_counts = [path_stats[p].get('total', 0) for p in paths]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(paths))
        width = 0.35
        
        plt.bar(x - width/2, sig_counts, width, label='Significant')
        plt.bar(x + width/2, total_counts, width, label='Total')
        
        plt.xlabel('Processing Paths')
        plt.ylabel('Count')
        plt.title('Significant vs Total Comparisons by Processing Path')
        plt.xticks(x, paths)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "path_significance_summary.png"), dpi=100)
        plt.close()
    
    # 3. Create a pie chart of overall significance
    sig_total = global_summary.get('significant_comparisons', 0)
    not_sig_total = global_summary.get('total_comparisons', 0) - sig_total
    
    plt.figure(figsize=(8, 8))
    plt.pie([sig_total, not_sig_total], 
            labels=['Significant', 'Not Significant'],
            autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff'])
    plt.title(f'Overall Significance (α={global_summary.get("significance_alpha", 0.05)})')
    plt.savefig(os.path.join(output_dir, "overall_significance_pie.png"), dpi=100)
    plt.close()
    
    # Create a text summary
    summary_txt_path = os.path.join(output_dir, "global_comparison_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("DISTRIBUTION COMPARISON SUMMARY\n")
        f.write("==============================\n\n")
        
        f.write(f"Total comparisons: {global_summary.get('total_comparisons', 0)}\n")
        f.write(f"Significant comparisons: {global_summary.get('significant_comparisons', 0)}\n")
        f.write(f"Significance threshold: {global_summary.get('significance_alpha', 0.05)}\n\n")
        
        f.write("ATTRIBUTE SUMMARY\n")
        f.write("----------------\n")
        for attr, stats in attr_stats.items():
            sig_pct = stats.get('significant', 0) / stats.get('total', 1) * 100 if stats.get('total', 0) > 0 else 0
            f.write(f"{attr}: {stats.get('significant', 0)}/{stats.get('total', 0)} significant ({sig_pct:.1f}%)\n")
        
        f.write("\nPROCESSING PATH SUMMARY\n")
        f.write("----------------------\n")
        for path, stats in path_stats.items():
            sig_pct = stats.get('significant', 0) / stats.get('total', 1) * 100 if stats.get('total', 0) > 0 else 0
            f.write(f"{path}: {stats.get('significant', 0)}/{stats.get('total', 0)} significant ({sig_pct:.1f}%)\n")
    
    LOGGER.info(f"Global summary saved to {summary_path} and {summary_txt_path}")
    
    return

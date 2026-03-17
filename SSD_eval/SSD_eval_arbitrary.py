# --- Standard Imports ---
import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging # Use standard logging
import itertools
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional, Any, Set

# --- Imports from your project ---
# (Make sure these paths are correct relative to where you run the script)
try:
    from shimmer import DomainModule
    from shimmer.modules.global_workspace import GlobalWorkspace2Domains
    from shimmer_ssd import DEBUG_MODE, LOGGER as SSD_LOGGER, PROJECT_DIR # Use your project's logger if needed
    from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
    from shimmer_ssd.modules.domains import load_pretrained_domains
    # Add other necessary shimmer/shimmer_ssd imports if required by load_global_workspace

    from simple_shapes_dataset import SimpleShapesDataModule # Assuming SimpleShapesDataModule isn't strictly needed for eval flow
    # from simple_shapes_dataset.cli import generate_image, get_transformed_coordinates # For generate command

    # Evaluation/Analysis specific imports
    from SSD_utils import (
        generate_fixed_colors, kl_divergence, # Ensure kl_divergence is here
        # Import other utils used by HueShapeAnalyzer if needed
    )
    from SSD_H_evaluation_functions import (
         HueShapeAnalyzer,
         comparison_metrics,
         # Add visualize_distribution_comparison if needed by helpers
         # visualize_distribution_comparison
    )

except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the correct directory or paths are configured.")
    exit(1)
from SSD_utils import binning_config_6144

# --- Setup Logging ---
# Configure logging for this script
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Optionally silence DEBUG logs from specific libraries if too verbose
# logging.getLogger('shimmer_ssd').setLevel(logging.INFO)


# Set debug mode globally for this script run
DEBUG_MODE = False # Set to True for quicker runs with potentially dummy processing

# ===========================================
#  HELPER FUNCTIONS (Model Loading, etc.)
# ===========================================

from SSD_eval_regularity import load_global_workspace

def find_significant_bin_comparisons(
    analysis_results: Dict[str, Any],
    path_name: str = "translated",
    alpha: float = 0.05 # Significance level
) -> Dict[str, Set[str]]:
    """
    Identifies attributes where at least one pair of bins showed a significant
    difference in H-channel distribution based on KS test p-value.

    Args:
        analysis_results: The results dictionary from a single run's analysis.
        path_name: The processing path to check (e.g., 'translated').
        alpha: The significance threshold (e.g., 0.05).

    Returns:
        A dictionary mapping attribute names (that had significant comparisons)
        to a set containing *all* bin names associated with that attribute.
        Example: {'shape': {'diamond', 'egg', 'triangle'}, 'rotation': {'0-90', ...}}
    """
    significant_attributes_found = {}
    LOGGER.info(f"Filtering attributes for significant cross-bin comparisons (p < {alpha}) in path '{path_name}'...")

    if not analysis_results:
        LOGGER.warning("Analysis results dictionary is empty or None. Cannot filter.")
        return significant_attributes_found

    for attribute, attr_data in analysis_results.items():
        if not isinstance(attr_data, dict): # Skip non-dict entries at top level
             LOGGER.debug(f"Skipping non-dictionary entry '{attribute}' in analysis results.")
             continue

        metrics_key = f'bin_comparison_metrics_{path_name}'
        if metrics_key not in attr_data:
            LOGGER.debug(f"No bin comparison metrics found for attribute '{attribute}', path '{path_name}'.")
            continue

        bin_comparison_metrics = attr_data.get(metrics_key, {})
        if not bin_comparison_metrics:
             LOGGER.debug(f"Bin comparison metrics for '{attribute}' path '{path_name}' is empty.")
             continue


        is_significant_attr = False
        all_bins_for_this_attr = set()

        for (bin1, bin2), metrics_dict in bin_comparison_metrics.items():
             # Collect all unique bin names encountered for this attribute
             all_bins_for_this_attr.add(bin1)
             all_bins_for_this_attr.add(bin2)

             if isinstance(metrics_dict, dict):
                 ks_pval = metrics_dict.get('ks_pval')
                 # Check if p-value is valid and below threshold
                 if ks_pval is not None and isinstance(ks_pval, (float, np.floating)) and np.isfinite(ks_pval) and ks_pval < alpha:
                     LOGGER.info(f"  Found significant difference (p={ks_pval:.4f}) between bins '{bin1}' and '{bin2}' for attribute '{attribute}'.")
                     is_significant_attr = True
                     # Optimization: If we only care *if* an attribute is significant,
                     # we could break the inner loop here. But we continue to ensure
                     # we collect *all* bin names for that attribute.
             else:
                  LOGGER.warning(f"Metrics entry for bins ({bin1}, {bin2}) in attribute '{attribute}' is not a dictionary: {metrics_dict}")


        # After checking all pairs for the attribute:
        if is_significant_attr:
            if all_bins_for_this_attr: # Ensure we collected bin names
                 LOGGER.info(f"Attribute '{attribute}' marked as significant for cross-model comparison.")
                 significant_attributes_found[attribute] = all_bins_for_this_attr
            else:
                 LOGGER.warning(f"Attribute '{attribute}' was marked significant, but no bin names were collected.")


    if not significant_attributes_found:
         LOGGER.warning(f"No attributes showed significant cross-bin differences (p < {alpha}) for path '{path_name}'.")
    else:
         LOGGER.info(f"Attributes identified for cross-model comparison: {list(significant_attributes_found.keys())}")

    return significant_attributes_found


# ======================================================
#  MODIFIED MAIN FUNCTION (for evaluation)
# ======================================================

def run_evaluation(
    full_attr: bool,
    run_id: str, # Mandatory identifier for the run (e.g., 'seed0')
    gw_checkpoint_path: str, # Explicit path to the checkpoint for this run
    model_version: str,
    output_parent_dir: str = "./evaluation_runs", # Parent dir for all runs
    dataset_csv: str = "./evaluation_set/attributes.csv", # Default dataset
    latent_dim: int = 12,
    encoders_hidden_dim: int = 256, # Default encoder hidden dim
    decoders_hidden_dim: int = 256, # Default decoder hidden dim
    debug_mode: bool = False,
    encoders_n_layers: int = 2, # Optional, adjust if needed
    decoders_n_layers: int = 2, 
    n=1, # Optional, adjust if needed
    parallel: bool = True, # Optional, adjust if needed
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Runs the evaluation for a single model configuration and saves results.

    Args:
        full_attr: Boolean indicating if the attribute domain uses color info.
        run_id: Unique identifier for this specific evaluation run (e.g., 'seed0').
        gw_checkpoint_path: Full path to the .ckpt file for this model run.
        model_version: String identifying the base model version (e.g., 'eowremm8').
        output_parent_dir: Base directory to store results for different runs.
        dataset_csv: Path to the dataset attributes file.
        latent_dim: Latent dimension of the GW.
        encoders_hidden_dim: Hidden dimension for GW encoders.
        decoders_hidden_dim: Hidden dimension for GW decoders.
        debug_mode: If True, enables debug settings.
        encoders_n_layers: Number of layers for GW encoders.
        decoders_n_layers: Number of layers for GW decoders.

    Returns:
        Tuple containing (Path to the saved analysis_results.pkl file, analysis_results dict) if successful, otherwise None.
    """
    LOGGER.info(f"--- Starting Evaluation for Run ID: {run_id} ---")
    my_hparams = {"temperature": 1, "alpha": 1} # Example hparams, adjust if needed
    seed = int(run_id.split("seed")[-1]) if "seed" in run_id else 0



    # --- Determine Config based on full_attr ---
    if full_attr :
        attr_domain_type = DomainModuleVariant.attr_legacy
        config_suffix = "avec_couleurs"
    else :
        attr_domain_type = DomainModuleVariant.attr_legacy_no_color
        config_suffix = "sans_couleurs"
    # Layer adjustments specific to your architecture (example)
    encoder_layers = encoders_n_layers
    decoder_layers = decoders_n_layers

    # --- Define Output Directory ---
    run_specific_dir_name = f"results_{model_version}_{config_suffix}_{run_id}"
    output_dir = os.path.join(output_parent_dir, run_specific_dir_name)

    if debug_mode:
        import time
        debug_timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(f"./evaluation_runs_debug", f"{run_specific_dir_name}_{debug_timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info(f"Output directory set to: {output_dir}")
    results_save_path = os.path.join(output_dir, "analysis_results.pkl")
    if os.path.exists(results_save_path):
        LOGGER.warning(f"Loading results from existing file: {results_save_path}")
        # Load existing results if available
        try:
            with open(results_save_path, 'rb') as f:
                analysis_results = pickle.load(f)
            LOGGER.info(f"Loaded existing analysis results successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to load existing results: {e}")
            return None
        return results_save_path, analysis_results

    # --- Setup Config and Load Model ---
    config = {
        "domains": [
            LoadedDomainConfig(
                domain_type=DomainModuleVariant.v_latents,
                checkpoint_path=Path("/mnt/HD2/alexis_data/checkpoints_backup/checkpoints/domain_v.ckpt"), # Ensure path is correct
            ),
            LoadedDomainConfig(
                domain_type=attr_domain_type,
                checkpoint_path=Path("/mnt/HD2/alexis_data/checkpoints_backup/checkpoints/domain_attr.ckpt"), # Ensure path is correct
                args=my_hparams,
            ),
        ],
        "global_workspace": {
            "latent_dim": latent_dim,
            "encoders": {"hidden_dim": encoders_hidden_dim, "n_layers": encoder_layers},
            "decoders": {"hidden_dim": decoders_hidden_dim, "n_layers": decoder_layers},
        },
        # Add other top-level config sections if load_global_workspace needs them
    }

    # --- Load Model ---
    try:

        if isinstance(gw_checkpoint_path, GlobalWorkspace2Domains):
            global_workspace = gw_checkpoint_path

        else:
            # Use the absolute path provided in the function argument
            gw_checkpoint_path_obj = Path(gw_checkpoint_path)

            global_workspace = load_global_workspace(gw_checkpoint_path_obj, config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"Using device: {device}")
        global_workspace.to(device) # Ensure model is on the correct device
        global_workspace.eval() # Set to evaluation mode
    except (FileNotFoundError, KeyError, AttributeError, RuntimeError) as e:
        LOGGER.error(f"Failed to load the Global Workspace model for run {run_id}: {e}", exc_info=True)
        return None # Stop this evaluation run

    # --- Initialize Analyzer ---
    analyzer = HueShapeAnalyzer(
        global_workspace=global_workspace,
        device=device,
        color=full_attr,
        output_dir=output_dir,
        debug = debug_mode,
        seed = seed, 
        num_bins=NUM_BINS,
        reverb_n = n,
        rgb=True,
        barycentre=True
    )

    # --- Run Analysis ---
    LOGGER.info(f"Running HueShapeAnalyzer analysis for run {run_id}...")
    try:
        analysis_results: Dict[str, Any] = analyzer.analyze_dataset(
            csv_path=dataset_csv,
            display_examples=True, # Or False if not needed
            binning_config=binning_config_6144,
            im_dir=dataset_csv.replace("/attributes.csv", ''),
            tilting=1,
            parrallel=parallel,
            conditioning=True,
        )
        LOGGER.info(f"Analysis completed for run {run_id}.")
    except Exception as e:
         LOGGER.error(f"Error during HueShapeAnalyzer analysis for run {run_id}: {e}", exc_info=True)
         return None


    # --- Save Results Dictionary ---
    try:
        with open(results_save_path, 'wb') as f:
            pickle.dump(analysis_results, f)
        LOGGER.info(f"Analysis results dictionary saved to: {results_save_path}")
        return results_save_path, analysis_results
    except Exception as e:
        LOGGER.error(f"Failed to save results dictionary for run {run_id} to {results_save_path}: {e}")
        return None # Indicate failure

    finally:
         # Clean up GPU memory if needed
         if 'global_workspace' in locals(): del global_workspace
         if 'analyzer' in locals(): del analyzer
         if torch.cuda.is_available():
             torch.cuda.empty_cache()
         LOGGER.info(f"--- Finished Evaluation for Run ID: {run_id} ---")

# ======================================================
# CROSS-MODEL COMPARISON FUNCTIONS (from first response)
# ======================================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper to Load Specific Bin Data from One Model's Results
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_hue_data_for_bin(
    results_path: str,
    attribute: str,
    bin_name: str,
    path_name: str = "translated" # e.g., 'translated', 'half_cycle'
) -> Optional[List[float]]:
    """Loads H-channel data for a specific attribute bin and path from saved results."""
    if not os.path.exists(results_path):
        LOGGER.error(f"Results file not found: {results_path}")
        return None
    try:
        with open(results_path, 'rb') as f:
            results_data = pickle.load(f)

        distribution_key = f'{path_name}_distributions'
        h_values = results_data.get(attribute, {}).get(distribution_key, {}).get(bin_name, {}).get('H', None)

        if h_values is None or not isinstance(h_values, list):
             LOGGER.warning(f"H data not found or not a list for {attribute}/{bin_name}/{path_name} in {results_path}")
             return None

        valid_h_values = [h for h in h_values if h is not None and not np.isnan(h)]
        if not valid_h_values:
            LOGGER.warning(f"No valid H data found for {attribute}/{bin_name}/{path_name} in {results_path} after filtering.")
            return None

        LOGGER.debug(f"Loaded {len(valid_h_values)} H values for {attribute}/{bin_name}/{path_name} from {results_path}")
        return valid_h_values

    except pickle.UnpicklingError:
        LOGGER.error(f"Error unpickling results file: {results_path}")
        return None
    except Exception as e:
        LOGGER.error(f"Unexpected error loading data from {results_path}: {e}", exc_info=True)
        return None

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Visualization: Multiple Distributions Overlay
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def visualize_multiple_distributions_overlay(
    distributions_dict: Dict[str, List[float]], title: str, save_path: str,
    num_bins: int = 50, xlabel: str = "Hue Value", ylabel: str = "Density",
    alpha: float = 0.6, histtype: str = 'stepfilled', density: bool = False ):
    """Plots multiple distributions (histograms) overlaid on the same axes."""
    if not distributions_dict:
        LOGGER.warning("No distributions provided for overlay visualization.")
        return
    plt.figure(figsize=(10, 6))
    all_values_list = [v for v in distributions_dict.values() if v] # Filter out empty lists
    if not all_values_list:
         LOGGER.warning("No valid data points in any distribution for overlay plot.")
         plt.close()
         return
    all_values = np.concatenate(all_values_list)
    min_val, max_val = (np.min(all_values), np.max(all_values)) if len(all_values) > 0 else (0, 255)
    if min_val == max_val: min_val -= 1; max_val += 1
    hist_range = (min_val, max_val)

    for label, values in distributions_dict.items():
        if values:
            plt.hist(values, bins=num_bins, range=hist_range, density=density,
                     alpha=alpha, label=f"{label} (n={len(values)})", histtype=histtype)
    plt.title(title, wrap=True)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if len(distributions_dict) > 1: plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try:
        plt.savefig(save_path); LOGGER.info(f"Saved overlay distribution plot to {save_path}")
    except Exception as e: LOGGER.error(f"Failed to save overlay plot {save_path}: {e}")
    plt.close()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Visualization: Metric Heatmap for Model Pairs
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def visualize_metric_heatmap_model_pairs(
    metric_results: Dict[Tuple[str, str], Dict[str, float]], metric_key: str,
    model_labels: List[str], save_path: str, title: Optional[str] = None,
    cmap: str = 'viridis', invert_cmap: bool = False, vmin: Optional[float] = None,
    vmax: Optional[float] = None, annotation_fmt: str = ".3f"):
    """Visualizes a heatmap of a specific metric calculated between pairs of models."""
    n_models = len(model_labels)
    if n_models < 2: LOGGER.warning("Skipping model pair heatmap - requires >= 2 models."); return
    matrix = np.full((n_models, n_models), np.nan)
    diag_val = 0.0 if 'kl' in metric_key.lower() or 'ks_stat' in metric_key.lower() or 'H' in metric_key else 1.0 if 'pval' in metric_key.lower() else np.nan

    for i, j in itertools.product(range(n_models), range(n_models)):
        if i == j: matrix[i, j] = diag_val; continue
        label_i, label_j = model_labels[i], model_labels[j]
        result_dict = metric_results.get((label_i, label_j), metric_results.get((label_j, label_i), None))
        if result_dict and metric_key in result_dict:
            value = result_dict[metric_key]
            if isinstance(value, (int, float)) and np.isfinite(value): matrix[i, j] = value

    if np.isnan(matrix).all(): LOGGER.warning(f"Skipping model pair heatmap for '{metric_key}' - no valid data."); return
    valid_values = matrix[~np.isnan(matrix) & np.isfinite(matrix)]
    if len(valid_values) == 0: LOGGER.warning(f"Skipping model pair heatmap for '{metric_key}' - only NaN/Inf values found."); return

    if vmin is None: vmin = np.min(valid_values) if len(valid_values) > 0 else 0
    if vmax is None: vmax = np.max(valid_values) if len(valid_values) > 0 else 1
    if 'pval' in metric_key.lower(): vmax = min(vmax, 1.0) # Cap pval display at 1.0
    if vmin == vmax: vmin -= 0.1; vmax += 0.1; vmin = max(0, vmin) if 'pval' in metric_key.lower() else vmin

    current_cmap = cmap + "_r" if invert_cmap and not cmap.endswith("_r") else cmap
    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.0), max(5, n_models * 0.8)))
    im = ax.imshow(matrix, cmap=current_cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n_models)); ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_labels, rotation=45, ha="right"); ax.set_yticklabels(model_labels)

    for i, j in itertools.product(range(n_models), range(n_models)):
        value = matrix[i, j]; value_text = "N/A"; text_color = "grey"
        if not np.isnan(value):
            norm_value = (value - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else 0.5
            text_color = "white" if norm_value < 0.5 else "black"
            if 'pval' in metric_key.lower() and value < 0.05: text_color = "red" if norm_value >= 0.5 else "lime"
            value_text = f"{value:{annotation_fmt}}" if np.isfinite(value) else ("Inf" if value > 0 else "-Inf")
        ax.text(j, i, value_text, ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label=metric_key.replace('_', ' ').title())
    if title is None: title = f"{metric_key.replace('_', ' ').title()} between Models"
    ax.set_title(title, wrap=True); plt.tight_layout()
    try:
        plt.savefig(save_path); LOGGER.info(f"Saved model pair heatmap to {save_path}")
    except Exception as e: LOGGER.error(f"Failed to save model pair heatmap {save_path}: {e}")
    plt.close(fig)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main Comparison Function Across Models for a Single Bin
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def calculate_distribution_metrics(dist1: np.ndarray, dist2: np.ndarray, 
                                  num_bins: int = 50) -> Dict[str, float]:
    """
    Calculate comparison metrics between two distributions.
    
    Args:
        dist1: First distribution array
        dist2: Second distribution array
        num_bins: Number of bins for histogram-based metrics
        
    Returns:
        Dictionary of metrics including KL divergence and KS test results
    """
    metrics = {}
    kl_h_12, kl_h_21, kl_h_sym, ks_stat, ks_pval = comparison_metrics(dist1, dist2, num_bins)
    # Calculate KL divergences
    metrics['kl_1_vs_2'] = kl_h_12
    metrics['kl_2_vs_1'] = kl_h_21
    metrics['kl_symmetric'] = kl_h_sym
    # KS test
    metrics['ks_stat'] = ks_stat
    metrics['ks_pval'] = ks_pval
    
    return metrics


def save_comparison_results(
    h_values1: List[float], h_values2: List[float],
    metrics: Dict[str, float],
    output_dir: str,
    comparison_name: str,
    label1: str, label2: str,
    target_attribute: str, target_bin_name: str, path_name: str,
    significance_alpha: float,
    num_histogram_bins: int = 50
) -> Tuple[str, str]:
    """
    Save comparison visualizations and summary to disk.
    
    Args:
        h_values1, h_values2: The H-channel values from each model
        metrics: Dictionary of computed metrics
        output_dir: Directory where files should be saved
        comparison_name: Base name for generated files
        label1, label2: Model labels for comparison
        target_attribute, target_bin_name, path_name: Details for report
        significance_alpha: Significance threshold used
        num_histogram_bins: Number of bins for histogram visualization
        
    Returns:
        Tuple of (overlay_plot_path, summary_path)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save overlay plot
    overlay_plot_path = os.path.join(output_dir, f"{comparison_name}_overlay.png")
    overlay_title = (f"Hue Dist: {target_attribute.title()} = {target_bin_name}\n"
                     f"Path: {path_name.title()} ({label1} vs {label2})")
    
    visualize_multiple_distributions_overlay(
        distributions_dict={label1: h_values1, label2: h_values2},
        title=overlay_title,
        save_path=overlay_plot_path,
        num_bins=num_histogram_bins
    )
    
    # 2. Save summary text file
    summary_path = os.path.join(output_dir, f"{comparison_name}_summary.txt")
    is_significant = metrics['ks_pval'] < significance_alpha
    
    try:
        with open(summary_path, 'w') as f:
            f.write(f"Cross-Model Hue Distribution Comparison Summary\n")
            f.write(f"Models: {label1} vs {label2}\n")
            f.write(f"Attribute: {target_attribute}\n")
            f.write(f"Bin: {target_bin_name}\n")
            f.write(f"Path: {path_name}\n\n")
            f.write(f"Significance Threshold (alpha): {significance_alpha}\n")
            f.write(f"KS Test p-value: {metrics['ks_pval']:.6f} ({'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'})\n")
            f.write(f"KS Test Statistic: {metrics['ks_stat']:.4f}\n\n")
            f.write("Metrics:\n")
            f.write(f"  - KL({label1}||{label2}): {metrics['kl_1_vs_2']:.4f}\n")
            f.write(f"  - KL({label2}||{label1}): {metrics['kl_2_vs_1']:.4f}\n")
            f.write(f"  - Symmetric KL: {metrics['kl_symmetric']:.4f}\n")
            f.write(f"\nData points: {label1}={len(h_values1)}, {label2}={len(h_values2)}\n")
            f.write(f"\nOverlay plot saved to: {os.path.basename(overlay_plot_path)}\n")
    except Exception as e:
        LOGGER.error(f"Failed to save summary file {summary_path}: {e}")
    
    return overlay_plot_path, summary_path


def compare_hue_distribution_for_bin_across_models(
    model_results_paths: List[str],
    target_attribute: str,
    target_bin_name: str,
    output_dir: str,
    path_name: str = "translated",
    model_labels: Optional[List[str]] = None,
    num_histogram_bins: int = 50,
    significance_alpha: float = 0.05,
    save_results: bool = True,
    save_nonsignificant: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Simplified function to compare H-channel distributions between two models.
    
    Args:
        model_results_paths: List containing paths to the 2 results files to compare
        target_attribute: The attribute to focus on (e.g., 'shape')
        target_bin_name: The specific bin within the attribute (e.g., 'diamond')
        output_dir: Directory where results should be saved
        path_name: The processing path results to compare ('translated', etc.)
        model_labels: List containing the 2 labels for the models
        num_histogram_bins: Number of bins for visualization and KL calculation
        significance_alpha: P-value threshold for significance
        save_results: Whether to save any results to disk
        save_nonsignificant: Whether to save results for non-significant comparisons
        
    Returns:
        Dictionary containing the calculated metrics and significance flag, or None if comparison failed
    """
    # --- 1. Validate inputs ---
    if not model_results_paths or len(model_results_paths) != 2:
        LOGGER.error("Exactly two model results paths required")
        return None
        
    if not model_labels or len(model_labels) != 2:
        model_labels = [f"Model{i+1}" for i in range(2)]
        LOGGER.warning(f"Using generic model labels: {model_labels}")
    
    label1, label2 = model_labels
    
    # --- 2. Load data ---
    h_values1 = load_hue_data_for_bin(model_results_paths[0], target_attribute, target_bin_name, path_name)
    h_values2 = load_hue_data_for_bin(model_results_paths[1], target_attribute, target_bin_name, path_name)
    
    if h_values1 is None or len(h_values1) <= 1 or h_values2 is None or len(h_values2) <= 1:
        LOGGER.warning(f"Insufficient H data for {target_attribute}/{target_bin_name} comparison")
        return None
    
    # --- 3. Calculate metrics ---
    try:
        metrics = calculate_distribution_metrics(
            np.array(h_values1), np.array(h_values2), 
            num_bins=num_histogram_bins
        )
    except Exception as e:
        LOGGER.error(f"Failed to calculate metrics: {e}")
        return None
    
    # --- 4. Check significance and conditionally save results ---
    is_significant = metrics['ks_pval'] < significance_alpha
    metrics['is_significant'] = is_significant
    
    LOGGER.info(f"Comparison {target_attribute}-{target_bin_name}: " + 
                f"p={metrics['ks_pval']:.4f} " +
                f"({'SIGNIFICANT' if is_significant else 'not significant'})")
    
    # Save results if requested and either significant or save_nonsignificant=True
    if save_results and (is_significant or save_nonsignificant):
        # Create simple comparison name
        safe_bin_name = target_bin_name.replace("/", "_").replace(" ", "_").replace(":", "-")
        output_dir = Path(output_dir)
        output_dir = output_dir / path_name / target_attribute / safe_bin_name
        comparison_name = f"{label1}_vs_{label2}"
        # Save the results
        save_comparison_results(
            h_values1, h_values2, metrics, output_dir, comparison_name,
            label1, label2, target_attribute, target_bin_name, path_name,
            significance_alpha, num_histogram_bins
        )
    
    return metrics



# ======================================================
#                     MAIN EXECUTION
# ======================================================

ablated_cycle_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High_cycle - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) High Cycle (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) High Cycle (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    # { "run_id": "colored0", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) High cycle (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"},
]

ablated_contrastive_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed1) HCo (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) H_Co (10) v5/checkpoints/last.ckpt" },
    # { "run_id": "colored0", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"},
]
ablated_translation_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High translation (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) High_Translation (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) H_T (10) v5/checkpoints/last.ckpt" },
    # { "run_id": "colored0", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) H_T (10) v5/checkpoints/last.ckpt"},
]
base_configs = [
        { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": f"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" }, # Replace!
        { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": f"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
        { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" }, # Replace!
        # { "run_id": "colored", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" }, # Replace!
    ]

high_alpha_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High alpha 5 (color) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) H_alpha (5) v5/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) H_alpha (5) v5/checkpoints/last.ckpt" },
    # { "run_id": "colored", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) H_alpha (5) v5/checkpoints/last.ckpt"},
]
# corrected model configs

high_contrastive_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) (bis) High_contrastive (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) (bis) High_contrastive (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) (bis) High_contrastive (10) v5/checkpoints/last.ckpt" },
    # { "run_id": "colored0", "full_attr": True,"gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) High_contrastive (1) v5/checkpoints/last.ckpt"}
]
high_cycle_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) High_cycles (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) (bis) High_cycles (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) High_cycles (10) v5/checkpoints/epoch=1225.ckpt" },
    # { "run_id": "colored0", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) High_cycle (10) v5/checkpoints/last.ckpt"},
]

high_translation_configs = [
    { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) (ter) High_translation (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) High_Translation (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) (ter) High_translation (10) v5/checkpoints/last.ckpt" },
    # { "run_id": "colored0", "full_attr": True,"gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) H_T (10) v5/checkpoints/last.ckpt"},
]



# Define all model versions to evaluate
REVERB_N = 1
model_versions_and_configs = [
    ("High_cycle_v5", high_cycle_configs),
    ("Base_params_v5", base_configs),
    ("High_alpha_v5", high_alpha_configs),
    ("High_contrastive_v5", high_contrastive_configs),
    ("High_translation_v5", high_translation_configs)
]

# REVERB_N = 2
# model_versions_and_configs = [
#     # ("Base_params_v5_n=2", base_configs),
#     ("High_cycle_v5_n=2", high_cycle_configs),
#     # ("High_contrastive_v5_n=2", high_contrastive_configs),
#     # ("High_alpha_v5_n=2", high_alpha_configs),
#     ("High_translation_v5_n=2", high_translation_configs)
# ]


# REVERB_N = 3
# model_versions_and_configs_n3 = [
#     ("Base_params_v5_n=3", base_configs),
#     ("High_cycle_v5_n=3", high_cycle_configs),
#     ("High_contrastive_v5_n=3", high_contrastive_configs),
#     ("High_alpha_v5_n=3", high_alpha_configs),
#     ("High_translation_v5_n=3", high_translation_configs)
# ]

# model_versions_and_configs = model_versions_and_configs_n3


DATASET_CSV = "./evaluation_set_FINAL/attributes.csv"
NUM_BINS = 15
PARALLEL = False # Set to True for parallel processing

if __name__ == "__main__":
    for MODEL_VERSION, model_configs in model_versions_and_configs:
        LOGGER.info(f"\n\n==========================================")
        LOGGER.info(f"=== EVALUATING MODEL VERSION: {MODEL_VERSION} ===")
        LOGGER.info(f"==========================================\n")
        
        evaluation_parent_dir = f"./FINAL_AVEC_BRUIT_0.01_v4/{MODEL_VERSION}"
        
        if len(model_configs) < 2:
            LOGGER.error(f"Need >= 2 model configs for {MODEL_VERSION}. Skipping.")
            continue
        
        os.makedirs(evaluation_parent_dir, exist_ok=True)

        # --- Run Evaluation for All Models ---
        model_results_paths = {}
        analysis_results_first_model = None
        successful_runs = []

        LOGGER.info(f"--- Starting Evaluation for {len(model_configs)} Models ---")
        for i, config in enumerate(model_configs):
            run_id = config["run_id"]
            LOGGER.info(f"--- Evaluating Model {i+1}/{len(model_configs)} (ID: {run_id}) ---")
            eval_result = run_evaluation(
                full_attr=config["full_attr"], run_id=run_id,
                gw_checkpoint_path=config["gw_checkpoint_path"], model_version=MODEL_VERSION,
                output_parent_dir=evaluation_parent_dir, debug_mode=DEBUG_MODE, dataset_csv=DATASET_CSV,
                n= REVERB_N, parallel=PARALLEL, # Pass parallel flag
            )
            if eval_result:
                results_path, analysis_results = eval_result
                model_results_paths[run_id] = results_path
                successful_runs.append(run_id)
                if i == 0: analysis_results_first_model = analysis_results
                del analysis_results
                LOGGER.info(f"Evaluation successful for {run_id}.")
            else:
                LOGGER.error(f"Evaluation failed for {run_id}.")

        # --- Check if Comparison is Possible ---
        if analysis_results_first_model is None:
            LOGGER.error(f"Eval failed for first model of {MODEL_VERSION}. Skipping to next version.")
            continue
            
        if len(successful_runs) < 2:
            LOGGER.error(f"Need >= 2 successful evals for {MODEL_VERSION}. Skipping to next version.")
            continue
            
        LOGGER.info(f"\nSuccessfully evaluated models for {MODEL_VERSION}: {successful_runs}")

        # --- Filter Attributes based on FIRST Model's Significance (Across Paths) ---
        paths_to_check = ['translated', 'half_cycle', 'full_cycle']
        significance_alpha_filter = 0.05 # Alpha for initial filtering based on run 1
        significance_alpha_compare = 0.05 # Alpha for saving cross-model results
        all_significant_attributes_union: Dict[str, Set[str]] = {}

        LOGGER.info(f"\n--- Filtering Attributes Based on FIRST Model ({successful_runs[0]}) Significance (p < {significance_alpha_filter}) For Each Path ---")
        for path_name in paths_to_check:
            significant_attributes_current_path = find_significant_bin_comparisons(
                analysis_results=analysis_results_first_model, path_name=path_name,
                alpha=significance_alpha_filter
            )
            for attr, bins_set in significant_attributes_current_path.items():
                all_significant_attributes_union.setdefault(attr, set()).update(bins_set)

        LOGGER.info("\n--- Significance Filtering Summary ---")
        if not all_significant_attributes_union:
            LOGGER.info(f"No attributes met the filtering criteria (p < {significance_alpha_filter}) in ANY path of the first model for {MODEL_VERSION}.")
        else:
            LOGGER.info(f"Attributes showing significance (p < {significance_alpha_filter}) in at least one path (and their bins):")
            for attr, bins_set in all_significant_attributes_union.items():
                LOGGER.info(f"  - {attr}: {sorted(list(bins_set))}")

        # --- Perform Pairwise Cross-Model Comparison (Conditionally Saving Details) ---
        if not all_significant_attributes_union:
            LOGGER.info(f"Skipping pairwise cross-model comparison for {MODEL_VERSION}.")
        else:
            LOGGER.info(f"\n--- Starting Pairwise Cross-Model Comparison for {len(successful_runs)} Models of {MODEL_VERSION} ---")
            LOGGER.info(f"--- Details will be saved ONLY if comparison KS p-value < {significance_alpha_compare} ---")
            comparison_output_dir = f"{evaluation_parent_dir}/cross_model_comparison_{MODEL_VERSION}_N{len(successful_runs)}"

            model_pairs = list(itertools.combinations(successful_runs, 2))
            LOGGER.info(f"Performing {len(model_pairs)} pairwise comparisons.")

            num_significant_comparisons_found = 0

            # Iterate through each PAIR of models
            for id1, id2 in model_pairs:
                LOGGER.debug(f"\n--- Comparing Pair: {id1} vs {id2} ---")
                path1 = model_results_paths[id1]
                path2 = model_results_paths[id2]

                # Iterate through attributes found significant in the *first* model
                for attribute, bin_names_set in all_significant_attributes_union.items():
                    # Iterate through the paths ('translated', 'half_cycle', 'full_cycle')
                    for path_name_for_comparison in paths_to_check:
                        # Iterate through the bins associated with the significant attribute
                        for bin_name in sorted(list(bin_names_set)):
                            # Call the modified comparison function
                            metrics_dict = compare_hue_distribution_for_bin_across_models(
                                model_results_paths=[path1, path2],
                                target_attribute=attribute,
                                target_bin_name=bin_name,
                                output_dir=comparison_output_dir, # Pass base dir
                                path_name=path_name_for_comparison,
                                model_labels=[id1, id2],
                                num_histogram_bins=NUM_BINS,
                                significance_alpha=significance_alpha_compare # Pass alpha for saving
                            )

                            if metrics_dict and metrics_dict.get('is_significant', False):
                                num_significant_comparisons_found += 1
                            elif metrics_dict is None:
                                LOGGER.warning(f"      Comparison failed for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2}.")

            LOGGER.info(f"\nPairwise cross-model comparisons finished for {MODEL_VERSION}.")
            if num_significant_comparisons_found > 0:
                LOGGER.info(f"Found and saved details for {num_significant_comparisons_found} significant comparisons.")
                LOGGER.info(f"Significant comparison results saved in subdirectories within: {comparison_output_dir}")
            else:
                LOGGER.info(f"No significant differences found between model pairs for {MODEL_VERSION} for the filtered attributes/bins/paths.")
                # Optionally delete the comparison_output_dir if it's empty and was created
                try:
                    if os.path.exists(comparison_output_dir) and not os.listdir(comparison_output_dir):
                        os.rmdir(comparison_output_dir)
                        LOGGER.info(f"Removed empty output directory: {comparison_output_dir}")
                except OSError as e:
                    LOGGER.warning(f"Could not remove empty directory {comparison_output_dir}: {e}")

    LOGGER.info("\n=== All Model Versions Evaluated ===")
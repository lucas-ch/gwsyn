import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from matplotlib.figure import Figure
from matplotlib.axes import Axes


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


def visualize_distributions_flexible(
    distributions_data: Dict[str, Dict[str, List[float]]],
    attr: str,
    bin_name: str,
    save_path: str,
    visualization_mode: str = "channel_per_path",  # or "path_per_channel"
    channels_to_plot: List[str] = None,
    paths_to_plot: List[str] = None,
    title: Optional[str] = None,
    num_bins: int = 15,
    hist_range: Optional[Dict[str, Tuple[float, float]]] = None,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (12, 7),
    colors: Optional[List[str]] = None,
    density: bool = False,
    filename_suffix: str = ""
) -> None:
    """
    Flexible function to visualize distributions with two main modes:
    1. "channel_per_path": One figure showing same channel (e.g., H) across multiple paths
    2. "path_per_channel": One figure showing multiple channels (e.g., R,G,B) for a single path
    
    Args:
        distributions_data: Dict mapping path names to their data dictionaries
                         Format: {'translated': {'H': [val1,...], 'R': [val1,...]}, 'full_cycle': {...}}
        attr: Name of the attribute being analyzed
        bin_name: Name of the bin being visualized
        save_path: Directory where to save the output figures
        visualization_mode: Either "channel_per_path" or "path_per_channel"
        channels_to_plot: List of channels to include (e.g., ["H"] or ["R", "G", "B"])
        paths_to_plot: List of paths to include (e.g., ["translated", "full_cycle"])
        title: Optional custom title override. If None, a default title is generated
        num_bins: Number of bins for the histograms
        hist_range: Dictionary mapping channel to its (min, max) range
        alpha: Transparency for histogram bars
        figsize: Size of the figure in inches
        colors: Optional custom colors list. If None, defaults are used
        density: Whether to normalize the histogram to form a probability density
        filename_suffix: Additional string to append to the saved filename
    """
    # Set defaults if not provided
    if channels_to_plot is None:
        channels_to_plot = ["H"] if visualization_mode == "channel_per_path" else ["R", "G", "B"]
    
    if paths_to_plot is None:
        paths_to_plot = list(distributions_data.keys())
    
    # Filter to only include the paths we want to plot
    filtered_distributions = {p: distributions_data[p] for p in paths_to_plot if p in distributions_data}
    
    # Set default colors if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    
    # Ensure output directory exists
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set default histogram ranges if not provided
    if hist_range is None:
        hist_range = {
            'H': (0, 360),  # Hue in degrees
            'R': (0, 255),  # RGB values
            'G': (0, 255),
            'B': (0, 255)
        }
    
    if visualization_mode == "channel_per_path":
        # Mode 1: For each channel, create one plot with all paths
        for channel in channels_to_plot:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot each path's data for this channel
            for i, (path_name, path_data) in enumerate(filtered_distributions.items()):
                if channel in path_data:
                    values = np.array([x for x in path_data[channel] if x is not None and not np.isnan(x)])
                    if len(values) > 0:
                        color_idx = i % len(colors)
                        path_label = path_name.replace('_', ' ').title()
                        ax.hist(values, bins=num_bins, range=hist_range.get(channel), density=density,
                                alpha=alpha, label=f"{path_label} (N={len(values)})", color=colors[color_idx])
            
            # Format and save
            ax.set_xlabel(f"{channel} Value")
            ax.set_ylabel("Density" if density else "Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if title is None:
                title = f"{channel}-Channel Comparison Across Paths\nAttribute: {attr.title()}, Bin: {bin_name}"
            ax.set_title(title, wrap=True)
            
            file_name = f'{attr}_{bin_name}_{channel}_multi_path{filename_suffix}.png'
            output_file = os.path.join(output_path, file_name)
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved channel-per-path plot to {output_file}")
            plt.close(fig)
            
    elif visualization_mode == "path_per_channel":
        # Mode 2: For each path, create one plot with all channels
        for path_name, path_data in filtered_distributions.items():
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot each channel for this path
            for i, channel in enumerate(channels_to_plot):
                if channel in path_data:
                    values = np.array([x for x in path_data[channel] if x is not None and not np.isnan(x)])
                    if len(values) > 0:
                        color_idx = i % len(colors)
                        ax.hist(values, bins=num_bins, range=hist_range.get(channel), density=density,
                                alpha=alpha, label=f"{channel} (N={len(values)})", color=colors[color_idx])
            
            # Format and save
            ax.set_xlabel("Channel Value")
            ax.set_ylabel("Density" if density else "Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if title is None:
                path_title = path_name.replace('_', ' ').title()
                channels_str = ", ".join(channels_to_plot)
                title = f"{path_title}: {channels_str} Channel Distributions\nAttribute: {attr.title()}, Bin: {bin_name}"
            ax.set_title(title, wrap=True)
            
            file_name = f'{attr}_{bin_name}_{path_name}_multi_channel{filename_suffix}.png'
            output_file = os.path.join(output_path, file_name)
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved path-per-channel plot to {output_file}")
            plt.close(fig)
    else:
        raise ValueError(f"Invalid visualization_mode: {visualization_mode}. Must be 'channel_per_path' or 'path_per_channel'.")


def visualize_multiple_distributions(
    distributions: List[Dict[str, List[float]]],
    labels: List[str],
    bin_name: str,
    attr: str,
    save_path: str,
    channels: List[str] = ["H"],
    title: Optional[str] = None,
    num_bins: int = 15,
    hist_range: Optional[Dict[str, Tuple[float, float]]] = None,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (12, 7),
    colors: Optional[List[str]] = None,
    density: bool = False
) -> None:
    """
    Generates and saves a histogram plot comparing multiple distributions for specified channels.
    
    Args:
        distributions: List of dictionaries containing data for each distribution.
                      Expected format: [{'H': [val1, val2,...], ...}, {...}, ...]
        labels: List of labels for each distribution (e.g., ["Input", "Translated", "Half-cycle", "Full-cycle"])
        bin_name: Name of the bin or description of the comparison
        attr: Name of the attribute being analyzed
        save_path: Full path where the plot images will be saved
        channels: List of channel keys to plot (e.g., ['H'] or ['H', 'R', 'G', 'B'])
        title: Optional custom title for the plot. If None, a default title is generated
        num_bins: Number of bins to use in the histograms
        hist_range: Dictionary mapping channel to its (min, max) range for the histogram X-axis.
                   If None, defaults are used (0, 360) for H and (0, 255) for RGB
        alpha: Transparency level for histogram bars
        figsize: Size of the figure (width, height) in inches
        colors: Optional list of colors to use for each distribution. If None, uses default colors
        density: Whether to normalize the histogram to form a probability density
    """
    # Validate input
    if len(distributions) != len(labels):
        raise ValueError(f"Number of distributions ({len(distributions)}) must match number of labels ({len(labels)})")
    
    # Set default colors if not provided
    if colors is None:
        # Get default color cycle (will cycle if more distributions than colors)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    
    # Ensure the output directory exists
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating output directory: {output_path}")
    
    # Set default histogram ranges if not provided
    if hist_range is None:
        hist_range = {
            'H': (0, 360),  # Hue in degrees
            'R': (0, 255),  # RGB values
            'G': (0, 255),
            'B': (0, 255)
        }
    
    # Process each channel separately
    for channel in channels:
        # Extract channel data for all distributions, removing None/NaN
        values_list = []
        for dist in distributions:
            values = np.array([x for x in dist.get(channel, []) if x is not None and not np.isnan(x)])
            values_list.append(values)
        
        # Check if we have any data to plot
        if all(len(values) == 0 for values in values_list):
            warnings.warn(f"No valid {channel} data for any distribution for attribute '{attr}', bin '{bin_name}'. Skipping plot.")
            continue
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram for each distribution
        for i, (values, label) in enumerate(zip(values_list, labels)):
            if len(values) > 0:
                color_idx = i % len(colors)  # Handle case where more distributions than colors
                ax.hist(values, bins=num_bins, range=hist_range.get(channel), density=density,
                        alpha=alpha, label=f"{label} (N={len(values)})", color=colors[color_idx])
        
        # --- Formatting ---
        ax.set_xlabel(f"{channel} Value")
        ax.set_ylabel("Density" if density else "Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if title is None:
            # Generate a default title if none provided
            distributions_str = " vs ".join(labels)
            title = f"{channel}-Channel Distribution Comparison: {distributions_str}\nAttribute: {attr.title()}, Bin: {bin_name}"
        ax.set_title(title, wrap=True)
        
        # --- Saving ---
        file_name = f'{attr}_{bin_name}_{channel}_multi_comparison.png'
        output_file = os.path.join(output_path, file_name)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved multi-distribution plot to {output_file}")
        
        # Close the figure to free memory
        plt.close(fig)


def create_standard_visualization_set(
    output_dir: str,
    save_path: str,
    attr: str,
    bin_name: str,
    include_input: bool = False,
    include_half_cycle: bool = False
) -> None:
    """
    Creates a standard set of distribution visualizations for a specific attribute and bin:
    1. Hue comparison across paths (translated vs full_cycle)
    2. RGB channels for translated path
    3. RGB channels for full_cycle path
    
    Args:
        output_dir: Directory where binned results are saved
        save_path: Directory where plots will be saved
        attr: The attribute to analyze
        bin_name: The bin to analyze
        include_input: Whether to include input distribution in Hue comparison
        include_half_cycle: Whether to include half_cycle distribution in Hue comparison
    """
    # Load binned results
    result_data = load_binned_results(output_dir)
    if result_data is None:
        print("No results data available. Cannot create visualizations.")
        return
    
    # Unpack result data
    (input_colors_by_attr, translated_colors_by_attr, 
     half_cycle_colors_by_attr, full_cycle_colors_by_attr,
     _, _, _) = result_data
    
    # Create a dictionary of path data
    paths_data = {
        'input': input_colors_by_attr.get(attr, {}).get(bin_name, {}),
        'translated': translated_colors_by_attr.get(attr, {}).get(bin_name, {}),
        'half_cycle': half_cycle_colors_by_attr.get(attr, {}).get(bin_name, {}),
        'full_cycle': full_cycle_colors_by_attr.get(attr, {}).get(bin_name, {})
    }
    paths_data['input']['H'] = [2*x for x in paths_data['input']['H']]

    # Print the mean of the difference between reconstructed full cycle and ground truth
    # Print the mean of the difference between reconstructed full cycle and ground truth
    if 'input' in paths_data and 'full_cycle' in paths_data:
        print(f"\nMean Differences Between Full Cycle and Ground Truth for {attr}, bin {bin_name}:")
        
        # List of channels to compare
        channels_to_compare = ['H']
        
        for channel in channels_to_compare:
            # Check if both have this channel
            if channel in paths_data['input'] and channel in paths_data['full_cycle']:
                # Get values, ignoring None and NaN
                input_values = np.array([x for x in paths_data['input'][channel] if x is not None and not np.isnan(x)])
                full_cycle_values = np.array([x for x in paths_data['full_cycle'][channel] if x is not None and not np.isnan(x)])
                random_values = np.random.uniform(0, 360, size=len(input_values))
                
                diff = full_cycle_values - input_values
                
                mean_diff = np.mean(diff)
                var_diff = np.var(diff)


                print(f"  {channel}-Channel: Mean Difference = {mean_diff:.2f}")
                print(f"  {channel}-Channel: Variance Difference = {var_diff:.2f}")


                 
            else:
                if channel not in paths_data['input']:
                    print(f"  {channel}-Channel: Not available in input data")
                else:
                    print(f"  {channel}-Channel: Not available in full cycle data")
    else:
        if 'input' not in paths_data:
            print("\nGround truth (input) data not available for comparison")
        else:
            print("\nFull cycle data not available for comparison")
    # Filter out any empty paths
    paths_data = {k: v for k, v in paths_data.items() if v}
    
    if not paths_data:
        print(f"No data available for attribute '{attr}', bin '{bin_name}'. Skipping visualizations.")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(save_path) / attr / bin_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Hue comparison across paths (translated vs full_cycle and optionally others)
    paths_to_include = ['translated', 'full_cycle']
    if include_input:
        paths_to_include.insert(0, 'input')
    if include_half_cycle:
        paths_to_include.insert(-1, 'half_cycle')
    
    # Filter to only include paths that have data
    paths_to_include = [p for p in paths_to_include if p in paths_data]
    
    if len(paths_to_include) >= 2:  # Need at least 2 paths to compare
        visualize_distributions_flexible(
            distributions_data=paths_data,
            attr=attr,
            bin_name=bin_name,
            save_path=str(output_path),
            visualization_mode="channel_per_path",
            channels_to_plot=["H"],
            paths_to_plot=paths_to_include,
            title=f"Hue Distribution Comparison\nAttribute: {attr.title()}, Bin: {bin_name}",
            filename_suffix="_hue_comparison"
        )
    
    # 2. RGB channels for translated path
    if 'translated' in paths_data:
        visualize_distributions_flexible(
            distributions_data=paths_data,
            attr=attr,
            bin_name=bin_name,
            save_path=str(output_path),
            visualization_mode="path_per_channel",
            channels_to_plot=["B", "R", "G"],
            paths_to_plot=["translated"],
            title=f"Translated RGB Channels\nAttribute: {attr.title()}, Bin: {bin_name}",
            filename_suffix="_rgb_channels"
        )
    
    # 3. RGB channels for full_cycle path
    if 'full_cycle' in paths_data:
        visualize_distributions_flexible(
            distributions_data=paths_data,
            attr=attr,
            bin_name=bin_name,
            save_path=str(output_path),
            visualization_mode="path_per_channel",
            channels_to_plot=["R", "G", "B"],
            paths_to_plot=["full_cycle"],
            title=f"Full Cycle RGB Channels\nAttribute: {attr.title()}, Bin: {bin_name}",
            filename_suffix="_rgb_channels"
        )


def compare_distributions_from_binned_results(
    output_dir: str,
    save_path: str,
    distributions_to_compare: List[str],
    attrs_to_analyze: Optional[List[str]] = None,
    bins_to_analyze: Optional[Dict[str, List[str]]] = None,
    channels: List[str] = ["H"],
    num_bins: int = 15,
    custom_titles: Optional[Dict[str, Dict[str, str]]] = None,
    density: bool = False
) -> None:
    """
    Load binned results and compare distributions across specified attributes and bins.
    
    Args:
        output_dir: Directory where binned results are saved
        save_path: Directory where comparison plots will be saved
        distributions_to_compare: List of distribution names to compare
                                 Valid values: 'input', 'translated', 'half_cycle', 'full_cycle'
        attrs_to_analyze: List of attributes to analyze. If None, all available attributes are used
        bins_to_analyze: Dictionary mapping attributes to lists of bins to analyze.
                        If None, all available bins for each attribute are used
        channels: List of channels to analyze (e.g., ['H'] or ['H', 'R', 'G', 'B'])
        num_bins: Number of bins for the histograms
        custom_titles: Optional dictionary mapping attributes and bins to custom titles
        density: Whether to normalize the histogram to form a probability density
    """
    # Load binned results
    result_data = load_binned_results(output_dir)
    if result_data is None:
        print("No results data available. Cannot compare distributions.")
        return
    
    # Unpack result data
    (input_colors_by_attr, translated_colors_by_attr, 
     half_cycle_colors_by_attr, full_cycle_colors_by_attr,
     examples_by_attr, binning_config, analysis_attributes) = result_data
    
    # Map distribution names to their data objects
    dist_data_map = {
        'input': input_colors_by_attr,
        'translated': translated_colors_by_attr,
        'half_cycle': half_cycle_colors_by_attr,
        'full_cycle': full_cycle_colors_by_attr
    }
    
    # Validate requested distributions
    invalid_dists = [d for d in distributions_to_compare if d not in dist_data_map]
    if invalid_dists:
        raise ValueError(f"Invalid distribution names: {invalid_dists}. "
                         f"Valid options are: {list(dist_data_map.keys())}")
    
    # Use all available attributes if none specified
    if attrs_to_analyze is None:
        attrs_to_analyze = analysis_attributes
    
    # Process each attribute
    for attr in attrs_to_analyze:
        if attr not in input_colors_by_attr:
            print(f"Warning: Attribute '{attr}' not found in results. Skipping.")
            continue
        
        # Determine which bins to analyze for this attribute
        if bins_to_analyze and attr in bins_to_analyze:
            attr_bins = bins_to_analyze[attr]
        else:
            # Use all available bins for this attribute
            attr_bins = list(input_colors_by_attr[attr].keys())
        
        # Process each bin
        for bin_name in attr_bins:
            # Prepare data for each requested distribution
            distributions = []
            labels = []
            
            for dist_name in distributions_to_compare:
                dist_data = dist_data_map[dist_name]
                if attr in dist_data and bin_name in dist_data[attr]:
                    distributions.append(dist_data[attr][bin_name])
                    # Format label with nice capitalization
                    label = dist_name.replace('_', ' ').title()
                    labels.append(label)
                else:
                    print(f"Warning: No data for distribution '{dist_name}', "
                          f"attribute '{attr}', bin '{bin_name}'. Skipping this distribution.")
            
            if not distributions:
                print(f"No valid distributions found for attribute '{attr}', bin '{bin_name}'. Skipping.")
                continue
            
            # Get custom title if provided
            title = None
            if custom_titles and attr in custom_titles and bin_name in custom_titles[attr]:
                title = custom_titles[attr][bin_name]
            
            # Create comparison plot
            bin_save_path = os.path.join(save_path, attr, bin_name)
            visualize_multiple_distributions(
                distributions=distributions,
                labels=labels,
                bin_name=bin_name,
                attr=attr,
                save_path=bin_save_path,
                channels=channels,
                title=title,
                num_bins=num_bins,
                density=density
            )

# if __name__ == "__main__":
#     # Example 1: Directly compare distributions
#     dist1 = {'H': np.random.normal(120, 30, 1000)}
#     dist2 = {'H': np.random.normal(180, 20, 800)}
#     dist3 = {'H': np.random.normal(240, 40, 1200)}
    
#     visualize_multiple_distributions(
#         distributions=[dist1, dist2, dist3],
#         labels=["Distribution 1", "Distribution 2", "Distribution 3"],
#         bin_name="Example",
#         attr="demo",
#         save_path="./output/demo",
#         title="Example Multi-Distribution Comparison"
#     )
    
    # Example 2: Compare from saved binned results
    # compare_distributions_from_binned_results(
    #     output_dir="./experiment_results",
    #     save_path="./comparison_results",
    #     distributions_to_compare=["input", "translated", "half_cycle", "full_cycle"],
    #     attrs_to_analyze=["shape", "color"],
    #     channels=["H", "R", "G", "B"]
    # )

# Example usage
if __name__ == "__main__":
    # # Example 1: Directly compare distributions
    # dist1 = {'H': np.random.normal(120, 30, 1000)}
    # dist2 = {'H': np.random.normal(180, 20, 800)}
    # dist3 = {'H': np.random.normal(240, 40, 1200)}
    
    # visualize_multiple_distributions(
    #     distributions=[dist1, dist2, dist3],
    #     labels=["Distribution 1", "Distribution 2", "Distribution 3"],
    #     bin_name="Example",
    #     attr="demo",
    #     save_path="./Figures_test",
    #     title="Example Multi-Distribution Comparison"
    # )
    
    # Example 2: Compare from saved binned results
    # compare_distributions_from_binned_results(
    #     output_dir="FINAL/DEBUG/Base_params_v5/results_Base_params_v5_sans_couleurs_seed0/",
    #     save_path="./Figures_test",
    #     distributions_to_compare=["translated", "full_cycle"],
    #     attrs_to_analyze=["rotation"],
    #     channels=["H", "R", "G", "B"]
    # )

        # Example 4: Generate standard set of visualizations
    create_standard_visualization_set(
        output_dir="FINAL/Base_params_v5/results_Base_params_v5_sans_couleurs_seed0",
        save_path="./Figures_test",
        attr="rotation",
        bin_name="5 Œx2pi_16",
        include_input=False,
    )
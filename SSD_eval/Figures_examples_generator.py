import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle


def load_binned_results(output_dir):
    """
    Load previously saved binned results from disk.
    """
    save_path = os.path.join(output_dir, 'binned_results.pkl')
    
    if not os.path.exists(save_path):
        print(f"No saved binned results found at {save_path}")
        return None
        
    with open(save_path, 'rb') as f:
        result_data = pickle.load(f)
    
    return (
        result_data['input_colors_by_attr'],
        result_data['translated_colors_by_attr'],
        result_data['half_cycle_colors_by_attr'],
        result_data['full_cycle_colors_by_attr'],
        result_data['examples_by_attr'],
        result_data['binning_config'],
        result_data['analysis_attributes']
    )


def visualize_compact_examples(
    output_dir: str,
    save_path: str,
    attr: str,
    bin_name: str,
    paths_to_show: List[str] = ["input", "translated", "full_cycle"],
    num_examples: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    random_seed: Optional[int] = 42
) -> None:
    """
    Creates a compact visualization showing examples from different paths side by side.
    This function creates a simple, concise visualization showing input and reconstructed images.
    
    Args:
        output_dir: Directory where binned results are saved
        save_path: Directory where the visualization will be saved
        attr: The attribute to visualize
        bin_name: The bin to visualize
        paths_to_show: List of paths to display (e.g., ["input", "translated", "full_cycle"])
        num_examples: Number of examples to show
        figsize: Size of the figure in inches
        random_seed: Random seed for reproducible example selection
    """
    # Load binned results
    result_data = load_binned_results(output_dir)
    if result_data is None:
        print("No results data available. Cannot create examples visualization.")
        return
    
    # Unpack result data
    (_, _, _, _, examples_by_attr, _, _) = result_data
    
    # Ensure the examples exist for this attribute and bin
    if attr not in examples_by_attr or bin_name not in examples_by_attr[attr]:
        print(f"No examples found for attribute '{attr}', bin '{bin_name}'")
        return
    
    # Get examples for this attribute and bin
    examples = examples_by_attr[attr][bin_name]
    
    # If we have fewer examples than requested, use all of them
    num_available = len(examples)
    if num_examples > num_available:
        num_examples = num_available
        print(f"Only {num_available} examples available for attribute '{attr}', bin '{bin_name}'")
    
    # Select random subset of examples if needed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if num_examples < num_available:
        selected_indices = np.random.choice(num_available, num_examples, replace=False)
        selected_examples = [examples[i] for i in selected_indices]
    else:
        selected_examples = examples
    
    # Filter paths to only include valid ones
    valid_paths = [p for p in paths_to_show if any(hasattr(ex, f'{p}_image') for ex in selected_examples)]
    
    if not valid_paths:
        print(f"None of the specified paths {paths_to_show} have valid images.")
        return
    
    # Create the figure
    fig, axes = plt.subplots(num_examples, len(valid_paths), figsize=figsize)
    
    # Handle case with single example
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Add column headers
    for col_idx, path in enumerate(valid_paths):
        path_title = path.replace('_', ' ').title()
        axes[0, col_idx].set_title(path_title, fontsize=12)
    
    # For each example and path, display the image
    for row_idx, example in enumerate(selected_examples):
        # Add attr_value as row header if available
        if hasattr(example, 'attr_value'):
            axes[row_idx, 0].set_ylabel(f"{attr}: {example.attr_value:.2f}", fontsize=10)
        
        for col_idx, path in enumerate(valid_paths):
            ax = axes[row_idx, col_idx]
            
            # Get the image data
            if f'{path}_image' in example:
                img = example[f'{path}_image']
                
                if img is not None:
                    # Check if image is a PIL Image
                    if hasattr(img, 'mode') and hasattr(img, 'size'):
                        # Convert PIL image to numpy array if needed
                        img = np.array(img)
                    
                    # Convert to RGB if needed
                    if img.ndim == 2:  # Grayscale
                        img = np.stack([img, img, img], axis=2)
                    
                    # Normalize numpy array if values are very small (scientific notation)
                    if img.max() < 1e-5:
                        # Check if it's an array with tiny values that needs scaling
                        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                    
                    # Display the image
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, "No image", ha='center', va='center', fontsize=9)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=9)
                ax.axis('off')
    
    # Add overall title
    plt.suptitle(f"{attr.title()} - {bin_name}", fontsize=14)
    
    # Create output directory if it doesn't exist
    output_path = Path(save_path) / attr / bin_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    file_name = f'{attr}_{bin_name}_comparison.png'
    output_file = os.path.join(output_path, file_name)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved examples comparison to {output_file}")
    plt.close(fig)


def visualize_examples_grid(
    output_dir: str,
    save_path: str,
    attr: str,
    bin_name: str,
    paths_to_show: List[str] = ["input", "translated", "full_cycle"],
    num_examples: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    random_seed: Optional[int] = 42,
    display_attr_value: bool = True
) -> None:
    """
    Creates a grid visualization showing examples from different paths side by side.
    
    Args:
        output_dir: Directory where binned results are saved
        save_path: Directory where the visualization will be saved
        attr: The attribute to visualize
        bin_name: The bin to visualize
        paths_to_show: List of paths to display (e.g., ["input", "translated", "full_cycle"])
        num_examples: Number of examples to show
        figsize: Size of the figure in inches
        random_seed: Random seed for reproducible example selection
        display_attr_value: Whether to display the attribute value for each example
    """
    # Load binned results
    result_data = load_binned_results(output_dir)
    if result_data is None:
        print("No results data available. Cannot create examples visualization.")
        return
    
    # Unpack result data
    (input_colors_by_attr, translated_colors_by_attr, 
     half_cycle_colors_by_attr, full_cycle_colors_by_attr,
     examples_by_attr, _, _) = result_data
    
    # Ensure the examples exist for this attribute and bin
    if attr not in examples_by_attr or bin_name not in examples_by_attr[attr]:
        print(f"No examples found for attribute '{attr}', bin '{bin_name}'")
        return
    
    # Get examples for this attribute and bin
    examples = examples_by_attr[attr][bin_name]
    
    # If we have fewer examples than requested, use all of them
    num_available = len(examples)
    if num_examples > num_available:
        num_examples = num_available
        print(f"Only {num_available} examples available for attribute '{attr}', bin '{bin_name}'")
    
    # Select random subset of examples if needed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if num_examples < num_available:
        selected_indices = np.random.choice(num_available, num_examples, replace=False)
        selected_examples = [examples[i] for i in selected_indices]
    else:
        selected_examples = examples
    
    # Get path data
    path_data_map = {
        'input': input_colors_by_attr,
        'translated': translated_colors_by_attr,
        'half_cycle': half_cycle_colors_by_attr,
        'full_cycle': full_cycle_colors_by_attr
    }
    
    # Filter to only include paths that exist in our data
    valid_paths = []
    for path in paths_to_show:
        if path in path_data_map:
            valid_paths.append(path)
    
    if not valid_paths:
        print(f"None of the specified paths {paths_to_show} are valid.")
        return
    
    # Create the figure and subplots
    num_cols = len(valid_paths)
    num_rows = num_examples
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Adjust for the case of a single example
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Set the title for the entire figure
    fig.suptitle(f"Examples for Attribute: {attr.title()}, Bin: {bin_name}", fontsize=16)
    
    # Add column headers (path names)
    for col_idx, path in enumerate(valid_paths):
        path_title = path.replace('_', ' ').title()
        axes[0, col_idx].set_title(path_title, fontsize=14)
    
    # For each example and path, display the image
    for row_idx, example in enumerate(selected_examples):
        for col_idx, path in enumerate(valid_paths):
            ax = axes[row_idx, col_idx]
            
            # Get the image data
            if f'{path}_image' in example:
                
               
                img = example[f'{path}_image']
                
                if img is not None:
                    # Check if image is a PIL Image
                    if hasattr(img, 'mode') and hasattr(img, 'size'):
                        # Convert PIL image to numpy array if needed
                        img = np.array(img)
                    
                    # Convert to RGB if needed
                    if img.ndim == 2:  # Grayscale
                        img = np.stack([img, img, img], axis=2)
                    
                    # Normalize numpy array if values are very small (scientific notation)
                    if img.max() < 1e-5:
                        # Check if it's an array with tiny values that needs scaling
                        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                    
                    # Display the image
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Add attribute value as caption if requested
                    if display_attr_value and hasattr(example, 'attr_value'):
                        ax.set_xlabel(f"{attr}: {example.attr_value:.3f}", fontsize=10)
                else:
                    ax.text(0.5, 0.5, f"No {path} image", 
                           horizontalalignment='center', verticalalignment='center')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"No {path} attribute", 
                       horizontalalignment='center', verticalalignment='center')
                ax.axis('off')
    
    # Create output directory if it doesn't exist
    output_path = Path(save_path) / attr / bin_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    file_name = f'{attr}_{bin_name}_examples_grid.png'
    output_file = os.path.join(output_path, file_name)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved examples grid to {output_file}")
    plt.close(fig)


def process_examples_for_attribute(
    output_dir: str,
    save_path: str,
    attr: str,
    bins_to_process: Optional[List[str]] = None,
    paths_to_show: List[str] = ["input", "translated", "full_cycle"],
    num_examples: int = 5
) -> None:
    """
    Process and visualize examples for all bins of a specific attribute.
    
    Args:
        output_dir: Directory where binned results are saved
        save_path: Directory where visualizations will be saved
        attr: The attribute to process
        bins_to_process: List of bins to process. If None, all bins for this attribute are used
        paths_to_show: List of paths to display
        num_examples: Number of examples to show per bin
    """
    # Load binned results
    result_data = load_binned_results(output_dir)
    if result_data is None:
        print("No results data available. Cannot process examples.")
        return
    
    # Unpack result data
    (input_colors_by_attr, _, _, _, examples_by_attr, _, _) = result_data
    
    # Check if attribute exists
    if attr not in examples_by_attr:
        print(f"Attribute '{attr}' not found in examples data.")
        return
    
    # Get all bins for this attribute if none specified
    if bins_to_process is None:
        bins_to_process = list(examples_by_attr[attr].keys())
    
    # Process each bin
    for bin_name in bins_to_process:
        if bin_name not in examples_by_attr[attr]:
            print(f"Bin '{bin_name}' not found for attribute '{attr}'. Skipping.")
            continue
        
        print(f"Generating examples visualization for attribute '{attr}', bin '{bin_name}'...")
        
        # Create the examples grid visualization
        visualize_examples_grid(
            output_dir=output_dir,
            save_path=save_path,
            attr=attr,
            bin_name=bin_name,
            paths_to_show=paths_to_show,
            num_examples=num_examples
        )
    
    print(f"Done! All example visualizations for attribute '{attr}' saved to {save_path}")


def process_all_examples(
    output_dir: str,
    save_path: str,
    attrs_to_process: Optional[List[str]] = None,
    bins_to_process: Optional[Dict[str, List[str]]] = None,
    paths_to_show: List[str] = ["input", "translated", "full_cycle"],
    num_examples: int = 5
) -> None:
    """
    Process and visualize examples for all specified attributes and bins.
    
    Args:
        output_dir: Directory where binned results are saved
        save_path: Directory where visualizations will be saved
        attrs_to_process: List of attributes to process. If None, all attributes with examples are used
        bins_to_process: Dictionary mapping attributes to lists of bins to process.
                         If None, all bins for each attribute are used
        paths_to_show: List of paths to display in the visualizations
        num_examples: Number of examples to show per bin
    """
    # Load binned results
    result_data = load_binned_results(output_dir)
    if result_data is None:
        print("No results data available. Cannot process examples.")
        return
    
    # Unpack result data
    (_, _, _, _, examples_by_attr, _, analysis_attributes) = result_data
    
    # Use all attributes with examples if none specified
    if attrs_to_process is None:
        attrs_to_process = list(examples_by_attr.keys())
    
    # Process each attribute
    for attr in attrs_to_process:
        if attr not in examples_by_attr:
            print(f"No examples found for attribute '{attr}'. Skipping.")
            continue
        
        # Determine which bins to process for this attribute
        attr_bins = None
        if bins_to_process and attr in bins_to_process:
            attr_bins = bins_to_process[attr]
        
        # Process the examples for this attribute
        process_examples_for_attribute(
            output_dir=output_dir,
            save_path=save_path,
            attr=attr,
            bins_to_process=attr_bins,
            paths_to_show=paths_to_show,
            num_examples=num_examples
        )
    
    print(f"Done! All example visualizations saved to {save_path}")


# # Example usage
# if __name__ == "__main__":
#     # Example 1: Process examples for a specific attribute and bin
#     visualize_examples_grid(
#         output_dir="./experiment_results",
#         save_path="./example_visualizations",
#         attr="shape",
#         bin_name="round",
#         paths_to_show=["input", "translated", "full_cycle"],
#         num_examples=5
#     )
    
#     # Example 2: Process all bins for a specific attribute
#     process_examples_for_attribute(
#         output_dir="./experiment_results",
#         save_path="./example_visualizations",
#         attr="color",
#         paths_to_show=["input", "translated", "full_cycle"],
#         num_examples=3
#     )
    
#     # Example 3: Process all examples for all attributes
#     process_all_examples(
#         output_dir="./experiment_results",
#         save_path="./example_visualizations",
#         paths_to_show=["input", "translated", "full_cycle"],
#         num_examples=4
#     )
OUTPUT_DIR = "FINAL/High_cycle_v5/results_High_cycle_v5_sans_couleurs_seed0"
SAVE_PATH = "./Figures_test"


# Example usage
if __name__ == "__main__":
    # Example 1: Process examples for a specific attribute and bin
    visualize_examples_grid(
        output_dir=OUTPUT_DIR,
        save_path="./Figures_test",
        attr="shape",
        bin_name="round",
        paths_to_show=["input", "translated", "full_cycle"],
        num_examples=5
    )
    
    # Example 2: Process all bins for a specific attribute
    process_examples_for_attribute(
        output_dir= OUTPUT_DIR,
        save_path="./Figures_test",
        attr="color",
        paths_to_show=["input", "translated", "full_cycle"],
        num_examples=3
    )
    
    # Example 3: Process all examples for all attributes
    process_all_examples(
        output_dir= OUTPUT_DIR,
        save_path="./Figures_test",
        paths_to_show=["input", "translated", "full_cycle"],
        num_examples=4
    )

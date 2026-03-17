import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
from collections import defaultdict


def load_binned_results(output_dir, eval_regularity_args=None, max_retries=1):
    """
    Load previously saved binned results from disk.
    If not found, generate them using eval_regularity with provided arguments.
    
    Args:
        output_dir: Directory containing the binned results
        eval_regularity_args: Dictionary of arguments for eval_regularity function
        max_retries: Maximum number of retry attempts if eval_regularity fails
    """
    save_path = os.path.join(output_dir, 'binned_results.pkl')
    
    if not os.path.exists(save_path):
        print(f"No saved binned results found at {save_path}, generating them now.")
        
        if eval_regularity_args is None:
            raise ValueError("eval_regularity_args must be provided when binned results don't exist")
        
        # Try to generate results with retry logic
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to generate binned results...")
                
                from SSD_eval_regularity import eval_regularity
                eval_regularity(**eval_regularity_args)
                
                # If successful, break out of retry loop
                print("Successfully generated binned results!")
                break
                
            except Exception as e:
                load_binned_results(output_dir, eval_regularity_args, max_retries - 1)
    # Load the results
    try:
        with open(save_path, 'rb') as f:
            result_data = pickle.load(f)
        print(f"Successfully loaded binned results from {save_path}")
        return result_data
    except Exception as e:
        print(f"Error loading binned results: {str(e)}")
        raise e


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
        ('full_cycle', 'Cycle-Complet visuel', 'red')
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


def create_example_plate(
    output_dir: str,
    save_path: str,
    plate_name: str = "diverse_examples",
    num_examples: int = 10,
    figsize: Tuple[int, int] = (20, 8),
    random_seed: Optional[int] = 42,
    max_diamonds: int = 3,
    eval_regularity_args: Optional[Dict] = None
) -> None:
    """
    Crée une plaquette d'exemples diversifiés avec 4 lignes :
    - Input (départ côté images)
    - Translation (départ côté attributs) 
    - Half-cycle (reconstruction - départ côté images)
    - Full-cycle (reconstruction - départ côté images)
    
    Args:
        output_dir: Répertoire contenant les résultats binned
        save_path: Répertoire de sauvegarde
        plate_name: Nom de la plaquette
        num_examples: Nombre d'exemples par ligne
        figsize: Taille de la figure
        random_seed: Graine aléatoire pour reproductibilité
        max_diamonds: Nombre maximum de diamants autorisés (défaut: 3)
        eval_regularity_args: Arguments pour eval_regularity si génération nécessaire
    """
    # Charger les résultats
    result_data = load_binned_results(output_dir, eval_regularity_args)
    if result_data is None:
        print("Aucune donnée de résultats disponible.")
        return
    
    # Le résultat est un dictionnaire, on récupère les exemples
    if isinstance(result_data, dict) and 'examples_by_attr' in result_data:
        examples_by_attr = result_data['examples_by_attr']
    else:
        print("Format de données non reconnu")
        return
    
    # Collecter des exemples diversifiés avec limitation des diamants
    # diverse_examples = collect_diverse_examples(
    #     examples_by_attr, 
    #     num_examples_per_plate=num_examples,
    #     random_seed=random_seed,
    #     max_diamonds=max_diamonds
    # )
    
    diverse_examples = examples_by_attr['diverse_examples']
    
    create_diverse_example_plate(
        examples_by_attr=examples_by_attr,
        save_path=save_path,
        plate_name=plate_name,
        figsize=figsize
    )

from SSD_utils import binning_config_6144  # Import your binning config
from SSD_eval_arbitrary import model_versions_and_configs
# base_configs = [
#         { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": f"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" }, # Replace!
#         { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": f"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
#         { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" }, # Replace!
#         { "run_id": "colored", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" }, # Replace!
#     ]

# high_alpha_configs = [
#     { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High alpha 5 (color) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" },
#     { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) H_alpha (5) v5/checkpoints/last.ckpt" },
#     { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) H_alpha (5) v5/checkpoints/last.ckpt" },
#     { "run_id": "colored", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) H_alpha (5) v5/checkpoints/last.ckpt"},
# ]
# # corrected model configs

# high_contrastive_configs = [
#     { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) (bis) High_contrastive (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) (bis) High_contrastive (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) (bis) High_contrastive (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "colored0", "full_attr": True,"gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) High_contrastive (1) v5/checkpoints/last.ckpt"}
# ]
# high_cycle_configs = [
#     { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) High_cycles (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) (bis) High_cycles (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) High_cycles (10) v5/checkpoints/epoch=1225.ckpt" },
#     { "run_id": "colored0", "full_attr": True, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) High_cycle (10) v5/checkpoints/last.ckpt"},
# ]

high_translation_configs = [
#     { "run_id": "seed0", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) (ter) High_translation (10) v5/checkpoints/last.ckpt" },
    { "run_id": "seed1", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 1) High_Translation (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "seed2", "full_attr": False, "gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) (ter) High_translation (10) v5/checkpoints/last.ckpt" },
#     { "run_id": "colored0", "full_attr": True,"gw_checkpoint_path": "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) H_T (10) v5/checkpoints/last.ckpt"},
]




def main():
    """
    Main function with all configuration parameters.
    """
    # /home/alexis/Desktop/FINAL_AVEC_BRUIT_0.01_v4/Base_params_v5/results_Base_params_v5_sans_couleurs_seed0
    # /home/alexis/Desktop/FINAL_AVEC_BRUIT_0.01_v4/High_cycle_v5_seed0/binned_results.pkl
    for model_name, model_config in model_versions_and_configs : 
        model_checkpoint = model_config[1]['gw_checkpoint_path']
    
        # name = f"FINAL_AVEC_BRUIT_0.01_v4/{model_name}/results_{model_name}_sans_couleurs_seed1"  # Replace with your model name
        
        name = "FINAL/High_translation_v5/results_High_translation_v5_sans_couleurs_seed1"
        # Configuration directories
        OUTPUT_DIR = f"/home/alexis/Desktop/{name}"
        SAVE_PATH = f"./Figures_test/{name}/plaquettes_bruit"
        
        # Configuration parameters for eval_regularity
        # These need to be defined based on your specific setup
        checkpoint_path = model_checkpoint  # Path to the model checkpoint
    # Replace with actual path
        
        num_bins = 15  # Replace with your desired number of bins
        # dataset_csv = "/home/alexis/Desktop/evaluation_set_6144_gray/attributes.csv"
        dataset_csv = "/home/alexis/Desktop/evaluation_set_examples/attributes.csv"  # Replace with your dataset CSV path
        parrallel = True  # Replace with your parallel setting
        conditioning = True  # Replace with your conditioning setting
        barycentre = True  # Replace with your barycentre setting
        
        # Arguments for eval_regularity function
        eval_regularity_args = {
            'full_attr': False,  # Avec couleurs
            'run_id': "",
            'checkpoint_base_path': checkpoint_path,
            'encoders_size': 256,
            'n_encoders': 2,
            'n_decoders': 2,
            'name': name,
            'num_bins': num_bins,
            'dataset_csv': dataset_csv,
            'reverb_n': 1,
            'rgb': True,
            'binning_config': binning_config_6144,
            'tilting': 1, 
            'parrallel': parrallel, 
            'conditioning': conditioning,
            'barycentre': barycentre,
        }
        
        # Créer une plaquette d'exemples diversifiés avec maximum 3 diamants
        create_example_plate(
            output_dir=OUTPUT_DIR,
            save_path=SAVE_PATH,
            plate_name="exemples_varies",
            num_examples=15,
            random_seed=42,
            max_diamonds=3,
            eval_regularity_args=eval_regularity_args
        )


if __name__ == "__main__":
    main()



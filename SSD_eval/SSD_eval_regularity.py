
from pathlib import Path
import torch
from shimmer.modules.global_workspace import GlobalWorkspace2Domains

from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig
from SSD_utils import binning_config_6144, default_binning_config

from shimmer_ssd.modules.domains import load_pretrained_domains

# Additional imports for shape generation and evaluation
import os

# from SSD_bin_consistency import test_color_consistency_across_bins, test_all_attributes_color_consistency
from SSD_H_evaluation_functions import HueShapeAnalyzer


import multiprocessing
import time


def load_global_workspace(gw_checkpoint: Path, config: dict) -> GlobalWorkspace2Domains:
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config["domains"],
        config["global_workspace"]["latent_dim"],
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
    )
    global_workspace = GlobalWorkspace2Domains.load_from_checkpoint(
        gw_checkpoint,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )
    return global_workspace


# ====================
# MAIN : ARGUMENTS & EXECUTION
# ====================


def eval_regularity(
    dataset_csv="./evaluation_set/attributes.csv", 
    full_attr=True, 
    run_id=None,  
    checkpoint_base_path=None, 
    n_encoders=2, 
    n_decoders=1, 
    encoders_size=256, 
    decoders_size=256, 
    name="àtrier",
    latent_dim=12,
    debug=False,
    num_bins=50,
    reverb_n=1,
    binning_config=None,
    rgb=False,
    tilting=1,
    parrallel=False,
    barycentre = False,
    conditioning = False,
    my_hparams={"temperature": 1, "alpha": 1}, # hyperparameters for the model
):
    """
    Evaluate model regularity on shape representations.

    Args:
        dataset_csv: Path to CSV with dataset attributes
        full_attr: Boolean indicating if the attribute domain uses color info
        run_id: Optional identifier for the run (e.g., seed value)
        checkpoint_base_path: Path to the model checkpoint
        n_encoders: Number of encoder layers
        n_decoders: Number of decoder layers
        encoders_size: Hidden dimension size for encoders
        decoders_size: Hidden dimension size for decoders
        name: Name for the output directory
        latent_dim: Dimension of the latent space
        debug: Whether to run in debug mode
        num_bins: Number of bins for distribution analysis
    """
   
    
    # Set seed if run_id is provided
    seed = int(run_id[-1]) if run_id else 0

    # Determine configuration based on full_attr
    if full_attr:
        attr_domain_type = DomainModuleVariant.attr_legacy
        config_suffix = "avec_couleurs"
    else:
        attr_domain_type = DomainModuleVariant.attr_legacy_no_color
        config_suffix = "sans_couleurs"

    print("_______________________________________________________________________________")
    print(f"Evaluation du modèle {version} {'avec' if full_attr else 'sans'} couleurs dans les attributs")
    if run_id:
        print(f"Run ID: {run_id}")
    print("_______________________________________________________________________________")

    # Define output directory
    output_base = f"./test_regularity/{name}"
    run_specific_part = f"results_{config_suffix}"
    if run_id:
        run_specific_part += f"_{run_id}"

    output_dir = os.path.join(output_base, run_specific_part)
    output_dir = Path(output_dir).resolve()

    if debug:
        import time
        debug_timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results_debug/{version}_{config_suffix}{'_' + run_id if run_id else ''}_{debug_timestamp}"

    # Check if output exists
    if os.path.exists(output_dir) and not debug:
        print(f"WARNING: Le répertoire {output_dir} existe déjà. Les résultats pourraient être écrasés ou incomplets.")

    os.makedirs(output_dir, exist_ok=True)

    # Setup config and load model
    config = {
        "domains": [
            LoadedDomainConfig(
                domain_type=DomainModuleVariant.v_latents,
                checkpoint_path=Path("/mnt/HD2/alexis_data/checkpoints_backup/checkpoints/domain_v.ckpt"),
            ),
            LoadedDomainConfig(
                domain_type=attr_domain_type,
                checkpoint_path=Path("/mnt/HD2/alexis_data/checkpoints_backup/checkpoints/domain_attr.ckpt"),
                args=my_hparams,
            ),
        ],
        "global_workspace": {
            "latent_dim": latent_dim,
            "encoders": {"hidden_dim": encoders_size, "n_layers": n_encoders},
            "decoders": {"hidden_dim": decoders_size, "n_layers": n_decoders},
        },
    }
    
    gw_checkpoint = Path(checkpoint_base_path)
    if not gw_checkpoint.exists():
        print(f"ERROR: Checkpoint file not found at {gw_checkpoint}")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEBUG_MODE:
            # Dont load the model if in debug mode
            print("DEBUG_MODE is ON. Skipping model loading.")
            global_workspace = None
        else:
            global_workspace = load_global_workspace(gw_checkpoint, config)
            global_workspace.to(device) 
        print(f"Model loaded successfully from {gw_checkpoint} onto {device}.")
    except Exception as e:
        print(f"ERROR: Failed to load model from checkpoint {gw_checkpoint}. Error: {e}")
        return
    
    analyzer = HueShapeAnalyzer(
        global_workspace=global_workspace,
        device=device,
        color=full_attr,
        output_dir=output_dir,
        debug=debug,
        seed=seed, 
        num_bins=num_bins,
        reverb_n = reverb_n,
        rgb=rgb,
        barycentre=barycentre,
    )

    # Run analysis
    print(f"Starting analysis. Results will be saved in: {output_dir}")
    analysis_results = analyzer.analyze_dataset(
        csv_path=dataset_csv,
        display_examples=True, 
        binning_config=binning_config,
        im_dir=dataset_csv.replace("/attributes.csv", ''),
        tilting=tilting,
        parrallel=parrallel,
        conditioning=conditioning,
    )
    print("Analysis completed.")


DEBUG_MODE = False

# # # Define DEBUG_MODE globally or pass via args
# # DEBUG_MODE = False # Set to True for quick tests without full processing






# PREVIOUS ANALYSIS NAMES 

# sidequest1 ######  /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/gw/version_colorFalse_boa56wxb/last. ###### encoders_size = 256, n_encoders = 2, n_decoders = 1,
# sidequest2 #######
# version_2l0yrcqq #####
# sidequest2_1536_(test) 
# best_valoss_model ###### /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/gw/version_phvker4h/epoch=49.ckpt ##### encoders_size = 128, n_encoders = 1, n_decoders = 2,
# high_cycle_and_converged ###### /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/gw/version_7ig35528/epoch=49.ckpt ##### encoders_size = 256, n_encoders = 1, n_decoders = 0,
# base_params_1202 ###### /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/training_logs/Removing_colors/Base_params/Removing_colors/Base_params/version_colorTrue_None/checkpoints/epoch=1202.ckpt ##### encoders_size = 256, n_encoders = 2, n_decoders = 2,
# high_cycle_(10)_300K ###### /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/training_logs/Removing_colors/high_cycle/Removing_colors/high_cycle/version_colorTrue_None/checkpoints/last.ckpt ##### encoders_size = 256, n_encoders = 2, n_decoders = 2,
# Base_params_Final_scheduler ###### /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/training_logs/DEBUG/Final_scheduler/checkpoints/epoch=1229.ckpt ##### encoders_size = 256, n_encoders = 2, n_decoders = 2,
# ongoing_temp_05_alpha2_new_scheduler_1_reverb ###### /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/training_logs/Removing_colors/temp_05_alpha2_new_scheduler/checkpoints/last.ckpt ##### encoders_size = 256, n_encoders = 2, n_decoders = 2,
# Side_quest_2  /mnt/HD2/alexis_data/checkpoints_backup/checkpoints/training_logs/shimmer-ssd_sidequests/Side_quest_2/checkpoints/epoch=1215.ckpt


# POSSIBLE DATASETSColored_Base_

# evaluation_set/attributes.csv
# blind_evaluation_set_1536/attributes.csv
# blind_evaluation_set_192/attributes.csv
# evaluation_set_6144/attributes.csv

parrallel = False
conditioning = True
barycentre = True

# name = "TEST Séquentiel"
# full_attr = False
# checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"
num_bins = 15
# name = f"Seed0_High_cycle_{num_bins}_bins_and new traduction"
version = "High_cycle_v5"
name = "HighTranslation_v5seed2"
full_attr = False

# checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 0) High_cycles (10) v5/checkpoints/last.ckpt"
checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Seed 2) (ter) High_translation (10) v5/checkpoints/last.ckpt"
dataset_csv = "/home/alexis/Desktop/evaluation_set_examples/attributes.csv"


if __name__ == "__main__":
    # checkpoint_base_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) H_T (10) v5/checkpoints/last-v2.ckpt"
     # my_hparams = {"temperature": 0.7, "alpha": 1.3}
    # debug_base_params = "/mnt/HD2/alexis_data/checkpoints_backup/checkpoints/training_logs/Removing_colors/Basem_modified_cycle/checkpoints/epoch=716.ckpt"
    # checkpoint_base_path = debug_base_params
    eval_regularity(
        full_attr=full_attr,  # Avec couleurs
        run_id="High_cycle0",
        checkpoint_base_path=checkpoint_path,
        encoders_size=256,
        n_encoders=2,
        n_decoders=2,
        name=f"DEBUG_EXAMPLES/{name}",
        num_bins=num_bins,
        dataset_csv=dataset_csv,
        reverb_n=1,
        rgb=True,
        binning_config=binning_config_6144,
        tilting=1, 
        parrallel=parrallel, 
        conditioning=conditioning,
        barycentre=barycentre,
    )

# name = "High_cycle_corrected_v5"
# checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High_cycle - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"


# name = "High_alpha_corrected_v5"
# checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High alpha 5 (color) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt" #Truly no color

# name = "High_contrastive_v5"
# checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"
# full_attr = False
# checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"
# full_attr = True


# name = "High_translation_v5"
#checkpoint_path = /home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High translation (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"


# def run_10_bins():
#     """Lance l'évaluation avec couleurs"""
#     print("=== DÉMARRAGE DE L'ÉVALUATION 6144 ===")
#     # Importation du module contenant eval_regularity
#     eval_regularity(
#         full_attr=full_attr,  # Avec couleurs
#         run_id="sweep0",
#         checkpoint_base_path=checkpoint_path,
#         encoders_size=256,
#         n_encoders=2,
#         n_decoders=2,
#         name=f"6144_gray/{name}_6144_gray",
#         num_bins=10,
#         dataset_csv="/home/alexis/Desktop/evaluation_set_6144_gray/attributes.csv",
#         reverb_n=1,
#         rgb=True,
#         binning_config=binning_config_6144,
#         tilting=1, 
#         parrallel=parrallel, 
#         conditioning=conditioning,
#         barycentre=barycentre,
#     )
#     print("=== FIN DE L'ÉVALUATION AVEC COULEURS ===")

# def run_50_bins():
#     """Lance l'évaluation sans couleurs"""
#     print("=== DÉMARRAGE DE L'ÉVALUATION 19200 ===")
    
#     eval_regularity(
#         full_attr=full_attr,  # Sans couleurs
#         run_id="sweep0",
#         checkpoint_base_path=checkpoint_path,
#         encoders_size=256,
#         n_encoders=2,
#         n_decoders=2,
#         name=f"19200/{name}",
#         num_bins=50,
#         dataset_csv="/home/alexis/Desktop/evaluation_set/attributes.csv",
#         reverb_n=1,
#         rgb=True,
#         # binning_config=binning_config_6144,
#         tilting=1, 
#         parrallel=parrallel, 
#         conditioning=conditioning,
#         barycentre=barycentre,
#     )
#     print("=== FIN DE L'ÉVALUATION SANS COULEURS ===")

# if __name__ == "__main__":
#     multiprocessing.set_start_method('spawn', force=True)
#     #     print("Méthode de démarrage multiprocessing définie sur 'spawn'.")
#     # Création des processus
#     process_avec_couleurs = multiprocessing.Process(target=run_10_bins)
#     process_sans_couleurs = multiprocessing.Process(target=run_50_bins)
    
#     # Démarrage des processus en parallèle
#     start_time = time.time()
#     process_avec_couleurs.start()
#     process_sans_couleurs.start()
    
#     # Attente de la fin des processus
#     process_avec_couleurs.join()
#     process_sans_couleurs.join()
    
#     # Calcul du temps d'exécution total
#     duration = time.time() - start_time
#     print(f"Exécution terminée en {duration:.2f} secondes")
    
#     # Vérification des résultats
#     output_dir_avec = Path(f"./test_regularity/{name}_6144_gray/results_sans_couleurs_sweep0/within_model_comparison_summary.json")
#     output_dir_sans = Path(f"./test_regularity/{name}/results_sans_couleurs_sweep0/within_model_comparison_summary.json")
    
#     print("\nRapport final:")
#     print(f"Résultats avec 6144: {'Disponibles' if output_dir_avec.exists() else 'Non disponibles'}")
#     print(f"Résultats 19200: {'Disponibles' if output_dir_sans.exists() else 'Non disponibles'}")


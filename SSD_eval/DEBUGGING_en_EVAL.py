from pathlib import Path
import time
import multiprocessing
from SSD_eval_regularity import eval_regularity
from SSD_utils import binning_config_6144, default_binning_config

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


# POSSIBLE DATASETS

#dataset_csv = evaluation_set/attributes.csv
#dataset_csv = blind_evaluation_set_1536/attributes.csv
#dataset_csv = blind_evaluation_set_192/attributes.csv
dataset_csv = "evaluation_set_6144/attributes.csv"

# dataset_csv = "evaluation_set_614_400_original/attributes.csv"
RUNNING_IN_SEQUENTIAL = True
parrallel = True
barycentre = True
conditioning = True

name = "DEBUG_H_comparisonpng"
full_attr = False
checkpoint_path = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"



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


def run_10_bins():
    """Lance l'évaluation avec couleurs"""
    print("=== DÉMARRAGE DE L'ÉVALUATION 6144 ===")
    # Importation du module contenant eval_regularity
    eval_regularity(
        full_attr=full_attr,  # Avec couleurs
        run_id="sweep0",
        checkpoint_base_path=checkpoint_path,
        encoders_size=256,
        n_encoders=2,
        n_decoders=2,
        name=f"{name}_10",
        num_bins=10,
        dataset_csv=dataset_csv,
        reverb_n=1,
        rgb=True,
        binning_config=binning_config_6144,
        tilting=1, 
        parrallel=parrallel, 
        conditioning=conditioning,
        barycentre=barycentre,
    )
    
    print("=== FIN DE L'ÉVALUATION AVEC COULEURS ===")

def run_50_bins():
    """Lance l'évaluation sans couleurs"""
    print("=== DÉMARRAGE DE L'ÉVALUATION 19200 ===")
    
    eval_regularity(
        full_attr=full_attr,  # Sans couleurs
        run_id="sweep0",
        checkpoint_base_path=checkpoint_path,
        encoders_size=256,
        n_encoders=2,
        n_decoders=2,
        name=f"{name}_50",
        num_bins=50,
        dataset_csv=dataset_csv,
        reverb_n=1,
        rgb=True,
        binning_config=binning_config_6144,
        tilting=1, 
        parrallel=parrallel, 
        conditioning=conditioning,
        barycentre=barycentre,
    )
    print("=== FIN DE L'ÉVALUATION SANS COULEURS ===")

if __name__ == "__main__":
    print("Barycentre is", barycentre) 
    print("Parrallel is", parrallel)
    print("conditioning is", conditioning)
    # multiprocessing.set_start_method('spawn', force=True)
    print("running in sequential ?")
    #     print("Méthode de démarrage multiprocessing définie sur 'spawn'.")
    if RUNNING_IN_SEQUENTIAL:
        start_time = time.time()
        print("running in sequential")
        run_10_bins()
        # run_50_bins()
        duration = time.time() - start_time
        print(f"Exécution terminée en {duration:.2f} secondes")
    else:
        print("running in parallel")
        # Vérification de la compatibilité avec le système d'exploitation

        # Création d'un pool de processus
        multiprocessing.set_start_method('spawn', force=True)

        # Création des processus
        process_avec_couleurs = multiprocessing.Process(target=run_10_bins)
        process_sans_couleurs = multiprocessing.Process(target=run_50_bins)
        
        # Démarrage des processus en parallèle
        start_time = time.time()
        process_avec_couleurs.start()
        process_sans_couleurs.start()
        
        # Attente de la fin des processus
        process_avec_couleurs.join()
        process_sans_couleurs.join()
        
        # Calcul du temps d'exécution total
        duration = time.time() - start_time
        print(f"Exécution terminée en {duration:.2f} secondes")
    
    # Vérification des résultats
    output_dir_avec = Path(f"./test_regularity/6144_gray/{name}_6144_gray/results_sans_couleurs_sweep0/within_model_comparison_summary.json")
    output_dir_sans = Path(f"./test_regularity/19200/{name}/results_sans_couleurs_sweep0/within_model_comparison_summary.json")
    
    print("\nRapport final:")
    print(f"Résultats avec 6144: {'Disponibles' if output_dir_avec.exists() else 'Non disponibles'}")
    print(f"Résultats 19200: {'Disponibles' if output_dir_sans.exists() else 'Non disponibles'}")


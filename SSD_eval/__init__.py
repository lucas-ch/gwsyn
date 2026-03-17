# # # Je dois faire les tests suivants : 

# # Pour étudier cela, nous nous intéresserons aux variations de distributions dans les couleurs après passage
# # dans le modèle. Plus spécifiquement, les distributions étudiées seront celles obtenues après :
# # • Traduction attribut-image
# # • Demi-cycle image
# # • Cycle complet image
# # Les critères que nous fixons sont les suivants :
# # 1. La régularité des associations sera évaluée en testant si les distributions des couleurs, diffèrent entre
# # formes (Kolmogorov-Smirnov, p < 0.05).
# # 2. L’arbitraire sera évalué en s’assurant que parmi 10 entraînements différents, au moins deux dis-
# # tributions pour une forme donnée diffèrent significativement (10
# # 2
# #  comparaisons par forme, tests de
# # Kolmogorov-Smirnov, p < 0.05)


# # Donc : 2 types de tests pour trois types de reconstruction



# # Pour le moment je travaille sur la traduction attributs image, je cherche à vérfier que les tests de KS et la divergence de KL sont bien implémentés. 

# # Objectif, obtenir une fonction qui prend en entrée un modèle et le dataset, et lui fait faire tout les tests possibles

# # étape 1 : Vérfier la cohérence de KS et KL, 
# # étape 2 : transposer.


# from collections.abc import Mapping, Sequence
# from pathlib import Path
# from typing import Any, cast

# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from lightning.pytorch import Callback, Trainer, seed_everything
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
# from shimmer import DomainModule, LossOutput
# from shimmer.modules.domain import DomainModule
# from shimmer.modules.global_workspace import GlobalWorkspace2Domains, SchedulerArgs
# from shimmer.modules.vae import (
#     VAE,
#     VAEDecoder,
#     VAEEncoder,
#     gaussian_nll,
#     kl_divergence_loss,
# )
# from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
# from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
# from shimmer_ssd.dataset.pre_process import TokenizeCaptions
# from shimmer_ssd.logging import (
#     LogAttributesCallback,
#     LogGWImagesCallback,
#     LogVisualCallback,
#     batch_to_device,
# )
# from shimmer_ssd.modules.domains import load_pretrained_domains
# from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
# from shimmer_ssd.modules.vae import RAEDecoder, RAEEncoder
# from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
# from torch import nn
# from torch.nn.functional import mse_loss
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.optimizer import Optimizer
# from torchvision.utils import make_grid

# from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
# from simple_shapes_dataset.cli import generate_image, get_transformed_coordinates

# # Additional imports for shape generation and evaluation
# import os
# import numpy as np
# import cv2
# import matplotlib.path as mpath
# from matplotlib import patches
# import csv
# import argparse
# import math
# import io
# import ast
# from PIL import Image
# import pandas as pd
# from scipy.stats import ks_2samp

# from SSD_evaluation_functions import evaluate_dataset
# from SSD_bin_consistency import test_color_consistency_across_bins, test_all_attributes_color_consistency








# DICTIONNAIRES DES MODELES


Base_params_v5 = {"seed0" :"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed1" : "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed2":"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"
}
colored_BP = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) Base_params - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"


High_cycle_corrected_v5 = {"seed0" :"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High_cycle - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed1" : "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 1) High Cycle (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed2":"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed 2) High Cycle (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"
}
colored_Hcy = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(colored) High cycle (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"



High_alpha_corrected_v5 = {"seed0" :"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High alpha 5 (color) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed1" : "/last.ckpt", 
"seed2":"/last.ckpt"
}



High_translation_v5 = {"seed0" :"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High translation (10) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed1" : "/last.ckpt", 
"seed2":"/last.ckpt"
}


High_contrastive_v5 = {"seed0" :"/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed1" : "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(seed1) HCo (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt", 
"seed2":"/last.ckpt"
}

colored_HCo = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/(Colored) High contrastive (1) - corrected v5 (Logsoftmax --> exp --> log)/checkpoints/last.ckpt"

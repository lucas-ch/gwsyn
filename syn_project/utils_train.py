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

def get_project_root():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

def save_training_params_pickle(config, project_name, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    root_dir = Path.cwd()
    log_dir = root_dir / "checkpoints" / project_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = log_dir / f"config_{timestamp}.pkl"

    with open(file_path, 'wb') as f:
        pickle.dump(config, f)
    
    return file_path

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


def get_experiment_name(condition, data, switch_epoch):
    experiment_name = f"{condition}_{data}"

    if switch_epoch > 0:
        experiment_name = f"{experiment_name}_switch_{switch_epoch}"

    return experiment_name











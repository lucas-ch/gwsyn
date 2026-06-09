import io
import math
from typing import Mapping, cast
import numpy as np
from shimmer import GlobalWorkspace2Domains
import torch
from PIL import Image
from torch.nn.functional import one_hot
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
import torch.nn.functional as F
import pandas as pd
from .utils_train import *

from simple_shapes_dataset.cli import generate_image

import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import json
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import warnings

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
 
from syn_project.utils_train import *
from syn_project.utils_color_analysis import *
from syn_project.utils_notebook import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


CAT2IDX = {"Diamond": 0, "Egg": 1, "Triangle": 2}

def split_softmax_category_attributes(concat_tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Sépare un tenseur (N, 8) en deux tenseurs :
    - Les 3 premiers (logits -> probabilités 0-1)
    - Les 5 derniers (inchangés)
    """
    # 1. Extraction des 3 premières colonnes (logits)
    logits = concat_tensor[:, :3]
    
    # 2. Conversion en probabilités (entre 0 et 1)
    # On utilise softmax sur la dimension 1 pour que la somme = 1
    # Si les 3 attributs sont indépendants, utilise torch.sigmoid(logits) à la place
    probs = F.softmax(logits, dim=1)
    
    # 3. Extraction des 5 colonnes restantes
    rest = concat_tensor[:, 3:]
    
    # Retourne une liste de deux tenseurs comme dans ton premier exemple
    return [probs, rest]

def split_binary_category_attributes(concat_tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Sépare un tenseur (N, 8) en deux tenseurs :
    - Les 3 premiers (logits -> convertis en 0 ou 1 exclusifs)
    - Les 5 derniers (inchangés)
    """
    # 1. Extraction des 3 premières colonnes (logits)
    logits = concat_tensor[:, :3]
    
    # 2. Trouver l'indice de la classe dominante (ex: index 1)
    predicted_indices = torch.argmax(logits, dim=1)
    
    # 3. Convertir en vecteur One-Hot (ex: [0, 1, 0])
    # num_classes=3 garantit qu'on a bien 3 colonnes en sortie
    binary_preds = F.one_hot(predicted_indices, num_classes=3).float()
    
    # 4. Extraction des 5 colonnes restantes
    rest = concat_tensor[:, 3:]
    
    return [binary_preds, rest]

@contextmanager
def total_silence():
    # 1. On bloque les Warnings Python
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 2. On bloque les sorties console (Prints et Warnings système)
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                
                # 3. On bloque l'affichage Matplotlib
                plt.ioff()
                try:
                    yield
                finally:
                    plt.ion()


def get_n_per_category(source_tensor, current_attr, n_per_cat=10):
    """
    Retourne les n premiers éléments de source_tensor pour chaque catégorie 
    définie dans current_attr.
    
    Args:
        source_tensor: Le tenseur d'où on extrait les données (ex: des images ou latents)
        current_attr: Le tenseur de catégories (one-hot, ex: [Batch, 3])
        n_per_cat: Nombre d'éléments à extraire par catégorie
    """
    # 1. Convertir le one-hot en indices (0, 1, 2)
    labels = current_attr.argmax(dim=-1)
    
    # 2. Trouver les indices pour chaque classe
    indices_list = []
    num_classes = current_attr.shape[-1]
    
    for i in range(num_classes):
        # On récupère les indices où la classe est i
        indices = (labels == i).nonzero(as_tuple=True)[0]
        
        if len(indices) < n_per_cat:
            print(f"Warning: Seulement {len(indices)} éléments trouvés pour la classe {i}")
            indices_list.append(indices)
        else:
            # On ne garde que les n premiers
            indices_list.append(indices[:n_per_cat])
    
    # 3. Concaténer tous les indices sélectionnés
    all_indices = torch.cat(indices_list)
    
    # 4. Retourner le sous-ensemble du tenseur source
    return source_tensor[all_indices]

def get_colors_labels_per_condition(condition: str, data = "biased_00", settings={"epoch": 0, "n": 100, "split":"test" }) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge une condition expérimentale et retourne (colors_np, labels).
 
    Parameters
    ----------
    condition : str, identifiant de la condition
 
    Returns
    -------
    colors_np_1 : np.ndarray (n, 3) RGB 0-255
    colors_np_2 : np.ndarray (n, 3) RGB 0-255
    labels    : np.ndarray (n,)   entiers de catégorie
    """
    experiment_name = get_experiment_name(condition, data, 0)
    modules         = get_setup_modules('syn', experiment_name)
    global_workspace = get_global_workspace(
        "syn", experiment_name, epoch=settings["epoch"], modules=modules
    )
    data_module  = get_data_module("syn", experiment_name, modules=modules)
    test_samples = get_data_samples(data_module, settings["n"], split=settings["split"])
 
    key = frozenset({'color', 'v_latents', 'attr'})
    original_attr      = test_samples[key]['attr']
    original_v_latents = test_samples[key]['v_latents']
    labels = original_attr[0].argmax(dim=1).detach().cpu().numpy()
 
    gw_mod        = global_workspace.gw_mod
    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])
    x0 = global_workspace.encode_domains(test_samples)
 
 
    # v_latents → attr
    x0_v = x0[frozenset({'v_latents'})]
    g0_v         = gw_mod.encode(x0_v)['v_latents']
    x1    = gw_mod.decode(g0_v, domains={'attr'})
    x1['attr'] = split_binary_category_attributes(x1['attr'])
    t = global_workspace.encode_domains({frozenset({'attr'}): x1})
 
    # attr → color
    x1_attr = t[frozenset({'attr'})]
    g1_attr = gw_mod.encode(x1_attr)['attr']
    x2    = gw_mod.decode(g1_attr, domains={'color', 'v_latents'})
    colors_x2 = x2['color'].detach().cpu().numpy()
 
    # color → v_latents (fusion)
    g2_color      = gw_mod.encode(x2)['color']
    fusion      = 0.5 * g2_color + 0.5 * g1_attr
    t_fused     = gw_mod.decode(fusion, domains={'v_latents'})['v_latents']
    decoded_images = visual_module.decode_images(t_fused)
 
    original_images_rgb = visual_module.decode_images(original_v_latents)
    d          = {"train_images": original_images_rgb, "images_decoded": decoded_images}
    colors_xvision4  = get_samples_rgb(d, "decoded_edge")
 
    del original_images_rgb, decoded_images
    torch.cuda.empty_cache()
 
    return colors_x2, colors_xvision4, labels


def run_conditions(
    conditions: list,
    cat_names: dict = None,
    value: float = 0.85,
    saturation_boost: float = 1.4,
    plot: bool = True,
    plots_per_row: int = 3,
) -> pd.DataFrame:
    """
    Itère sur une liste de conditions, calcule F et LDA pour chacune.

    Parameters
    ----------
    conditions    : list de valeurs passées une à une à load_fn
    load_fn       : callable(condition) → (colors_np, labels)
                    Doit retourner un tuple (array RGB, array labels).
    cat_names     : dict de noms de catégories (partagé entre conditions)
    value         : luminosité pour l'affichage
    saturation_boost : boost de saturation pour l'affichage
    plot          : bool, afficher les histogrammes
    plots_per_row : int, nombre de graphes par ligne

    Returns
    -------
    pd.DataFrame avec colonnes : condition, F, p, lda
    """
    results = []

    if plot:
        n = len(conditions)
        ncols = min(plots_per_row, n)
        nrows = (n + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(6 * ncols, 4 * nrows),
                                squeeze=False)

    for idx, condition in enumerate(conditions):
        print(f"\n── Condition : {condition} ──")
        _, colors_np, labels = get_colors_labels_per_condition(condition)

        if plot:
            row, col = divmod(idx, plots_per_row)
            ax = axs[row][col]
            metrics = hue_analysis(
                colors_np, labels,
                cat_names=cat_names,
                value=value,
                saturation_boost=saturation_boost,
                title=str(condition),
                ax=ax,
            )
        else:
            metrics = compute_hue_metrics(colors_np, labels)
            print(f"LDA = {metrics['lda_score']:.2%}")

        results.append({'condition': condition, **metrics})

    if plot:
        # Masquer les axes vides si le nombre de conditions ne remplit pas la grille
        for idx in range(len(conditions), nrows * plots_per_row):
            row, col = divmod(idx, plots_per_row)
            axs[row][col].set_visible(False)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).set_index('condition')

from typing import NamedTuple


class ModuleDataOutputs(NamedTuple):
    global_workspace: torch.Tensor
    domain_mods: list
    gw_mod: torch.nn.Module
    visual_module: torch.nn.Module
    original_colors: torch.Tensor
    original_attr: torch.Tensor
    original_v_latents: torch.Tensor
    cat: int
    latent_domains: dict
    
def get_modules_data_from_exp(experiment_name, n_samples_test=100, split='test', checkpoint_epoch=0):

    modules = get_setup_modules('syn', experiment_name)
    global_workspace = get_global_workspace("syn", experiment_name, epoch=checkpoint_epoch, modules=modules)
    data_module = get_data_module("syn",  experiment_name, modules=modules)

    test_samples = get_data_samples(data_module, n_samples_test, split=split)
    original_colors = test_samples[frozenset({'color', 'attr', 'v_latents'})]['color']
    original_attr = test_samples[frozenset({'color', 'attr', 'v_latents'})]['attr']
    original_v_latents = test_samples[frozenset({'color', 'attr', 'v_latents'})]['v_latents']
    cat = original_attr[0]

    domain_mods = global_workspace.domain_mods
    gw_mod = global_workspace.gw_mod
    latent_domains = global_workspace.encode_domains(test_samples)

    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])

    return ModuleDataOutputs(
        global_workspace=global_workspace,
        domain_mods=domain_mods,
        gw_mod=gw_mod,
        visual_module=visual_module,
        original_colors=original_colors,
        original_attr=original_attr,
        original_v_latents=original_v_latents,
        cat=cat,
        latent_domains=latent_domains
    )   


def get_objects_from_v_latents(latent_domains, gw_mod, global_workspace):
    latents_source_v_latents = latent_domains[frozenset({'v_latents'})]

    # à partir des latents vision, j'encode les représentations dans le gw
    z_input = gw_mod.encode(latents_source_v_latents)['v_latents']

    # je décode les attributs
    spatial = gw_mod.decode(z_input, domains={'attr'})

    # je les reformatte en attributs avec cat/autres séparés, puis je les re-encode dans la modalité attr correctement
    spatial['attr'] = split_binary_category_attributes(spatial['attr'])
    spatial = {frozenset({'attr'}): spatial}
    spatial = global_workspace.encode_domains(spatial)

    # à partir des attributs dérivés des v_latents, j'encode les représentation dans le gw
    latents_source_attr_2 = spatial[frozenset({'attr'})]
    z_spatial = gw_mod.encode(latents_source_attr_2)['attr']

    color1 = gw_mod.decode(z_input, domains={'color'})
    z_couleur = gw_mod.encode(color1)['color']

    # je décode couleur et v_latents. si tout va bien les v_latents sont la version grisés de l'image, la couleur correspond à la catégorie
    t = gw_mod.decode(z_spatial, domains={'color', 'v_latents'})
    color2 = t['color']
    vision1  = t['v_latents']

    z_vision_2 = 0.5*z_spatial + 0.5*z_couleur
    vision2 = gw_mod.decode(z_vision_2, domains={'v_latents'})['v_latents']

    # j'encode la couleur obtenue et je la décode vers v_latents: je devrais avoir des patch colorés
    z_syn = gw_mod.encode({'color': color2})['color']
    t = gw_mod.decode(z_syn, domains={'v_latents'})
    vision3 = t['v_latents']

    # fusion des décodages vers v_latents: v => a => et v=> a => c => v
    z_syn_fusion = 0.5*z_spatial + 0.5*z_syn
    t = gw_mod.decode(z_syn_fusion, domains={'v_latents'})
    vision4 = t['v_latents']

    return {
        'z_spatial': z_spatial,
        'z_couleur': z_couleur,
        'z_vision_2': z_vision_2,
        'z_syn': z_syn,
        'z_syn_fusion': z_syn_fusion,
        'vision1': vision1,
        'vision2': vision2,
        'vision3': vision3,
        'vision4': vision4,
    }

def get_objects_from_v_imagination(latent_domains, gw_mod, global_workspace):
    latents_source = latent_domains[frozenset({'v_latents', 'attr', 'color'})]

    # à partir des latents vision, j'encode les représentations dans le gw
    z_spatial = gw_mod.encode(latents_source)['attr']
    z_couleur = gw_mod.encode(latents_source)['color']
    vision1 = gw_mod.decode(z_spatial, domains={'v_latents'})['v_latents']

    z_vision_2 = 0.5*z_spatial + 0.5*z_couleur
    vision2 = gw_mod.decode(z_vision_2, domains={'v_latents'})['v_latents']

    #gw syn
    color2 = gw_mod.decode(z_spatial, domains={'color'})
    z_syn = gw_mod.encode(color2)['color']
    z_syn_fusion = 0.5*(z_syn + z_spatial)

    vision3 = gw_mod.decode(z_syn, domains={'v_latents'})['v_latents']
    vision4 = gw_mod.decode(z_syn_fusion, domains={'v_latents'})['v_latents']


    return {
        'z_spatial': z_spatial,
        'z_couleur': z_couleur,
        'z_vision_2': z_vision_2,
        'z_syn': z_syn,
        'z_syn_fusion': z_syn_fusion,
        'vision1': vision1,
        'vision2': vision2,
        'vision3': vision3,
        'vision4': vision4,
    }

def compute_dataset_stats(dataset):
    attr = dataset['attr'][0]    # (n, 3) one-hot
    color = dataset['color']  # (n, 3) RGB [0,1]
    n = attr.shape[0]

    # Décoder les classes depuis le one-hot
    classes = attr.argmax(dim=1)  # (n,)

    # Noms pour l'affichage
    class_names = {0: 'diamant', 1: 'oeuf', 2: 'triangle'}

    print("=== Fréquences des classes ===")
    for cls in range(3):
        mask = classes == cls
        print(f"  Classe {cls} ({class_names[cls]}): {mask.sum().item()/n:.2%}")

    # Détecter les couleurs fixes par proximité RGB
    # Rouge  ~ (255, 0, 0)   → (1.0, 0.0, 0.0)
    # Vert   ~ (0, 255, 0)   → (0.0, 1.0, 0.0)  [HLS hue=60 → à ajuster selon tes fixed_colors]
    # Bleu   ~ (0, 0, 255)   → (0.0, 0.0, 1.0)
    device = color.device
    fixed_colors_rgb = {
        'rouge': torch.tensor([1.0, 0.0, 0.0]).to(device),
        'vert':  torch.tensor([0.0, 1.0, 0.0]).to(device),
        'bleu':  torch.tensor([0.0, 0.0, 1.0]).to(device),
    }
    threshold = 0.15  # distance L2 max pour considérer une couleur comme "fixe"

    color_masks = {}
    print("\n=== Fréquences des couleurs fixes ===")
    for color_name, ref in fixed_colors_rgb.items():
        dist = (color - ref).norm(dim=1)          # distance L2 pour chaque sample
        mask = dist < threshold
        color_masks[color_name] = mask
        print(f"  {color_name}: {mask.sum().item()/n:.2%}")

    print("\n=== Matrice conjointe P(classe, couleur) — couleurs fixes uniquement ===")
    print(f"  {'':12}", end="")
    for color_name in fixed_colors_rgb:
        print(f"  {color_name:8}", end="")
    print()

    for cls in range(3):
        print(f"  {class_names[cls]:12}", end="")
        for color_name, color_mask in color_masks.items():
            joint = ((classes == cls) & color_mask).sum().item()
            print(f"  {joint/n:8.3f}", end="")
        print()

    print("\n=== P(couleur | classe) — parmi les couleurs fixes uniquement ===")
    for cls in range(3):
        cls_mask = classes == cls
        n_fixed_in_cls = sum(
            (cls_mask & cmask).sum().item() for cmask in color_masks.values()
        )
        print(f"  {class_names[cls]:12}", end="")
        for color_name, color_mask in color_masks.items():
            joint = ((cls_mask & color_mask)).sum().item()
            p = joint / n_fixed_in_cls if n_fixed_in_cls > 0 else 0
            print(f"  {color_name}:{p:.2%}", end="")
        print(f"  (n_fixed={n_fixed_in_cls})")
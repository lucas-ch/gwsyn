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

def get_image_from_interactive_attr(cat, x, y, size, rot, color_r, color_g, color_b):
    fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
    # The dataset generatoion tool has function to generate a matplotlib shape
    # from the attributes. 
    generate_image(
        ax,
        CAT2IDX[cat],
        [int(x * 18 + 7), int(y * 18 + 7)],
        size * 7 + 7,
        rot * 2 * math.pi,
        np.array([color_r * 255, color_g * 255, color_b * 255]),
        imsize=32,
    )
    ax.set_facecolor("black")
    plt.tight_layout(pad=0)
    # Return this as a PIL Image.
    # This is to have the same dpi as saved images
    # otherwise matplotlib will render this in very high quality
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image

def get_decoded_image_from_interactive_attr(cat, x, y, size, rot, color_r, color_g, color_b, training_params, device, global_workspace):
    exclude_colors = training_params["exclude_colors"]
    category = one_hot(torch.tensor([CAT2IDX[cat]]), 3)
    rotx = math.cos(rot * 2 * math.pi)
    roty = math.sin(rot * 2 * math.pi)
    
    attributes = torch.tensor(
        [[x * 2 - 1, y * 2 - 1, size * 2 - 1, rotx, roty, ]]
    )

    if not exclude_colors:
        attributes = torch.tensor(
            [[x * 2 - 1, y * 2 - 1, size * 2 - 1, rotx, roty, color_r * 2 - 1, color_g * 2 - 1, color_b * 2 - 1]]
        )

    samples = [category.to(device), attributes.to(device)]
    attr_gw_latent = global_workspace.gw_mod.encode({"attr": global_workspace.encode_domain(samples, "attr")})
    gw_latent = global_workspace.gw_mod.fuse(
        attr_gw_latent, {"attr": torch.ones(attr_gw_latent["attr"].size(0)).to(device)}
    )
    decoded_latents = global_workspace.gw_mod.decode(attr_gw_latent['attr'])["v_latents"]
    decoded_images = (
        global_workspace.domain_mods["v_latents"]
        .decode_images(decoded_latents)[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    return decoded_images

def get_decoded_images_from_gw_latent(
        gw_latent:Mapping[frozenset[str], torch.Tensor],
        global_workspace:GlobalWorkspace2Domains):
    gw_latents_decoded = global_workspace.decode(gw_latent, ["v_latents", "attr"])
    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])
    images = visual_module.decode_images(gw_latents_decoded["v_latents"]).detach().cpu()

    return images

def plot_interactive(image, decoded_image):
    fig, axes = plt.subplots(1, 2)
    axes[0].set_facecolor("black")
    axes[0].set_title("Original image from attributes")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].imshow(image)

    # normalize the attribute for the global workspace.
    axes[1].imshow(decoded_image)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Translated image through GW")
    plt.show()

import matplotlib.patches as patches

import matplotlib.patches as patches
import math

def plot_img_comparison_triple_stripes(img_tensor_one: torch.Tensor, img_tensor_two: torch.Tensor, n_samples: int = 25):
    original_np = img_tensor_one.permute(0, 2, 3, 1).detach().cpu().numpy()
    decoded_np = img_tensor_two.permute(0, 2, 3, 1).detach().cpu().numpy()

    n_samples = min(n_samples, len(original_np))
    n_stripes = 5
    rows = math.ceil(n_samples / n_stripes)

    # 1. Création de la figure
    fig, axes = plt.subplots(rows, n_stripes * 2, figsize=(14, 1.4 * rows), facecolor='white')
    
    # Force le layout pour fixer les positions avant de dessiner les bandes
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    for s in range(n_stripes):
        col_left = s * 2
        col_right = s * 2 + 1
        
        # --- ASTUCE POUR LA LARGEUR DES BANDES ---
        # On récupère les positions relatives des colonnes
        # On calcule le milieu entre les stripes pour que les bandes se touchent presque
        x_min = axes[0, col_left].get_position().x0 - 0.02
        x_max = axes[0, col_right].get_position().x1 + 0.02
        
        if s % 2 == 0:
            # On dessine un rectangle qui prend TOUTE la hauteur de la figure (0 à 1)
            # mais seulement la largeur de la stripe (x_min à x_max)
            rect = patches.Rectangle((x_min, 0), x_max - x_min, 1, 
                                     transform=fig.transFigure, 
                                     facecolor='#F2F2F2', zorder=-1)
            fig.patches.append(rect)

        for r in range(rows):
            idx = r + (s * rows)
            ax_orig = axes[r, col_left]
            ax_dec = axes[r, col_right]

            if idx < n_samples:
                ax_orig.imshow(original_np[idx], cmap='gray')
                ax_dec.imshow(decoded_np[idx], cmap='gray')
                
                if r == 0:
                    ax_orig.set_title("Orig", fontsize=9, fontweight='bold')
                    ax_dec.set_title("Dec", fontsize=9, fontweight='bold')
            
            ax_orig.axis('off')
            ax_dec.axis('off')

    plt.close()
    return fig

def plot_img_comparison(idx, img_tensor_one: torch.Tensor, image_tensor_two: torch.Tensor):
    original_images_np = img_tensor_one.permute(0, 2, 3, 1).detach().cpu().numpy()
    decoded_images_np = image_tensor_two.permute(0, 2, 3, 1).detach().cpu().numpy()

    # Création d'une figure avec 1 ligne et 2 colonnes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Affichage de la prédiction
    axes[0].imshow(original_images_np[idx])
    axes[0].set_title(f"original (idx {idx})")
    axes[0].axis('off') # Optionnel : enlève les axes gradués

    # Affichage de la cible
    axes[1].imshow(decoded_images_np[idx])
    axes[1].set_title(f"decoded (idx {idx})")
    axes[1].axis('off')

    fig = plt.tight_layout() # Ajuste l'espacement entre les images
    return fig

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

def analyze_attribute_drift(original: torch.Tensor, reconstructed: torch.Tensor):
    """
    Calculates the average absolute difference (drift) between original and 
    reconstructed attributes for each column.
    """
    # 1. Calcul de l'erreur absolue par élément
    absolute_errors = torch.abs(original - reconstructed)
    
    # 2. Moyenne par colonne (dim=0)
    mean_errors = torch.mean(absolute_errors, dim=0)
    
    # 3. Conversion en pourcentages
    drift_percentages = mean_errors * 100
    
    # Affichage sous forme de tableau pour plus de clarté
    df_metrics = pd.DataFrame({
        'Attribute_Index': range(len(drift_percentages)),
        'Mean_Absolute_Error': mean_errors.cpu().numpy(),
        'Drift_Percentage': [f"{p:.2f}%" for p in drift_percentages.cpu().numpy()]
    })
    
    return df_metrics

def get_attr_orig_reconstr(global_workspace, samples):

    # on regarde l'ecart entre les attributs originaux et les attributs encodés puis décodés
    attribut = samples[frozenset(["v_latents", "attr"])]["attr"]
    original_attributes = torch.cat((attribut[0], attribut[1]), dim=1)

    unimodal_latents = global_workspace.encode_domains(samples)
    gw_latents = global_workspace.encode(unimodal_latents)
    decoded_attr = global_workspace.decode(gw_latents[frozenset({'attr','v_latents'})])['attr']['attr'].detach()
    reconstructed_attr = split_softmax_category_attributes(decoded_attr)
    reconstructed_attr_col = torch.cat((reconstructed_attr[0], reconstructed_attr[1]), dim=1)

    return original_attributes, reconstructed_attr_col

def get_correlation_stats(colors, categories_indices):
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)

    X = colors 
    y = categories_indices

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

def get_constrat_stats(colors, categories_indices):
    n = len(categories_indices)
    contrast = {}
    c_sum = 0
    for i in range(3):
        h_stat, p_val = kruskal_colors(colors, i, categories_indices)
        eps_sq = h_stat / ((n**2 - 1) / (n + 1))
        contrast[i] = {"eps_sq": eps_sq, "p_value": p_val}
        c_sum += eps_sq

    contrast["mean"] = c_sum/3

    return contrast

def get_training_stats(project_name, experiment_name):
    with open(f"{ROOT_PATH}/checkpoints/{project_name}/{experiment_name}/wandb/latest-run/files/wandb-summary.json", "r") as jsonfile: 
        data = json.load(jsonfile)
    return data

def generate_summary_table(correlation_stats, constrast_stats, training_stats):
    report = correlation_stats["classification_report"]
    acc = correlation_stats["accuracy_score"]

    eps_0 = constrast_stats[0]['eps_sq']
    eps_1 = constrast_stats[1]['eps_sq']
    eps_2 = constrast_stats[2]['eps_sq']

    # Métrique globale : Moyenne des epsilon carrés
    global_epsilon = constrast_stats['mean']

    class_mapping = {
        '0': 'diamant (0)',
        '1': 'egg (1)',
        '2': 'triangle (2)'
    }
    cols = [class_mapping[c] for c in ['0', '1', '2']] + ['global']
    
    data_dict = {
        'precision': [report['0']['precision'], report['1']['precision'], report['2']['precision'], None],
        'recall':    [report['0']['recall'], report['1']['recall'], report['2']['recall'], None],
        'f1':        [report['0']['f1-score'], report['1']['f1-score'], report['2']['f1-score'], None],
        
        'global_accuracy':  [None, None, None, acc],
        
        'epsilon_squared': [eps_0, eps_1, eps_2, global_epsilon],
        'p_value':   [constrast_stats[0]['p_value'], constrast_stats[1]['p_value'],constrast_stats[2]['p_value'], None],
        
        'demi_cycle_v_latents': [None, None, None, training_stats.get("val/demi_cycle_v_latents")],
        'demi_cycle_attr':      [None, None, None, training_stats.get("val/demi_cycle_attr")],
        'epoch':                [None, None, None, training_stats.get("epoch")],
    }

    df = pd.DataFrame(data_dict, index=cols).T
    
    styled_df = df.style.format(precision=3, na_rep='-')
    
    return styled_df

def get_stats(project_name, experiment_name, n_samples_test, split="test", saturation=0.0):

    global_workspace = get_global_workspace(project_name, experiment_name)
    data_module = get_data_module(project_name,  experiment_name)
    test_samples = get_data_samples(data_module, n_samples_test, split= split)
    data_translated = get_data_translated(global_workspace, test_samples, n_samples_test, saturation=saturation)

    colors_np = get_samples_rgb(data_translated, "decoded_edge")
    categories_indices_train = get_categories_indices(data_translated, 'train_attr')

    original_attr, reconstructed_attr = get_attr_orig_reconstr(global_workspace, test_samples)
    
    attr_drift_stats = analyze_attribute_drift(original_attr, reconstructed_attr)
    correlation_stats = get_correlation_stats(colors_np, categories_indices_train)
    constrast_stats = get_constrat_stats(colors_np, categories_indices_train)
    training_stats = get_training_stats(project_name, experiment_name)

    table  = generate_summary_table(correlation_stats, constrast_stats, training_stats)

    orig_subset, decoded_subset = get_top_8_per_category(data_translated)
    fig_rgb_distrib = plot_rgb_distribution(colors_np, categories_indices_train, n_bins=50)
    fig_original_translated = plot_original_translated_comparison(orig_subset, decoded_subset)

    return {
        "correlation_stats": correlation_stats,
        "constrast_stats": constrast_stats,
        "training_stats": training_stats,
        "attr_drift_stats": attr_drift_stats,
        "table": table,
        "fig_rgb_distrib": fig_rgb_distrib,
        "fig_original_translated": fig_original_translated
        }

def get_global_metrics_series(experiment_name, stats):            
    global_metrics = {
        'Accuracy category classification by color': stats["correlation_stats"]["accuracy_score"],
        'Color contrast': stats["constrast_stats"]['mean'],
        'Demi-cycle v_latents loss': stats["training_stats"].get("val/demi_cycle_v_latents"),
        'Demi-cycle Attr loss': stats["training_stats"].get("val/demi_cycle_attr"),
    }
    
    return pd.Series(global_metrics, name=experiment_name)

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

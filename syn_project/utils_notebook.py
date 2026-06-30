import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from typing import cast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F

from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
from shimmer_ssd.logging import batch_to_device

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
 
from syn_project.utils_train import *
from syn_project.utils_color_analysis import *

CAT2IDX = {"Diamond": 0, "Egg": 1, "Triangle": 2}

def get_setup_modules(project_name, experiment_name):
    training_params = get_training_params(project_name, experiment_name)
    modules = ['attr', 'v_latents']
    if 'good' in experiment_name:
        modules = ['attr', 'v_latents', 'color']
    if 'modules' in training_params.keys():
        modules = training_params['modules']
    return modules


def get_data_module(project_name,  experiment_name, modules=['attr', 'v_latents']):
    training_params = load_training_params_pickle(project_name,  experiment_name)
    config = training_params["config"]
    exclude_colors = training_params["exclude_colors"]

    domain_classes = get_default_domains(modules)

    root_path = get_project_root()

    if str(root_path) in config.dataset.path:
        data_path = f"{config.dataset.path}"
    else:
        data_path = f"{root_path}/{config.dataset.path}"
       
    data_module = SimpleShapesDataModule(
        data_path,
        domain_classes,
        config.domain_proportions,
        config.training.batch_size,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=custom_collate_factory(exclude_colors=exclude_colors),
    )

    return data_module

def get_data_samples(data_module:SimpleShapesDataModule, n_samples:int, split="train", noise = 0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = data_module.get_samples(split, n_samples)
    train_samples = batch_to_device(train_samples, device)

    if noise > 0:
        category = train_samples[frozenset({'attr'})]["attr"][0]
        category = category.float()
        mean = 0.0      # Moyenne du bruit
        std = noise     # Écart-type (plus c'est haut, plus le bruit est fort)

        noise = torch.randn_like(category) * std + mean
        category_noisy =  torch.clamp(category + noise, min=1e-8)

        category_noisy_normalized = category_noisy / (category_noisy.sum(dim=-1, keepdim=True))

        train_samples[frozenset({'attr'})]["attr"][0] = category_noisy_normalized
        train_samples[frozenset({'attr', 'v_latents'})]["attr"][0] = category_noisy_normalized


    return train_samples


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
    experiment_name = condition
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
    t = global_workspace.encode_domains(x1)
 
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
    original_data: dict
    latent_domains: dict
    modules_name: list
    
def get_modules_data_from_exp(experiment_name, n_samples_test=100, split='test', checkpoint_epoch=0):

    modules = get_setup_modules('syn', experiment_name)
    global_workspace = get_global_workspace("syn", experiment_name, epoch=checkpoint_epoch, modules=modules)
    data_module = get_data_module("syn",  experiment_name, modules=modules)

    test_samples = get_data_samples(data_module, n_samples_test, split=split)
    original_data = test_samples[frozenset(modules)]

    domain_mods = global_workspace.domain_mods
    gw_mod = global_workspace.gw_mod
    latent_domains = global_workspace.encode_domains(test_samples)

    visual_module = cast(VisualLatentDomainModule, global_workspace.domain_mods["v_latents"])

    return ModuleDataOutputs(
        global_workspace=global_workspace,
        domain_mods=domain_mods,
        gw_mod=gw_mod,
        visual_module=visual_module,
        original_data =original_data,
        latent_domains=latent_domains,
        modules_name= modules
    )   


def get_objects_from_v_latents(
        latent_domains,
        gw_mod,
        global_workspace,
        modules_name,
        start_v=True,
        modality_from='attr',
        modality_through='color',
        modality_main=['attr'],
        modality_add='color'):

    latents_source_1 = []
    if start_v == True:
        latents_source_v_latents = latent_domains[frozenset({'v_latents'})]
        g0 = gw_mod.encode(latents_source_v_latents)['v_latents']
        x1 = gw_mod.decode(g0, domains=modules_name)
        latents_source_1 = x1
    else:
        latents_source_1 = latent_domains[frozenset(modules_name)]

    g1 = gw_mod.encode(latents_source_1)
    x2 = gw_mod.decode(g1[modality_from])
    g2 = gw_mod.encode(x2)

    # z_vision1 : moyenne des g1[x] pour x dans modality_main
    z_vision1 = sum(g1[m] for m in modality_main) / len(modality_main)
    vision1 = gw_mod.decode(z_vision1, domains={'v_latents'})['v_latents']

    # z_vision2 : moyenne des g1[x] pour x dans modules_name
    z_vision2 = [g1[m] for m in modality_main] + [g1[modality_add]]
    z_vision2 = sum(z_vision2) / len(z_vision2)
    vision2 = gw_mod.decode(z_vision2, domains={'v_latents'})['v_latents']

    # vision3 : décodage depuis g2[modality_through]
    x3 = gw_mod.decode(g2[modality_through], domains={'v_latents'})
    vision3 = x3['v_latents']

    # z_fusion : moyenne de (g1[x] pour x dans modality_main) + g2[modality_add]
    all_fusion = [g1[m] for m in modality_main] + [g2[modality_add]]
    z_fusion = sum(all_fusion) / len(all_fusion)
    vision4 = gw_mod.decode(z_fusion, domains={'v_latents'})['v_latents']

    return {
        'g1': g1,
        'x2': x2,
        'g2': g2,
        'z_vision1': z_vision1,
        'z_vision2': z_vision2,
        'z_fusion': z_fusion,
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

def logistic_probe(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Régression logistique pour prédire y à partir de X.

    Returns
    -------
    dict {'accuracy': float, 'report': str, 'model': LogisticRegression, 'scaler': StandardScaler}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform uniquement, pas fit_transform

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "model": model,
        "scaler": scaler,
    }

def evaluate_robust_transfer(objects, category, n_per_cat=3, n_iterations=50):
    """
    Exécute l'évaluation n_iterations fois pour obtenir des stats robustes.
    """
    all_results = {"z_spatial": [], "z_couleur": [], "z_vision_2": [], "z_syn": [], "z_syn_fusion": []}
    y = torch.argmax(category, dim=1).cpu().numpy() if category.dim() > 1 else category.cpu().numpy()
    
    for i in range(n_iterations):
        for name, data in [("z_spatial", objects["g1"]['attr']), ("z_couleur", objects["g1"]['color']), ("z_vision_2", objects["z_vision2"]), ("z_syn", objects["g2"]['color']), ("z_syn_fusion", objects["z_fusion"])]:
            X = data.detach().cpu().numpy()

            train_size = n_per_cat*3/1000
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, stratify=y
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)   # fit sur train seulement
            X_test  = scaler.transform(X_test)        # transform sur test
            
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, y_train)
            
            all_results[name].append(clf.score(X_test, y_test))

    stats = {}
    for name in all_results:
        scores = np.array(all_results[name])
        stats[name] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "ci_95": 1.96 * np.std(scores) / np.sqrt(n_iterations)
        }
        
    return stats


from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class Level2Spec:
    """Un domaine de niveau 2, recalculé en repassant par decode/encode."""
    from_domain: str
    target_domain: str
    score_name: Optional[str] = None

    def get_score_name(self) -> str:
        return self.score_name or f"{self.target_domain}2"


@dataclass
class AttentionTreeConfig:
    """Décrit toute l'arborescence à 2 niveaux."""
    root_input_domain: str = "v_latents"
    level1_domains: List[str] = field(default_factory=lambda: ["attr", "color"])
    level2_domains: List[Level2Spec] = field(default_factory=list)

    def all_score_names(self) -> List[str]:
        names = list(self.level1_domains)
        names += [spec.get_score_name() for spec in self.level2_domains]
        return names


def get_action_from_v_latents(
    latent_domains,
    gw_mod,
    global_workspace,
    modules_name,
    tree_config: Optional[AttentionTreeConfig] = None,
    fixed_weights: Optional[Dict[str, float]] = None,
):
    """
    Reproduit les mêmes chemins que MyAttentionGWLosses._compute_leaves /
    step, mais en partant de v_latents comme entrée unique (pas de raw_data,
    pas de module d'attention appris -- les poids sont fixes, passés ou
    par défaut).

    Args:
        latent_domains: dict de LatentsDomainGroupsT (comme dans la fonction
            d'origine).
        gw_mod: le module GW (encode/decode).
        global_workspace: utilisé pour encode_domains après split_binary_category_attributes.
        modules_name: domaines à décoder lors du passage GW -> domaines bruts.
        tree_config: config de l'arbre (par défaut si non fournie).
        fixed_weights: dict {score_name: poids}. Si une clé est absente,
            poids 0.0 par défaut. Pas de softmax/normalisation -- les poids
            sont utilisés tels quels (pondération libre, pour debug).

    Returns:
        dict avec 'original_category', 'action', 'z', 'leaves', 'scores'.
    """
    cfg = tree_config or AttentionTreeConfig(
        root_input_domain="v_latents",
        level1_domains=["attr"],
        level2_domains=[
            Level2Spec(from_domain="attr", target_domain="color", score_name="color2"),
        ],
    )

    score_names = cfg.all_score_names()
    weights = fixed_weights or {}
    # poids par défaut 0.0 pour toute feuille non spécifiée
    scores = {name: float(weights.get(name, 0.0)) for name in score_names}

    # --- 1. encode root_input_domain -> g ---
    key = next(k for k in latent_domains.keys() if len(k) > 1)
    latents_source = latent_domains[key]
    g = gw_mod.encode(latents_source)[cfg.root_input_domain]

    # --- 2. decode(g) -> x (domaines bruts demandés) ---
    x = gw_mod.decode(g, domains=modules_name)
    g1 = gw_mod.encode(x)

    leaves: Dict[str, torch.Tensor] = {}
    g_level1: Dict[str, torch.Tensor] = {}
    g_level0: Dict[str, torch.Tensor] = {}

    for dom in cfg.level0_domains:
        g_dom = gw_mod.encode(latents_source)[dom]
        g_level0[dom] = g_dom
        leaves[dom] = g_dom

    for dom in cfg.level1_domains:
        g_dom = g1[dom]
        g_level1[dom] = g_dom
        leaves[dom] = g_dom

    for spec in cfg.level2_domains:
        x_from = gw_mod.decode(g1[spec.from_domain], domains={spec.target_domain})
        g_target = gw_mod.encode(x_from)[spec.target_domain]
        leaves[spec.get_score_name()] = g_target

    # --- 5. combinaison pondérée (poids fixes, pas de softmax) ---
    z = sum(scores[name] * leaves[name] for name in leaves)

    # g_task = gw_mod.encode(latent_domains[frozenset({'task'})])["task"]
    # z = z + 0.5 * g_task

    # --- 6. decode z -> action ---
    action = gw_mod.decode(z, domains={"action"})["action"]

    # --- 7. catégorie d'origine, pour comparaison ---
    original_latents = latent_domains[frozenset(modules_name)]
    original_attr = original_latents["attr"]
    if isinstance(original_attr, (list, tuple)):
        original_category = torch.argmax(original_attr[0], dim=1)
    else:
        original_category = torch.argmax(original_attr, dim=1)

    return {
        "original_category": original_category,
        "action": action,
        "z": z,
        "leaves": leaves,
        "scores": scores,
    }
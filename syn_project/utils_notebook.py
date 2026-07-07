import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from typing import cast
from typing import NamedTuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim

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

def compute_reconstruction_quality(original_rgb, decoded_rgb):
    """
    original_rgb, decoded_rgb : tensors ou numpy (N, H, W, 3) ou (N, 3, H, W), valeurs dans [0,1].
    Retourne dict avec mse, ssim_mean.
    """
    # → numpy (N, H, W, 3) dans [0,1]
    def to_nhwc(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] == 3:   # (N,3,H,W)
            x = x.transpose(0, 2, 3, 1)
        return x.astype(np.float32)

    orig = to_nhwc(original_rgb)
    dec  = to_nhwc(decoded_rgb)

    mse  = float(np.mean((orig - dec) ** 2))
    ssim_scores = [
        ssim(orig[i], dec[i], channel_axis=-1, data_range=1.0)
        for i in range(len(orig))
    ]
    return {"mse": mse, "ssim": float(np.mean(ssim_scores))}

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

def _compute_leaves(gw_mod, domain_latents: dict, key: frozenset, tree_config) -> Dict[str, torch.Tensor]:
    """Reproduit exactement MyAttentionGWLosses._compute_leaves."""
    cfg = tree_config
    g = gw_mod.encode(domain_latents[key])[cfg.root_input_domain]
    x = gw_mod.decode(g)

    leaves: Dict[str, torch.Tensor] = {}

    for dom in cfg.level0_domains:
        leaves[dom] = gw_mod.encode(domain_latents[key])[dom]

    for dom in cfg.level1_domains:
        leaves[dom] = gw_mod.encode(x)[dom]

    for spec in cfg.level2_domains:
        g_from = gw_mod.encode(x)[spec.from_domain]
        x_target = gw_mod.decode(g_from)
        leaves[spec.get_score_name()] = gw_mod.encode(x_target)[spec.target_domain]

    return leaves


@torch.no_grad()
def compute_classifier_accuracy(
    gw_mod,
    latent_domains: dict,
    key: frozenset,
    tree_config,
    fixed_weights: Dict[str, float],
    target: torch.Tensor,
) -> float:
    """
    Calcule l'accuracy du classifieur d'attention (poids fixes appris) sur un split donné.
    Reproduit step() de MyAttentionGWLosses en remplaçant self.attention par les poids fixes.
    """
    leaves = _compute_leaves(gw_mod, latent_domains, key, tree_config)
    if not leaves:
        return float("nan")

    z = sum(fixed_weights[name] * leaves[name] for name in leaves)
    pred_logits = gw_mod.decode(z)["action"]
    pred = torch.argmax(pred_logits, dim=1)

    return (pred == target).float().mean().item()



def _get_category_labels(original_data: dict, modules_name: Sequence[str]) -> torch.Tensor:
    """Récupère les labels de catégorie, que ce soit via 'attr' ou 'cat'."""
    if "attr" in modules_name:
        return original_data["attr"][0]
    if "cat" in modules_name:
        return original_data["cat"]
    raise KeyError(f"Ni 'attr' ni 'cat' trouvés dans modules_name={modules_name}")

def load_fixed_weights_from_checkpoint(condition: str, checkpoint_epoch: int, score_names: List[str]) -> Dict[str, float]:
    """Charge les raw_weights depuis le checkpoint et renvoie les poids softmax fixes."""

    checkpoint_path = f"{root_path}/checkpoints/syn/{condition}/checkpoints/last.ckpt"
    if checkpoint_epoch > 0:
        checkpoint_path = f"{root_path}/checkpoints/syn/{condition}/checkpoints/save-epoch={checkpoint_epoch}.ckpt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    raw_weights_key = next((k for k in state_dict if "raw_weights" in k), None)
    if raw_weights_key is None:
        print(f"[SKIP] {condition} : pas de raw_weights")
        return {}

    raw_weights = state_dict[raw_weights_key]

    softmax_weights = torch.softmax(raw_weights, dim=0)
    return {name: softmax_weights[i].item() for i, name in enumerate(score_names)}

def evaluate_condition(
    condition: str,
    start_vision: Sequence[bool],
    n_samples_test: int,
    split: str,
    checkpoint_epoch: int,
    cat_names: Sequence[str],
    modality_kwargs: dict,
    compute_accuracy: bool = True,
) -> List[dict]:
    """Calcule les métriques (ssim, lda_score, accuracy) pour une condition, pour chaque valeur de start_v."""
    rows: List[dict] = []

    with total_silence():
        (global_workspace, domain_mods, gw_mod,
         visual_module, original_data,
         latent_domains, modules_name) = get_modules_data_from_exp(
            experiment_name=condition,
            n_samples_test=n_samples_test,
            split=split,
            checkpoint_epoch=checkpoint_epoch,
        )

    original_images_rgb = visual_module.decode_images(original_data["v_latents"])
    cat = _get_category_labels(original_data, modules_name)
    cats_np = cat.argmax(dim=1).detach().cpu().numpy()
    target = cat.argmax(dim=1).long()

    # --- accuracy du classifieur d'attention (poids fixes appris), une seule fois par condition ---
    accuracy = float("nan")
    fixed_weights: Dict[str, float] = {}
    if compute_accuracy:
        training_params = get_training_params("syn", condition)
        tree_config = training_params["attention_tree_config"]
        score_names = tree_config.all_score_names()
        fixed_weights = load_fixed_weights_from_checkpoint(condition, checkpoint_epoch, score_names)
        if fixed_weights:
            key = frozenset(modules_name)
            accuracy = compute_classifier_accuracy(
                gw_mod, latent_domains, key, tree_config, fixed_weights, target,
            )

    for start_v in start_vision:
        objects = get_objects_from_v_latents(
            latent_domains, gw_mod, global_workspace, modules_name,
            start_v=start_v, **modality_kwargs,
        )

        decoded_images = visual_module.decode_images(objects["vision2"])
        recon = compute_reconstruction_quality(original_images_rgb, decoded_images)

        colors_np = np.clip(
            (objects["x2"]["color"].detach().cpu().numpy() + 1) / 2, 0, 1
        )
        metrics, _ = hue_analysis(
            colors_np, cats_np,
            cat_names=cat_names,
            value=0.75, saturation_boost=1.8,
        )
        logistic_probe(colors_np, cats_np)  # calculé pour effet de bord / cohérence, non stocké ici

        row = {
            "condition": condition,
            "start_v": start_v,
            "ssim": recon["ssim"],
            "lda_score": metrics["lda_score"],
            "accuracy": accuracy,
        }

        row.update({f"weight_{name}": w for name, w in fixed_weights.items()})
        rows.append(row)

        del decoded_images, objects
        torch.cuda.empty_cache()

    return rows

def compute_metrics_table(
    conditions: Sequence[str],
    start_vision: Sequence[bool],
    cat_names: Sequence[str],
    n_samples_test: int = 1000,
    split: str = "test",
    checkpoint_epoch: int = 0,
    modality_kwargs: dict | None = None,
    compute_accuracy: bool = True,
) -> pd.DataFrame:
    if modality_kwargs is None:
        modality_kwargs = dict(
            modality_from="attr",
            modality_through="color",
            modality_main=["attr"],
            modality_add="color",
        )

    rows: List[dict] = []
    for condition in conditions:
        rows.extend(
            evaluate_condition(
                condition=condition,
                start_vision=start_vision,
                n_samples_test=n_samples_test,
                split=split,
                checkpoint_epoch=checkpoint_epoch,
                cat_names=cat_names,
                modality_kwargs=modality_kwargs,
                compute_accuracy=compute_accuracy,
            )
        )

    df = pd.DataFrame(rows).set_index("condition")
    return df


def display_metrics_table(df: pd.DataFrame):
    """Affichage stylé (Jupyter) du tableau de métriques."""
    numeric_cols = [c for c in ("ssim", "lda_score", "accuracy") if c in df.columns]
    weight_cols = [c for c in df.columns if c.startswith("weight_")]
    fmt = {c: "{:.3f}" for c in numeric_cols}
    fmt.update({c: "{:.3f}" for c in weight_cols})
    return (
        df.style
        .format(fmt)
        .background_gradient(subset=numeric_cols, cmap="RdYlGn")
    )


# --------------------------------------------------------------------------- #
# 2. Poids d'attention fixes par condition (+ graphe)
# --------------------------------------------------------------------------- #


def compute_attention_weights(
    conditions: Sequence[str],
    key_dims: Sequence[str],
    checkpoint_epoch: int,
) -> Dict[str, Dict[str, float]]:
    """
    Pour chaque condition : charge le tree_config d'entraînement, extrait les poids
    fixes (softmax des raw_weights) et les renormalise sur `key_dims` uniquement.
    """
    all_weights: Dict[str, Dict[str, float]] = {}

    for condition in conditions:
        training_params = get_training_params("syn", condition)
        tree_config = training_params["attention_tree_config"]
        score_names = tree_config.all_score_names()

        fixed_weights = load_fixed_weights_from_checkpoint(condition, checkpoint_epoch, score_names)
        if not fixed_weights:
            continue

        sub = {k: fixed_weights.get(k, 0.0) for k in key_dims}
        total = sum(sub.values())
        all_weights[condition] = {k: v / total for k, v in sub.items()}

    return all_weights


def plot_attention_weights(
    all_weights: Dict[str, Dict[str, float]],
    key_dims: Sequence[str],
    x_labels: Optional[Dict[str, str]] = None,
    subplot_titles: Optional[Dict[str, str]] = None,
    colors: Union[Sequence[str], Dict[str, str]] = ("#534AB7", "#1D9E75", "#EF9F27", "#E15759", "#76B7B2"),
    figsize_per_plot: float = 3.0,
    suptitle: Optional[str] = "Poids d'attention par condition",
):
    x_labels = x_labels or {}
    subplot_titles = subplot_titles or {}
 
    # Résolution des couleurs par dimension, quel que soit le format d'entrée
    if isinstance(colors, dict):
        color_map = {dim: colors.get(dim, "#888888") for dim in key_dims}
    else:
        palette = list(colors)
        color_map = {dim: palette[i % len(palette)] for i, dim in enumerate(key_dims)}
    bar_colors = [color_map[dim] for dim in key_dims]
    tick_labels = [x_labels.get(dim, dim) for dim in key_dims]
 
    n = len(all_weights)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_plot * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
 
    for ax, (condition, weights) in zip(axes, all_weights.items()):
        values = [weights[k] for k in key_dims]
        bars = ax.bar(key_dims, values, color=bar_colors, width=0.6)
        ax.set_title(subplot_titles.get(condition, condition), fontsize=9)
        ax.set_ylim(0, 1.15)  # marge au-dessus de 1.0 pour que les labels ne dépassent pas
        ax.set_xticks(range(len(key_dims)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )
 
    axes[0].set_ylabel("proportion")
 
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])  # réserve de la place en haut pour le suptitle
    else:
        fig.tight_layout()
 
    plt.show()
    return fig, axes
 
def build_publication_table(
    metrics_df: pd.DataFrame,
    n_neurons: Dict[str, int],
    include_columns: Optional[Sequence[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    decimals: int = 2,
    start_v: Optional[bool] = None,
    n_neurons_column_name: str = "n_neurons",
) -> pd.DataFrame:
    """
    Construit un DataFrame prêt pour publication à partir de metrics_df.
 
    Args:
        metrics_df: sortie de compute_metrics_table (index = 'condition').
        n_neurons: mapping {condition: nombre de neurones} à ajouter comme colonne.
        include_columns: liste des colonnes à garder, dans l'ordre souhaité
            (noms *avant* renommage, ex. ["condition", "n_neurons", "accuracy", "ssim"]).
            Si None, toutes les colonnes de metrics_df + n_neurons sont gardées.
        rename_map: mapping {nom_original: nom_affiché} appliqué après la sélection
            (ex. {"ssim": "SSIM", "lda_score": "LDA", "accuracy": "Accuracy (%)"}).
        decimals: nombre de chiffres après la virgule pour les colonnes numériques.
        start_v: si metrics_df contient plusieurs lignes par condition (colonne 'start_v'),
            filtre sur cette valeur avant de construire le tableau. Si None et que
            plusieurs valeurs de start_v existent, lève une erreur explicite.
        n_neurons_column_name: nom de la colonne créée pour le nombre de neurones.
 
    Returns:
        DataFrame propre, une ligne par condition, colonnes sélectionnées/renommées,
        valeurs numériques arrondies.
    """
    df = metrics_df.reset_index()  # remet 'condition' en colonne normale
 
    if "start_v" in df.columns and df["start_v"].nunique() > 1:
        if start_v is None:
            raise ValueError(
                "metrics_df contient plusieurs valeurs de start_v "
                f"({sorted(df['start_v'].unique())}) ; précise start_v=True/False."
            )
        df = df[df["start_v"] == start_v].drop(columns=["start_v"])
    elif "start_v" in df.columns:
        df = df.drop(columns=["start_v"])
 
    df[n_neurons_column_name] = df["condition"].map(n_neurons)
    if df[n_neurons_column_name].isna().any():
        missing = df.loc[df[n_neurons_column_name].isna(), "condition"].tolist()
        raise ValueError(f"n_neurons manquant pour les conditions : {missing}")
 
    if include_columns is not None:
        missing_cols = [c for c in include_columns if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Colonnes demandées introuvables dans metrics_df : {missing_cols}")
        df = df[list(include_columns)]
 
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
 
    if rename_map:
        df = df.rename(columns=rename_map)
 
    return df.reset_index(drop=True)
 
def render_publication_table(
    df: pd.DataFrame,
    decimals: int = 2,
    col_widths: Optional[Sequence[float]] = None,
    fontsize: int = 10,
    header_fontsize: Optional[int] = None,
    font_family: str = "serif",
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
):
    """
    Rend un DataFrame déjà propre (voir build_publication_table) sous forme d'image
    de tableau au format "booktabs" (lignes horizontales uniquement, pas de grille
    verticale, en-tête en gras) — style standard pour une figure de tableau en article.
 
    Returns:
        (fig, ax) matplotlib, exportable en PDF/PNG haute résolution pour publication.
    """
    header_fontsize = header_fontsize or fontsize
 
    n_rows, n_cols = df.shape
    if figsize is None:
        figsize = (1.4 * n_cols, 0.5 * (n_rows + 1) + 0.5)
 
    # Formatage des valeurs (arrondi déjà fait en amont par build_publication_table,
    # mais on force ici l'affichage à `decimals` chiffres pour les colonnes numériques)
    display_df = df.copy()
    float_cols = display_df.select_dtypes(include="float").columns
    int_cols = display_df.select_dtypes(include="integer").columns
    for col in float_cols:
        display_df[col] = display_df[col].map(lambda v: f"{v:.{decimals}f}")
    for col in int_cols:
        display_df[col] = display_df[col].map(lambda v: f"{v:d}")
 
    with plt.rc_context({"font.family": font_family}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
 
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
 
        if col_widths is not None:
            for row_key, cell in table.get_celld().items():
                col_idx = row_key[1]
                cell.set_width(col_widths[col_idx])
 
        # Style "booktabs" : pas de bordures verticales, lignes horizontales sélectives,
        # en-tête en gras
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor("none")
            cell.set_linewidth(0)
            if row == 0:
                cell.set_text_props(weight="bold", fontsize=header_fontsize)
                cell.visible_edges = "TB"    # \toprule + ligne sous l'en-tête
                cell.set_linewidth(1.0)
                cell.set_edgecolor("black")
            elif row == n_rows:
                cell.visible_edges = "B"     # \bottomrule
                cell.set_linewidth(1.0)
                cell.set_edgecolor("black")
            else:
                cell.visible_edges = ""
 
        table.scale(1, 1.6)
 
        if title:
            ax.set_title(title, fontsize=fontsize + 1, pad=12)
 
        plt.tight_layout()
        plt.show()
 
    return fig, ax
 
 
def export_latex_table(
    df: pd.DataFrame,
    decimals: int = 2,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Génère le code LaTeX (style booktabs) du tableau, prêt à copier dans un article.
    Nécessite \\usepackage{booktabs} dans le document LaTeX cible.
    """
    fmt = {col: f"{{:.{decimals}f}}".format for col in df.select_dtypes(include="number").columns}
    styler = df.style.format(fmt, na_rep="--").hide(axis="index")
    return styler.to_latex(
        hrules=True,
        caption=caption,
        label=label,
        column_format="l" + "c" * (df.shape[1] - 1),
    )
 

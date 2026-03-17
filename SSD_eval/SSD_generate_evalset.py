# generate_ssd.py
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 # Importation nécessaire si generate_image l'utilise implicitement via matplotlib/patches
import matplotlib.path as mpath
from matplotlib import patches
import csv
import argparse
import math
from pathlib import Path
from PIL import Image # Importation nécessaire si generate_image l'utilise implicitement
from typing import cast
# Supposons que ces fonctions proviennent de modules externes comme indiqué dans le script original
# Si elles sont définies localement ou dans un fichier utils spécifique, ajustez l'importation
from simple_shapes_dataset.cli import generate_image # Assurez-vous que ce module/fonction est accessible
from SSD_utils import generate_fixed_colors, generate_fixed_colors_original         # Assurez-vous que ce module/fonction est accessible


# ====================
# FONCTION DE GENERATION DU DATASET
# ====================

def generate_dataset(output_dir: str, imsize: int = 32, min_scale: int = 7, max_scale: int = 14,
                     n_sizes: int = 4, n_colors: int = 100, seed: int = 0,
                     rotations_array: np.ndarray = None, positions_array: np.ndarray = None,
                     shapes: list[int] = [0, 1, 2], gen_col = None) -> None:
    """
    Génère un jeu de données d'images de formes simples avec leurs attributs.

    Args:
        output_dir: Répertoire où sauvegarder les images et le fichier CSV.
        imsize: Taille des images générées (imsize x imsize).
        min_scale: Taille minimale de la forme.
        max_scale: Taille maximale de la forme.
        n_sizes: Nombre de tailles différentes à générer entre min_scale et max_scale.
        n_colors: Nombre de couleurs différentes à générer.
        seed: Graine pour la génération aléatoire des couleurs (reproductibilité), optionnel mais utile si volonté d'introduire une génération de couleur aléatoire.
        Paramètre inactif pour le moment.
        rotations_array: Tableau NumPy des rotations à appliquer (en radians).
                         Si None, utilise [0, pi/2, pi, 3pi/2].
        positions_array: Tableau NumPy des positions [x, y] du centre des formes.
                         Si None, utilise les 4 coins avec une marge.
        shapes: Liste des identifiants des formes à générer (ex: [0, 1, 2] pour cercle, carré, triangle).
    """
    print("--- Démarrage de la génération du dataset ---")
    np.random.seed(seed)

    # Définition des rotations par défaut si non fournies
    if rotations_array is None:
        rotations_array = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        print(f"Utilisation des rotations par défaut: {rotations_array}")

    # Définition des positions par défaut si non fournies
    if positions_array is None:
        margin = max_scale // 2 + 1 # Ajout d'une marge pour éviter les coupures
        positions_array = np.array([
            [margin, margin],
            [margin, imsize - margin],
            [imsize - margin, margin],
            [imsize - margin, imsize - margin]
        ])
        print(f"Utilisation des positions par défaut:\n{positions_array}")

    # Calcul des tailles et génération des couleurs
    sizes = np.linspace(min_scale, max_scale, n_sizes, dtype=int)
    print(f"Génération avec {len(sizes)} tailles: {sizes}")

    if gen_col == "original":
        output_dir = output_dir + "_original"
        rgb_colors, _ = generate_fixed_colors_original(n_colors, seed=seed) # Utilisation de la seed ici aussi
        print(f"Génération avec {n_colors} couleurs uniques.")

    elif gen_col == "single_gray":
        rgb_colors = np.tile(np.array([[127, 127, 127]]), (n_colors, 1))
    else : 
        rgb_colors, _ = generate_fixed_colors(n_colors)
        print(f"Génération avec {n_colors} couleurs uniques.")

    

    # Création du répertoire de sortie et du fichier CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_file = out_dir / "attributes.csv"
    print(f"Le dataset sera sauvegardé dans: {out_dir.resolve()}")

    total_expected = len(shapes) * n_colors * len(sizes) * len(rotations_array) * len(positions_array)
    print(f"Nombre total d'images attendues: {total_expected}")
    count = 0

    # Boucle de génération
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "color_index", "size", "rotation", "location"])
        for color_idx in range(n_colors):
            for cls in shapes:
                for size in sizes:
                    for rotation in rotations_array:
                        for pos in positions_array:
                            color = rgb_colors[color_idx]
                            dpi = 1
                            
                            fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
                            ax = cast(plt.Axes, ax)
                            
                            generate_image(ax, cls, pos, size, rotation, color, imsize)
                            ax.set_facecolor("black")
                            plt.tight_layout(pad=0)
                            
                            filename = f"image_{count:05d}.png"
                            filepath = out_dir / filename
                            plt.savefig(filepath, dpi=dpi, format="png")
                            plt.close(fig)
                            writer.writerow([filename, cls, color_idx, size, rotation, pos.tolist()])
                            count += 1
                            if count % 1000 == 0:
                                print(f"... {count}/{total_expected} images générées.")

    print(f"\n--- Génération terminée ---")
    print(f"Dataset créé avec {count} images (attendu: {total_expected}).")
    print(f"Fichier d'attributs sauvegardé : {csv_file.resolve()}")



# ====================
# MAIN : ARGUMENTS & EXECUTION
# ====================
if __name__ == "__main__":
    rotation = np.array([i*np.pi/8 for i in range(16)])  
    margin = 8
    imsize =  32
    x_positions = [8, 13, 18, 24]
    y_positions = [13, 18]
    positions = []
    for x in x_positions:
        for y in y_positions:
            positions.append([x, y])
        positions_array = np.array(positions)
    generate_dataset(
        rotations_array = rotation,
        positions_array = positions_array,
        output_dir="./evaluation_set_examples",
        imsize=32, 
        min_scale=10,
        max_scale=14,
        n_sizes=8,
        n_colors=5,
        seed=0,
        gen_col = "normal",
    )

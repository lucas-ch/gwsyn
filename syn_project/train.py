import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    project_name = "syn"
    condition = "4mod4"
    data = "biased_00"
    switch_epoch = 0

    modules = ['position', 'cat', 'color', 'v_latents']

    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=["high_cycles.yaml"])
    if len(modules) > 2:
        config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=["color_mod.yaml"])
                
    experiment_name = get_experiment_name(condition, data, switch_epoch)
    exclude_colors = False if condition == "control" else True

    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    config.training.batch_size = 2056
    config.seed = 0

    apply_custom_init = True

    custom_hparams = {
        "temperature": 1,
        "alpha": 1
    }

    custom_weights = {
        'contrastive_cat_and_color': 0.1,
        'contrastive_cat_and_v_latents': 0.1,
        'contrastive_color_and_v_latents': 0.1,
        'contrastive_position_and_cat': 0.1,
        'contrastive_position_and_color': 0.1,
        'contrastive_position_and_v_latents': 0.1,

        'cycle_cat_through_color': 1.0,
        'cycle_cat_through_position': 1.0,
        'cycle_cat_through_v_latents': 1.0,
        'cycle_color_through_cat': 1.0,
        'cycle_color_through_position': 1.0,
        'cycle_color_through_v_latents': 1.0,
        'cycle_position_through_cat': 1.0,
        'cycle_position_through_color': 1.0,
        'cycle_position_through_v_latents': 1.0,
        'cycle_v_latents_through_cat': 1.0,
        'cycle_v_latents_through_color': 1.0,
        'cycle_v_latents_through_position': 1.0,

        'demi_cycle_cat': 1.0,
        'demi_cycle_color': 1.0,
        'demi_cycle_position': 1.0,
        'demi_cycle_v_latents': 1.0,

        # --- Cible: CAT ---
        'translation_color_to_cat': 1.0,
        'translation_position_to_cat': 1.0,
        'translation_v_latents_to_cat': 1.0,
        'translation_color/position_to_cat': 1.0,
        'translation_color/v_latents_to_cat': 1.0,
        'translation_position/v_latents_to_cat': 1.0,
        'translation_color/position/v_latents_to_cat': 1.0,

        # --- Cible: COLOR ---
        'translation_cat_to_color': 1.0,
        'translation_position_to_color': 1.0,
        'translation_v_latents_to_color': 1.0,
        'translation_cat/position_to_color': 1.0,
        'translation_cat/v_latents_to_color': 1.0,
        'translation_position/v_latents_to_color': 1.0,
        'translation_cat/position/v_latents_to_color': 1.0,

        # --- Cible: POSITION ---
        'translation_cat_to_position': 1.0,
        'translation_color_to_position': 1.0,
        'translation_v_latents_to_position': 1.0,
        'translation_cat/color_to_position': 1.0,
        'translation_cat/v_latents_to_position': 1.0,
        'translation_color/v_latents_to_position': 1.0,
        'translation_cat/color/v_latents_to_position': 1.0,

        # --- Cible: V_LATENTS ---
        'translation_cat_to_v_latents': 0.1,
        'translation_color_to_v_latents': 0.1,
        'translation_position_to_v_latents': 0.1,
        'translation_cat/color_to_v_latents': 1.0,
        'translation_cat/position_to_v_latents': 1.0,
        'translation_color/position_to_v_latents': 1.0,
        'translation_cat/color/position_to_v_latents': 10.0,
        }

    noise = {"mean": 0.0, "std": 0.0}

    log_training_params = {
        "experiment_name": experiment_name,
        "exclude_colors": exclude_colors,
        "apply_custom_init": apply_custom_init,
        "config": config,
        "custom_hparams": custom_hparams,
        "swith_epoch": switch_epoch,
        "custom_weights": custom_weights,
        "modules": modules
    }

    save_training_params_pickle(log_training_params, project_name, experiment_name)

    model, checkpoint_path = train_global_workspace(
        config,
        custom_hparams=custom_hparams, 
        project_name=project_name,
        experiment_name=experiment_name,
        apply_custom_init=apply_custom_init,
        exclude_colors=exclude_colors,
        load_from_checkpoint=False,
        switch_epoch=switch_epoch,
        custom_weights=custom_weights,
        noise=noise,
        modules=modules
    )
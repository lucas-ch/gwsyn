import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    project_name = "syn"
    condition = "3mod_cselc"
    data = "biased_00"
    switch_epoch = 0

    modules = ['attr', 'v_latents', 'color']
    modules_to_freeze = []
    load_from_checkpoint = False
    gw_checkpoint_path = None

    clean_names = [m.replace("_", "") for m in sorted(modules)]
    config_filename = f"{'_'.join(clean_names)}.yaml"

    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=[config_filename])
                
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
    # --- DEMI-CYCLE (Reconstruction directe) ---
    'demi_cycle_attr': 1.0,
    'demi_cycle_color': 1.0,
    'demi_cycle_v_latents': 1.0,

    # --- TRANSLATIONS (N-to-1) ---
    # Cible: ATTR
    'translation_color_to_attr': 1.0,
    'translation_v_latents_to_attr': 1.0,
    'translation_color/v_latents_to_attr': 1.0,

    # Cible: COLOR
    'translation_attr_to_color': 1.0,
    'translation_v_latents_to_color': 1.0,
    'translation_attr/v_latents_to_color': 1.0,

    # Cible: V_LATENTS
    'translation_attr_to_v_latents': 1.0,
    'translation_color_to_v_latents': 1.0,
    'translation_attr/color_to_v_latents': 1.0,

    # --- CYCLES (1-through-N) ---
    # Source: ATTR
    'cycle_attr_through_color': 1.0,
    'cycle_attr_through_v_latents': 1.0,
    'cycle_attr_through_color/v_latents': 1.0,

    # Source: COLOR
    'cycle_color_through_attr': 1.0,
    'cycle_color_through_v_latents': 1.0,
    'cycle_color_through_attr/v_latents': 1.0,

    # Source: V_LATENTS
    'cycle_v_latents_through_attr': 1.0,
    'cycle_v_latents_through_color': 1.0,
    'cycle_v_latents_through_attr/color': 1.0,

    # --- CONTRASTIVE (Pairs) ---
    'contrastive_attr_and_color': 1.0,
    'contrastive_attr_and_v_latents': 1.0,
    'contrastive_color_and_v_latents': 1.0,
}



    noise = {"mean": 1.0, "std": 0.0}

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
        load_from_checkpoint=load_from_checkpoint,
        gw_checkpoint_path=gw_checkpoint_path,
        switch_epoch=switch_epoch,
        custom_weights=custom_weights,
        noise=noise,
        modules=modules,
        modules_to_freeze=modules_to_freeze
    )
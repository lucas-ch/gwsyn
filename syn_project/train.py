import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    project_name = "syn"
    condition = "debug"
    data = "biased_00"
    switch_epoch = 0

    modules = ['attr', 'color', 'v_latents']

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
        'cycle_color_through_v_latents': 0.0,
        'demi_cycle_attr': 1.0,
        'demi_cycle_color': 1.0,
        'demi_cycle_v_latents': 1.0,
        'translation_v_latents_to_color': 1.0,
        'translation_attr_to_v_latents': 1.0,
        'translation_color_to_v_latents': 0.0,
        'translation_v_latents_to_attr': 1.0,
        'cycle_attr_through_color_loss_cat': 1.0,
        'attr_color_loss': 1.0
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
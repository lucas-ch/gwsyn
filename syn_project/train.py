
import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=["high_cycles.yaml"])
    
    project_name = "syn"
    condition = "debug"
    data = "test1"
    switch_epoch = 10000

    experiment_name = get_experiment_name(condition, data, switch_epoch)
    exclude_colors = False if condition == "control" else True

    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    config.training.max_steps = 150000
    config.training.batch_size = 32
    config.seed = 0

    apply_custom_init = True

    custom_hparams = {
        "temperature": 1,
        "alpha": 1
    }

    custom_weights = {
            "cycle_attr_through_v_latents_loss_attr": 0.0,
            "cycle_attr_through_v_latents_loss_cat": 0.0,
            "cycle_v_latents_through_attr": 0.0,
            "demi_cycle_attr": 0.0,
            "demi_cycle_v_latents": 0.0,
            "translation_v_latents_to_attr_loss_attr": 0.0,
            "translation_v_latents_to_attr_loss_cat": 0.0,
            "translation_attr_to_v_latents": 0.0,
            "contrastive_loss": 1.0,
            "shape_loss": 0.0
        }

    noise = {"mean": 0.0, "std": 0.0}

    log_training_params = {
        "experiment_name": experiment_name,
        "exclude_colors": exclude_colors,
        "apply_custom_init": apply_custom_init,
        "config": config,
        "custom_hparams": custom_hparams,
        "swith_epoch": switch_epoch,
        "custom_weights": custom_weights
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
        noise=noise
    )
import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    project_name = "syn"
    condition = "task_att_debug2"
    data = "biased_00"
    switch_epoch = 0

    modules = ['attr', 'color', 'v_latents', 'action']
    modules_to_freeze = ['attr', 'color', 'v_latents']
    load_from_checkpoint = True
    gw_checkpoint_path = "/home/lucas/gwsyn/checkpoints/syn/base_task_biased_00/checkpoints/last.ckpt"
    fusion_activation_fn = torch.tanh

    clean_names = [m.replace("_", "") for m in sorted(modules)]
    config_filename = f"{'_'.join(clean_names)}.yaml"

    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=[config_filename])
                
    experiment_name = get_experiment_name(condition, data, switch_epoch)
    exclude_colors = False if condition == "control" else True

    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    config.training.batch_size = 2056
    config.seed = 126

    apply_custom_init = True

    custom_hparams = {
        "temperature": 1,
        "alpha": 0.5
    }

    custom_weights = {}

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
        modules_to_freeze=modules_to_freeze,
        fusion_activation_fn=fusion_activation_fn
    )
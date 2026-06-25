import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    project_name = "syn"
    condition = "narbi_3"
    data = "form_50_35_15_cgc_3_20"
    switch_epoch = 0

    modules = ['attr', 'color', 'v_latents']
    modules_to_freeze = []
    load_from_checkpoint = False
    gw_checkpoint_path = None
    exclude_colors = True
    custom_hparams = {
        "temperature": 1,
        "alpha": 0.5
    }
    attention_tree_config = None

    clean_names = [m.replace("_", "") for m in sorted(modules)]
    config_filename = f"{'_'.join(clean_names)}.yaml"
    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=[config_filename])
                
    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    config.training.batch_size = 2056
    config.max_train_size = 500000
    config.seed = 126

    log_training_params = {
        "experiment_name": condition,
        "exclude_colors": exclude_colors,
        "apply_custom_init": True,
        "config": config,
        "custom_hparams": custom_hparams,
        "swith_epoch": switch_epoch,
        "custom_weights": None,
        "modules": modules,
        "attention_tree_config": attention_tree_config
    }

    save_training_params_pickle(log_training_params, project_name, condition)

    model, checkpoint_path = train_global_workspace(
        config,
        custom_hparams=custom_hparams, 
        project_name=project_name,
        experiment_name=condition,
        exclude_colors=exclude_colors,
        load_from_checkpoint=load_from_checkpoint,
        gw_checkpoint_path=gw_checkpoint_path,
        switch_epoch=switch_epoch,
        modules=modules,
        modules_to_freeze=modules_to_freeze,
        attention_tree_config=attention_tree_config
    )
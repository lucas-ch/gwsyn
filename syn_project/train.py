import torch
from shimmer_ssd.config import load_config
from .utils_train import *


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    project_name = "syn"
    condition = "biased_50"
    data = "biased_50"

    modules = ['attr', 'v_latents', 'color']
    modules_to_freeze = []
    load_from_checkpoint = False
    checkpoint_to_load = None
    exclude_colors = True
    custom_hparams = {
        "temperature": 1,
        "alpha": 0.5
    }

    attention_weight = 0.0

    attention_tree_config = None

    attention_init=None



    clean_names = [m.replace("_", "") for m in sorted(modules)]
    config_filename = f"{'_'.join(clean_names)}.yaml"
    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=[config_filename])
                
    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    config.training.batch_size = 2056
    config.max_train_size = 500000
    config.seed = 0

    log_training_params = {
        "experiment_name": condition,
        "exclude_colors": exclude_colors,
        "apply_custom_init": True,
        "config": config,
        "custom_hparams": custom_hparams,
        "custom_weights": None,
        "modules": modules,
        "attention_tree_config": attention_tree_config,
        "attention_init": attention_init,
        "attention_weight": attention_weight
    }

    save_training_params_pickle(log_training_params, project_name, condition)

    model, checkpoint_path = train_global_workspace(
        config,
        custom_hparams=custom_hparams, 
        project_name=project_name,
        experiment_name=condition,
        exclude_colors=exclude_colors,
        load_from_checkpoint=load_from_checkpoint,
        gw_checkpoint_path=f"{ROOT_PATH}/checkpoints/syn/{checkpoint_to_load}/checkpoints/last.ckpt",
        modules=modules,
        modules_to_freeze=modules_to_freeze,
        attention_tree_config=attention_tree_config,
        attention_init=attention_init,
        attention_weight=attention_weight
    )
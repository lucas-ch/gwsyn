from syn_project.train import ROOT_PATH, train_global_workspace
from syn_project.utils_notebook import get_experiment_name
from shimmer_ssd.config import Config, load_config

def test_callback():
    config = load_config(f"{ROOT_PATH}/config", use_cli=False, load_files=["high_cycles.yaml"])
    
    project_name = "test"
    condition = "control"
    data = "test1"
    switch_epoch = 0

    experiment_name = get_experiment_name(condition, data, switch_epoch)
    experiment_name = f"{condition}_{data}"

    if switch_epoch > 0:
        experiment_name = f"{experiment_name}_switch_{switch_epoch}"
    
    config.dataset.path = f"{ROOT_PATH}/simple_shapes_dataset_{data}"
    exclude_colors = False if condition == "control" else True
    apply_custom_init = True
    config.seed = 0

    custom_hparams = {
        "temperature": 1,
        "alpha": 1
    }

    config.training.max_steps = 30

    model, checkpoint_path = train_global_workspace(
        config,
        custom_hparams=custom_hparams, 
        project_name=project_name,
        experiment_name=experiment_name,
        apply_custom_init=apply_custom_init,
        exclude_colors=exclude_colors,
        load_from_checkpoint=False,
        switch_epoch=switch_epoch
    )
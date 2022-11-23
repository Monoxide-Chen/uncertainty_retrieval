import json
import os
import pprint as pp
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb


def fix_random_seed_as(random_seed):
    if random_seed == -1:
        random_seed = np.random.randint(100000)
        print("RANDOM SEED: {}".format(random_seed))

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    return random_seed


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def create_experiment_export_folder(experiment_dir, experiment_description):
    print(os.path.abspath(experiment_dir))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    print(os.path.abspath(experiment_path))
    os.mkdir(experiment_path)
    print("folder created: " + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def export_config_as_json(config, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=2)


def generate_tags(config):
    tags = []
    tags.append(config.get('generator', config.get('text_encoder')))
    tags.append(config.get('trainer'))
    tags = [tag for tag in tags if tag is not None]
    return tags


def set_up_gpu(device_idx):
    if device_idx:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_idx
        return {
            'num_gpu': len(device_idx.split(","))
        }
    else:
        idxs = os.environ['CUDA_VISIBLE_DEVICES']
        return {
            'num_gpu': len(idxs.split(","))
        }


def setup_experiment(config):
    device_info = set_up_gpu(config['device_idx'])
    config.update(device_info)

    random_seed = fix_random_seed_as(config['random_seed'])
    config['random_seed'] = random_seed
    export_root = create_experiment_export_folder(config['experiment_dir'], config['experiment_description'])
    export_config_as_json(config, export_root)
    config['export_root'] = export_root

    pp.pprint(config, width=1)
    os.environ['WANDB_SILENT'] = "true"
    tags = generate_tags(config)
    project_name = config['wandb_project_name']
    wandb_account_name = config['wandb_account_name']
    experiment_name = config['experiment_description']
    experiment_name = experiment_name if config['random_seed'] != -1 else experiment_name + "_{}".format(random_seed)
    wandb.init(config=config, name=experiment_name, project=project_name,
               entity=wandb_account_name, tags=tags)
    return export_root, config

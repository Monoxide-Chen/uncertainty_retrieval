from options.command_line import load_config_from_command
from options.config_file import load_config_from_file


def _merge_configs(configs_ordered_by_increasing_priority):
    merged_config = {}
    for config in configs_ordered_by_increasing_priority:
        for k, v in config.items():
            merged_config[k] = v
    return merged_config


def _check_mandatory_config(config_from_config_file, user_defined_configs,
                            exception_keys=('experiment_description', 'device_idx')):
    exception_keys = [] if exception_keys is None else exception_keys
    trigger = False
    undefined_configs = []
    for key, val in config_from_config_file.items():
        if val == "":
            if key not in user_defined_configs and key not in exception_keys:
                trigger = True
                undefined_configs.append(key)
                print("Must define {} setting from command".format(key))
    if trigger:
        raise Exception('Mandatory configs not defined:', undefined_configs)


def _generate_experiment_description(configs, config_from_command):
    experiment_description = configs['experiment_description']

    if experiment_description == "":
        remove_keys = ['dataset', 'trainer', 'config_path', 'device_idx']
        for key in remove_keys:
            if key in config_from_command:
                config_from_command.pop(key)

        descriptors = []
        for key, val in config_from_command.items():
            descriptors.append(key + str(val))
        experiment_description = "_".join([configs['dataset'], configs['trainer'], *descriptors])
    return experiment_description


def get_experiment_config():
    config_from_command, user_defined_configs = load_config_from_command()
    config_from_config_file = load_config_from_file(config_from_command['config_path'])
    _check_mandatory_config(config_from_config_file, user_defined_configs)
    merged_configs = _merge_configs([config_from_command, config_from_config_file, user_defined_configs])
    experiment_description = _generate_experiment_description(merged_configs, user_defined_configs)
    merged_configs['experiment_description'] = experiment_description
    return merged_configs

import json


def load_config_from_file(json_path):
    if not json_path:
        return {}

    with open(json_path, 'r') as f:
        config = json.load(f)

    print("Config at '{}' has been loaded".format(json_path))
    return config

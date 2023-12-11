import os
import yaml

__all__ = ["load_config"]


def load_config():
    calling_script_path = os.path.abspath(__file__)
    tests_path = os.path.dirname(calling_script_path)
    config_path = os.path.join(tests_path, "config.yml")

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config

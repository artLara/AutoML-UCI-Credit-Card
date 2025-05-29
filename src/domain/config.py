import yaml
from pathlib import PosixPath

def parse_yaml_config(yaml_file:PosixPath) -> dict:
    config = yaml.safe_load(open(yaml_file))

    return config
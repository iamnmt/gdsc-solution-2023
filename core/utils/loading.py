import yaml


def load_yaml(path: str):
    with open(path, "rt") as f:
        return yaml.safe_load(f)

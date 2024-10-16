from glob import glob
import yaml
import pickle
from paths import data
from pathlib import Path
import os


def build_model_size_dict():
    with open(data / "name_to_dir_map.pkl", "rb") as handle:
        name_to_dir = pickle.load(handle)

    model_size_dict = {}

    for name, path in name_to_dir.items():
        yaml_path_parts = Path(path).parts[:-2]
        yaml_file = glob(os.path.join(*yaml_path_parts, "*.yaml"))[0]
        with open(yaml_file, "r") as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        model_size_dict[name] = d["nr_of_parameters"]

    return model_size_dict


if __name__ == "__main__":
    model_size_dict = build_model_size_dict()

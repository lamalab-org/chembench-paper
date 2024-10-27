import fire
import os
from glob import glob
import yaml
import pickle
from pathlib import Path
from utils import obtain_chembench_repo
from paths import data

def build_name_dir_map(report_base_dir=None):
    if report_base_dir is None:
        report_base_dir = obtain_chembench_repo()
        report_base_dir = os.path.join(report_base_dir, "reports")
    yaml_files = glob(f"{report_base_dir}/**/*.yaml")

    name_dir_map = {}

    # in the dir with the yaml files, there is a directory reports and in there is a directory
    # we want to get the path to that directory
    # |-- reports
    # |   |-- model_name
    # |   - model_name.yaml
    # |       |-- reports
    # |          |-- model_name

    for yaml_file in yaml_files:
        yaml_dir = Path(yaml_file).parent
        report_subdirs = glob(f"{yaml_dir}/reports/*")
        if len(report_subdirs) == 0:
            raise ValueError(f"No reports found in {yaml_dir}")
        if len(report_subdirs) > 1:
            raise ValueError(f"Multiple reports found in {yaml_dir}")

        report_subdir = report_subdirs[0]
        model_name = os.path.basename(report_subdir)

        with open(yaml_file) as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        if "model_name" in yaml_data:
            if "temperature" in yaml_data and yaml_data["temperature"] == 0:
                model_name = yaml_data["model_name"]
            elif "temperature" in yaml_data and yaml_data["temperature"] == 1:
                if yaml_data["model_name"] == "o1":
                    model_name = yaml_data["model_name"]
                else:
                    model_name = yaml_data["model_name"] + "-T-one"
            else:
                model_name = yaml_data["model_name"]
        name_dir_map[model_name] = os.path.abspath(report_subdir)

    with open(data / "name_to_dir_map.pkl", "wb") as f:
        pickle.dump(name_dir_map, f)

    return name_dir_map

if __name__ == "__main__":
    fire.Fire(build_name_dir_map)

from glob import glob
import yaml
import pickle
from paths import data
from pathlib import Path
import os
import matplotlib.pyplot as plt
from paths import scripts, figures, data
from plotutils import model_color_map, range_frame
from utils import ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, TWO_COL_WIDTH_INCH
import numpy as np

plt.style.use(scripts / "lamalab.mplstyle")


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


llama_models_31 = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-405B-Instruct",
]

llama_models_3 = ["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"]


def plot_size_vs_performance(model_scores_dict):
    overall_scores = model_scores_dict["overall"]
    model_size_dict = build_model_size_dict()

    # llama3 models
    llama3_scores = [overall_scores[m]["all_correct_"].mean() for m in llama_models_3]

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    ax.plot(
        [model_size_dict[m] for m in llama_models_3],
        llama3_scores,
        ls="-",
        marker="o",
        label="Llama-3",
    )

    # for i, m in enumerate(llama_models_3):
    #     ax.annotate(m, (model_size_dict[m], llama3_scores[i]))

    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Performance")

    # llama31 models
    llama31_scores = [overall_scores[m]["all_correct_"].mean() for m in llama_models_31]

    ax.plot(
        [model_size_dict[m] for m in llama_models_31],
        llama31_scores,
        ls="-",
        marker="o",
        label="Llama-3.1",
    )

    # ax.set_xscale("log")

    range_frame(
        ax,
        np.array([model_size_dict[m] for m in llama_models_31]),
        np.array(llama31_scores),
    )

    fig.tight_layout()

    ax.legend()
    fig.savefig(figures / "model_size_plot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    model_size_dict = build_model_size_dict()
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        model_scores_dict = pickle.load(handle)

    plot_size_vs_performance(model_scores_dict)

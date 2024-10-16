from glob import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plotutils import range_frame
from paths import output, scripts, figures
import pickle
from paths import data
from utils import obtain_chembench_repo
from definitions import MODELS_TO_PLOT

plt.style.use(scripts / "lamalab.mplstyle")


human_dir = output / "human_scores"

chembench_repo = obtain_chembench_repo()

with open(data / "name_to_dir_map.pkl", "rb") as handle:
    model_file_name_to_label = pickle.load(handle)

human_reports_dir = os.path.join(chembench_repo, "reports", "humans")

human_scores_with_tools_files = glob(os.path.join(human_reports_dir, "*_h_tool*.json"))

human_scores_without_tools_files = glob(
    os.path.join(human_reports_dir, "*_h_notool*.json")
)

human_scores_with_tools, human_scores_without_tools = [], []

for file in human_scores_with_tools_files:
    with open(file, "r") as handle:
        d = json.load(handle)
        human_scores_with_tools.append(d["fraction_correct"])

for file in human_scores_without_tools_files:
    with open(file, "r") as handle:
        d = json.load(handle)
        human_scores_without_tools.append(d["fraction_correct"])

human_scores_with_tools = np.array(human_scores_with_tools)
human_scores_without_tools = np.array(human_scores_without_tools)

all_human_scores = np.concatenate([human_scores_with_tools, human_scores_without_tools])


def plot_performance(
    model_scores,
    outname,
    human_scores=None,
):
    model_scores = sorted(model_scores, key=lambda x: x[1])
    print(model_scores)
    fig, ax = plt.subplots()
    ax.hlines(
        np.arange(len(model_scores)),
        0,
        [s[1] for s in model_scores],
        label="Model scores",
        color="#007acc",
        alpha=0.2,
        linewidth=5,
    )
    ax.plot(
        [s[1] for s in model_scores],
        np.arange(len(model_scores)),
        "o",
        markersize=5,
        color="#007acc",
        alpha=0.6,
    )

    scores = [s[1] for s in model_scores]

    range_frame(ax, np.array([0, max(scores)]), np.arange(len(model_scores)), pad=0.1)
    ax.set_yticks(np.arange(len(model_scores)))
    ax.set_yticklabels([s[0] for s in model_scores])

    if human_scores is not None:
        ax.vlines(
            np.mean(human_scores),
            -1,
            len(model_scores),
            label="Human mean",
            color="#00B945",
            alpha=0.6,
        )
        ax.vlines(
            np.max(human_scores),
            -1,
            len(model_scores),
            label="Human max",
            color="#FF9500",
            alpha=0.6,
        )
        ax.vlines(
            np.min(human_scores),
            -1,
            len(model_scores),
            label="Human min",
            color="#845B97",
            alpha=0.6,
        )

        ax.text(
            np.mean(human_scores),
            len(model_scores) - 0.5,
            "average human score",
            rotation=45,
            c="#00B945",
        )

        ax.text(
            np.max(human_scores),
            len(model_scores) - 0.5,
            "highest human score",
            rotation=45,
            c="#FF9500",
        )

        ax.text(
            np.min(human_scores),
            len(model_scores) - 0.5,
            "lowest human score",
            rotation=45,
            c="#845B97",
        )

    ax.set_xlabel("fraction of correct answers")
    # fig.tight_layout()
    fig.savefig(outname, bbox_inches="tight")


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)

    overall_model_scores_df = model_scores["overall"]
    overall_model_scores = []
    for model, df in overall_model_scores_df.items():
        if model not in MODELS_TO_PLOT:
            continue
        overall_model_scores.append((model, df["all_correct_"].mean()))

    combined_human_scores_model_scores_df = model_scores["human_aligned_combined"]
    combined_human_scores_model_scores = []
    for model, df in combined_human_scores_model_scores_df.items():
        if model not in MODELS_TO_PLOT:
            continue
        combined_human_scores_model_scores.append((model, df["all_correct_"].mean()))

    plot_performance(overall_model_scores, figures / "overall_performance.pdf")

    plot_performance(
        combined_human_scores_model_scores,
        figures / "human_subset_performance.pdf",
        all_human_scores,
    )

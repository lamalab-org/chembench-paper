import matplotlib.pyplot as plt
from paths import scripts, figures, data
import pickle
import seaborn as sns

plt.style.use(scripts / "lamalab.mplstyle")

from plotutils import range_frame, model_color_map
from utils import ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, ONE_COL_WIDTH_INCH
import pandas as pd

from glob import glob
import os
import json


human_dir = data / "human_scores"


def collect_human_scores():
    human_jsons = [
        c for c in glob(os.path.join(human_dir, "cl*.json")) if not "claude" in c
    ]

    scores = []

    for json_file in human_jsons:

        with open(json_file, "r") as handle:
            d = json.load(handle)

        if len(d["model_scores"]) > 100:
            if d["fraction_correct"] < 1:  # exclude this cheater
                scores.append(d["fraction_correct"])

    return scores


def plot(human_scores):
    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    sns.histplot(human_scores, kde=True, kde_kws={"cut": 0}, ax=ax)
    ax.set_xlabel("human score")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(figures / "human_score_distribution.pdf")


if __name__ == "__main__":
    human_scores = collect_human_scores()
    plot(human_scores)

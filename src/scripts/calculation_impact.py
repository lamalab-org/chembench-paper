import pandas as pd
import matplotlib.pyplot as plt
from paths import scripts, figures, data, output
import pickle
import numpy as np
from plotutils import range_frame

plt.style.use(scripts / "lamalab.mplstyle")

from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
)


def get_overall_performance(
    model_scores, requires_calculation: bool = False, suffix: str = ""
):
    original_model_scores_len = len(model_scores)
    if requires_calculation:
        model_scores = model_scores[model_scores["requires_calculation"]]
    else:
        model_scores = model_scores[~model_scores["requires_calculation"]]
    means = model_scores.groupby("model")["all_correct_"].mean()
    print(
        f"original model scores: {original_model_scores_len}, "
        f"filtered model scores: {len(model_scores)}"
    )
    # sort by mean
    means = means.sort_values()

    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH * 2)
    )

    ax.hlines(means.index, xmin=0, xmax=means, linewidth=5, alpha=0.2)
    ax.plot(means, means.index, "o", markersize=5, alpha=0.6)

    ax.set_xlabel("overall performance")
    ax.set_ylabel("model")

    range_frame(ax, means.fillna(0), np.arange(len(means)))

    fig.tight_layout()
    fig.savefig(figures / f"overall_performance_{suffix}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as f:
        model_scores = pickle.load(f)

    combined_scores = []
    for model, scores in model_scores["overall"].items():
        scores["model"] = model
        combined_scores.append(scores)
    combined_scores = pd.concat(combined_scores)
    # get_overall_performance(
    #     combined_scores, requires_calculation=True, suffix="requires_calculation"
    # )
    get_overall_performance(
        combined_scores, requires_calculation=False, suffix="no_calculation"
    )

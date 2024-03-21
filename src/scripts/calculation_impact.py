import pandas as pd
import matplotlib.pyplot as plt
from paths import scripts, figures, data, output
import pickle

plt.style.use(scripts / "lamalab.mplstyle")

from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
)


def get_overall_performance(
    model_scores, requires_calculation: bool = False, suffix: str = ""
):
    means = model_scores.groupby("name")["all_correct_"].mean()

    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    ax.hlines(means.index, xmin=0, xmax=means, linewidth=5, alpha=0.2)
    ax.plot(means, means.index, "o", markersize=5, alpha=0.6)

    ax.set_xlabel("overall performance")
    ax.set_ylabel("model")

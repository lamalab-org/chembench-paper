import matplotlib.pyplot as plt

from paths import scripts, figures, data

plt.style.use(scripts / "lamalab.mplstyle")

import pandas as pd

import numpy as np

from plotutils import range_frame
from utils import (
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
)
import pickle


def prepare_data_for_parallel_coordinates(model_score_dict):
    # the input contains a dictionary with the model name as key
    # the value is a dataframe with the performance per question
    # we also have a column "topic" which is the topic of the question
    # this will be the different axes in the parallel coordinates plot
    # the color is determined by the model

    all_aligned = []
    for model, frame in model_score_dict.items():
        frame["model"] = model
        all_aligned.append(frame)

    all_aligned = pd.concat(all_aligned)

    parallel_coordinates_data = all_aligned.pivot_table(
        index="model", columns="topic", values="all_correct_", aggfunc=np.mean
    )

    return parallel_coordinates_data


def plot_parallel_coordinates(parallel_coordinates_data, suffix=""):
    fig, ax = plt.subplots(
        1, 1, figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    parallel_coordinates_data = parallel_coordinates_data.T

    parallel_coordinates_data.plot(ax=ax, alpha=0.5)

    ax.set_ylabel("Performance")

    ax.set_xlabel("Topic")

    range_frame(
        ax,
        np.arange(len(parallel_coordinates_data.columns)),
        parallel_coordinates_data.columns,
    )

    fig.savefig(figures / f"parallel_coordinates_{suffix}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)["overall"]

    parallel_coordinates_data = prepare_data_for_parallel_coordinates(model_scores)

    plot_parallel_coordinates(parallel_coordinates_data, suffix="overall")

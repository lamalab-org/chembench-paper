from glob import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotutils import range_frame
from paths import output, scripts, figures, data
from loguru import logger

from utils import (
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
)

plt.style.use(scripts / "lamalab.mplstyle")


model_file_name_to_label = {
    "claude2": "Claude 2",
    "claude2-react": "Claude 2 + ReAct",
    "claude3": "Claude 3",
    "galatica_120b": "Galactica 120B",
    "gemini-pro-zero-T": "Gemini Pro",
    # "gemini-pro": "Gemini Pro",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    # "gpt-4": "GPT-4",
    "gpt-4-zero-T": "GPT-4",
    "gpt-35-turbo-react": "GPT-3.5 Turbo + ReAct",
    "llama-2-70b-chat": "LLaMA 70b",
    "mixtral-8x7b-instruct": "Mixtral 8x7b",
    "pplx-7b-chat": "Perplexity 7B chat",
    "pplx-7b-online": "Perplexity 7B online",
    "random_baseline": "Random baseline",
}


human_dir = output / "human_scores"
model_subset_dir = output / "human_subset_model_scores"
model_dir = output / "overall_model_scores"


def get_requires_calc_scores(model, output_dir, df, requires_cal: bool = True):
    name = model + ".json"
    with open(output_dir / name, "r") as handle:
        d = json.load(handle)

    scores = d["model_scores"]
    scores = pd.DataFrame(scores)

    merged = pd.merge(scores, df, left_on="question_name", right_on="name")

    score = merged[merged["requires_calculation"] == requires_cal]["all_correct"].mean()

    return score


def make_requires_calc_plot(output_dir, df, name):

    scores = []

    for model, label in model_file_name_to_label.items():
        for requires_cal in [True, False]:
            score = get_requires_calc_scores(
                model, output_dir=output_dir, df=df, requires_cal=requires_cal
            )
            scores.append(
                {
                    "model": model,
                    "name": label,
                    "score": score,
                    "requires_cal": requires_cal,
                }
            )

    frame = pd.DataFrame(scores)

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(0.8 * TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
        sharey=True,
        sharex=True,
    )

    w_cal = frame[frame["requires_cal"]]
    no_cal = frame[~frame["requires_cal"]]

    w_cal.sort_values(by="score", inplace=True)

    ax[0].hlines(
        w_cal["name"],
        0,
        w_cal["score"],
        label="Model scores",
        color="#007acc",
        alpha=0.2,
        linewidth=5,
    )
    ax[0].plot(
        w_cal["score"],
        w_cal["name"],
        "o",
        markersize=5,
        color="#007acc",
        alpha=0.6,
    )

    ax[1].hlines(
        no_cal["name"],
        0,
        no_cal["score"],
        label="Model scores",
        color="#007acc",
        alpha=0.2,
        linewidth=5,
    )

    ax[1].plot(
        no_cal["score"],
        no_cal["name"],
        "o",
        markersize=5,
        color="#007acc",
        alpha=0.6,
    )

    range_frame(
        ax[1],
        no_cal["score"],
        np.arange(len(no_cal["name"])),
    )

    range_frame(ax[0], w_cal["score"], np.arange(len(no_cal["name"])))
    fig.text(0.45, -0.15, "fraction correct")

    ax[0].set_title("w/ calculation")
    ax[1].set_title("w/o calculation")

    fig.savefig(figures / name, bbox_inches="tight")


if __name__ == "__main__":
    df = pd.read_pickle(data / "questions.pkl")

    make_requires_calc_plot(model_dir, df, "model_overall_cal.pdf")
    make_requires_calc_plot(model_subset_dir, df, "model_subset_cal.pdf")

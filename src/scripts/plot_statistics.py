from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
)
import matplotlib.pyplot as plt
from paths import output, figures, scripts

plt.style.use(scripts / "lamalab.mplstyle")
import pandas as pd
from glob import glob

import numpy as np
from plotutils import range_frame


def overall_question_count_barplot(df):
    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    topics, counts = df["topic"].value_counts().index, df["topic"].value_counts().values

    ax.hlines(topics, xmin=0, xmax=topics, linewidth=5, alpha=0.2)
    ax.plot(counts, topics, "o", markersize=5, alpha=0.6)
    ax.set_xscale("log")
    range_frame(ax, counts, np.arange(len(topics)))

    ax.set_xlabel("Number of Questions")

    fig.tight_layout()

    fig.savefig(figures / "question_count_barplot.pdf", bbox_inches="tight")


def question_count_barplot(df):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
        sharex=True,
    )

    df_mcq = df[df["is_classification"]]
    df_not_mcq = df[~df["is_classification"]]
    print(len(df_mcq), len(df_not_mcq))
    topics_mcq, counts_mcq = (
        df_mcq["topic"].value_counts().index,
        df_mcq["topic"].value_counts().values,
    )
    topics_general, counts_general = (
        df_not_mcq["topic"].value_counts().index,
        df_not_mcq["topic"].value_counts().values,
    )
    # ensure general topics are in the same order as mcq topics. Sort keys and values accordingly
    # by the order of the keys in topics_mcq
    print(topics_general, topics_mcq)
    sort_idx = np.argsort(topics_general)
    print(sort_idx)
    topics_general = topics_general[sort_idx]
    counts_general = counts_general[sort_idx]

    all_counts = np.concatenate([counts_mcq, counts_general])

    ax[0].hlines(topics_mcq, xmin=0, xmax=topics_mcq, linewidth=5, alpha=0.2)
    ax[0].plot(counts_mcq, topics_mcq, "o", markersize=5, alpha=0.6)
    ax[0].set_xscale("log")
    range_frame(ax[0], all_counts, np.arange(len(topics_general)))

    ax[1].hlines(topics_general, xmin=0, xmax=topics_general, linewidth=5, alpha=0.2)
    ax[1].plot(counts_general, topics_general, "o", markersize=5, alpha=0.6)
    ax[1].set_xscale("log")
    range_frame(ax[1], all_counts, np.arange(len(topics_general)))

    ax[0].set_xlabel("Number of Questions")

    fig.savefig(
        figures / "question_count_barplot_mcq_vs_general.pdf", bbox_inches="tight"
    )


def plot_question_statistics():
    df = pd.read_pickle(output / "questions.pkl")
    overall_question_count_barplot(df)
    question_count_barplot(df)


if __name__ == "__main__":
    plot_question_statistics()

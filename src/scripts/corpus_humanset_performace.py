import matplotlib.pyplot as plt
import numpy as np
from paths import data, figures, scripts
from plotutils import range_frame
from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
)
import re

plt.style.use(scripts / "lamalab.mplstyle")


def plot_corpuses(complete_scores: dict, human_aligned_scores: dict, outname):
    # Get models sorted by complete set performance
    models = sorted(
        complete_scores.keys(), key=lambda x: complete_scores[x], reverse=True
    )
    # filter out models starting with a number
    models = [model for model in models if not re.match(r"^\d", model)]
    complete_accuracies = [complete_scores[model] for model in models]
    human_aligned_accuracies = [human_aligned_scores[model] for model in models]

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    # Set width of bars and positions of the bars
    bar_width = 0.25
    x = np.arange(len(models))

    # Create bars
    ax.bar(
        x - bar_width / 2,
        complete_accuracies,
        bar_width,
        label="ChemBench",
        color="#a9dce3",
        alpha=0.9,
    )  # 007acc
    ax.bar(
        x + bar_width / 2,
        human_aligned_accuracies,
        bar_width,
        label="ChemBench-Mini",
        color="#7689de",
        alpha=0.9,
    )  # cc0000

    y_data = complete_accuracies + human_aligned_accuracies
    x_data = np.arange(len(models))
    range_frame(ax, x_data, np.array(y_data), pad=0.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("accuracy", fontsize=12)
    # ax.set_xlabel('model', fontsize=12)
    # ax.set_title('Model Performance Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    # legend in two colum so that not cut gpt-4
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    plt.savefig(outname, bbox_inches="tight", format="pdf")
    plt.close()


def compile_model_scores(results_dict):
    """Get scores as a dict of model: score pairs."""
    model_scores = []
    for model, df in results_dict.items():
        accuracy = df["all_correct_"].mean()
        model_scores.append((model, accuracy))
    return dict(model_scores)


if __name__ == "__main__":
    import pickle

    with open(data / "model_score_dicts.pkl", "rb") as handle:
        overall_scores = pickle.load(handle)

    complete_scores = compile_model_scores(overall_scores["overall"])
    human_aligned_scores = compile_model_scores(
        overall_scores["human_aligned_combined"]
    )

    plot_corpuses(
        complete_scores, human_aligned_scores, figures / "corpus_human_comparison.pdf"
    )

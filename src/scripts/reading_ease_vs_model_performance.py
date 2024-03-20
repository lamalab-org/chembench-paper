import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from paths import scripts, output, figures, data
import pickle

from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
)
import seaborn as sns

from plotutils import range_frame, model_color_map

plt.style.use(scripts / "lamalab.mplstyle")


def align_model_scores_reading_ease(model_scores, reading_ease_frame):
    out_frame = pd.merge(
        model_scores, reading_ease_frame, left_on="name", right_on="name"
    )
    return out_frame


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)["overall"]

    reading_ease_frame = pd.read_pickle(output / "reading_ease.pkl")

    all_aligned = []
    for model, frame in model_scores.items():
        aligned = align_model_scores_reading_ease(frame, reading_ease_frame)
        aligned["model"] = model
        all_aligned.append(aligned)

    all_aligned = pd.concat(all_aligned)

    fig, ax = plt.subplots(
        1, 1, figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    relevant_models = [
        "gpt4",
        "claude2",
        "claude3",
        "llama70b",
        "gemini_pro",
        "galactica_120b",
    ]

    all_aligned = all_aligned[all_aligned["model"].isin(relevant_models)]

    sns.violinplot(
        data=all_aligned,
        x="all_correct_",
        y="flesch_kincaid_reading_ease",
        hue="model",
        ax=ax,
        palette=model_color_map,
        cut=0,
    )

    # change labels for the colors
    # to be more readable
    handles, labels = ax.get_legend_handles_labels()
    name_replace = {
        "gpt4": "GPT-4",
        "claude2": "Claude 2",
        "claude3": "Claude 3",
        "llama70b": "Llama 70B",
        "gemini_pro": "Gemini Pro",
        "galactica_120b": "Galactica 120B",
    }
    labels = [name_replace[label] for label in labels]
    ax.legend(handles, labels, ncol=3, bbox_to_anchor=(0.8, 1.2))

    ax.set_xlabel("completly correct")
    ax.set_ylabel("Flesch-Kincaid Reading Ease")

    range_frame(
        ax,
        np.array([-0.4, 1.4]),
        np.array(
            [
                all_aligned["flesch_kincaid_reading_ease"].min(),
                all_aligned["flesch_kincaid_reading_ease"].max(),
            ]
        ),
    )

    fig.tight_layout()

    fig.savefig(figures / "reading_ease_vs_model_performance.pdf", bbox_inches="tight")

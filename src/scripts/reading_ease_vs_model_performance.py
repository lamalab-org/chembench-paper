import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from paths import scripts, output, figures, data
import pickle
import textacy
from textacy import text_stats as ts
from utils import (
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
)
import seaborn as sns
from definitions import MODELS_TO_PLOT
from plotutils import range_frame, model_color_map
from chembench.utils import (
    remove_ce,
    remove_pu,
    remove_math,
    remove_smiles,
    remove_rxnsmiles,
)
import spacy

spacy.cli.download("en_core_web_sm")


plt.style.use(scripts / "lamalab.mplstyle")


def get_reading_ease(df):
    documents = []
    for question in df["question"]:
        doc = textacy.make_spacy_doc(
            remove_smiles(
                remove_math(remove_pu(remove_rxnsmiles(remove_ce(question))))
            ),
            lang="en_core_web_sm",
        )
        documents.append(doc)

    flesch_kincaid_reading_ease = []

    for doc in documents:
        flesch_kincaid_reading_ease.append(ts.flesch_reading_ease(doc))

    reading_ease_frame = pd.DataFrame(
        {"flesch_kincaid_reading_ease": flesch_kincaid_reading_ease, "name": df["name"]}
    )

    reading_ease_frame.to_pickle(output / "reading_ease.pkl")
    return reading_ease_frame


def align_model_scores_reading_ease(model_scores, reading_ease_frame):
    out_frame = pd.merge(
        model_scores, reading_ease_frame, left_on="name", right_on="name"
    )
    return out_frame


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)["overall"]

    df = pd.read_pickle(data / "questions.pkl")

    reading_ease_frame = get_reading_ease(df)

    all_aligned = []
    for model, frame in model_scores.items():
        aligned = align_model_scores_reading_ease(frame, reading_ease_frame)
        aligned["model"] = model
        all_aligned.append(aligned)

    all_aligned = pd.concat(all_aligned)

    fig, ax = plt.subplots(
        1, 1, figsize=(TWO_COL_WIDTH_INCH * 0.8, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    all_aligned = all_aligned[all_aligned["model"].isin(MODELS_TO_PLOT)]

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
    # name_replace = {
    #     "gpt4": "GPT-4",
    #     "claude2": "Claude 2",
    #     "claude3": "Claude 3",
    #     "llama70b": "LLaMA 70B",
    #     "gemini_pro": "Gemini Pro",
    #     "galactica_120b": "Galactica 120B",
    # }
    # labels = [name_replace[label] for label in labels]
    ax.legend(handles, labels, ncol=3, bbox_to_anchor=(0.8, 1.2))

    ax.set_xlabel("fraction correct")
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

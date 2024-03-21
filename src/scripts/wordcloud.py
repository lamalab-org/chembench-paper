from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH
from paths import figures, output, data, scripts
import textacy
from textacy import text_stats as ts
from chembench.utils import (
    remove_ce,
    remove_pu,
    remove_math,
    remove_smiles,
    remove_rxnsmiles,
)
import numpy as np
from plotutils import range_frame

plt.style.use(scripts / "lamalab.mplstyle")


def create_wordcloud(df):
    all_formatted = " ".join(df["formatted"].values)
    all_formatted = (
        all_formatted.replace("Question:", "")
        .replace("Answer:", "")
        .replace("\ce", "")
        .replace("\pu", "")
        .replace("[START_SMILES]", "")
        .replace("[END_SMILES]", "")
        .strip()
    )
    wordcloud = WordCloud(
        background_color="white",
        relative_scaling=0.01,
        collocations=False,
        min_word_length=3,
    ).generate(all_formatted)
    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    ax.imshow(wordcloud, interpolation="bilinear")
    fig.axes[0].axis("off")
    fig.tight_layout()

    fig.savefig(figures / "wordcloud.pdf", bbox_inches="tight")


def plot_reading_ease(df):
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

    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    counts, edges, _ = ax.hist(flesch_kincaid_reading_ease)
    ax.set_xlabel("Flesch-Kincaid reading ease")
    ax.set_ylabel("number of questions")
    range_frame(ax, edges, counts)
    fig.tight_layout()
    fig.savefig(figures / "flesch_kincaid_reading_ease.pdf", bbox_inches="tight")

    with open(output / "flesch_kincaid_reading_ease.txt", "w") as f:
        f.write(
            f"\\num{{{np.mean(flesch_kincaid_reading_ease):.2f} \pm {np.std(flesch_kincaid_reading_ease):.2f}}}"
            + "\endinput"
        )

    reading_ease_frame = pd.DataFrame(
        {"flesch_kincaid_reading_ease": flesch_kincaid_reading_ease, "name": df["name"]}
    )

    reading_ease_frame.to_pickle(output / "reading_ease.pkl")


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_pickle(data / "questions.pkl")
    create_wordcloud(df)
    plot_reading_ease(df)

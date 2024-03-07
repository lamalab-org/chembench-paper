from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH
from paths import figures, output


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


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_pickle(output / "questions.pkl")
    create_wordcloud(df)

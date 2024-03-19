import pandas as pd
import matplotlib.pyplot as plt

from paths import scripts, output, figures, data
import pickle

from utils import ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, ONE_COL_WIDTH_INCH
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

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(
        data=all_aligned, x="reading_ease", y="all_correct", hue="model", ax=ax
    )

    fig.savefig(figures / "reading_ease_vs_model_performance.pdf", bbox_inches="tight")

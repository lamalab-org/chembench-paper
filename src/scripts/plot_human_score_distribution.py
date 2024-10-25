import matplotlib.pyplot as plt
from paths import scripts, figures
import seaborn as sns
import numpy as np

plt.style.use(scripts / "lamalab.mplstyle")

from plotutils import range_frame
from utils import ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, ONE_COL_WIDTH_INCH


from collect_human_scores import obtain_human_scores


human_scores_with_tools, human_scores_without_tools, all_human_scores = (
    obtain_human_scores()
)


def plot():
    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    sns.distplot(
        human_scores_with_tools,
        kde=True,
        rug=True,
        kde_kws={"cut": 0},
        ax=ax,
        bins=20,
        label="/w tools",
    )
    sns.distplot(
        human_scores_without_tools,
        kde=True,
        rug=True,
        kde_kws={"cut": 0},
        ax=ax,
        bins=20,
        label="w/o tools",
    )
    ax.set_xlabel("fraction correct")
    ax.set_ylabel("count")
    ax.legend()
    range_frame(ax, np.array(all_human_scores), np.array([0, 20]))
    plt.tight_layout()
    plt.savefig(figures / "human_score_distribution.pdf")


if __name__ == "__main__":
    plot()

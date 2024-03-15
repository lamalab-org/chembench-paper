import pandas as pd
import matplotlib.pyplot as plt
from paths import scripts, figures
import seaborn as sns
from utils import (
    obtain_chembench_repo,
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
)
from plotutils import range_frame, model_color_map
import os
import numpy as np

plt.style.use(scripts / "lamalab.mplstyle")


def plot():
    chemnbench = obtain_chembench_repo()
    gpt = pd.read_csv(
        os.path.join(
            chemnbench,
            "reports",
            "confidence_estimates",
            "confidence_estimates",
            "results_gpt-4.csv",
        )
    )
    gpt["model"] = "GPT-4"

    claude_2 = pd.read_csv(
        os.path.join(
            chemnbench,
            "reports",
            "confidence_estimates",
            "confidence_estimates",
            "results_claude2.csv",
        )
    )

    claude_2["model"] = "Claude 2"

    # claude_3 = pd.read_csv(
    #     os.path.join(
    #         chemnbench,
    #         "reports",
    #         "confidence_estimates",
    #         "confidence_estimates",
    #         "results_claude3.csv",
    #     )
    # )

    # claude_3["model"] = "Claude 3"

    all_results = pd.concat([gpt, claude_2]).reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    print(all_results)
    sns.violinplot(data=all_results, x="estimate", y="model", ax=ax)
    # sns.swarmplot(data=all_results, x="estimate", y="model", ax=ax)

    ax.set_xlabel("confidence estimate returned by the model")
    ax.set_ylabel("model")

    range_frame(ax, np.array([1, 5]), np.array([0, 2]))

    fig.savefig(figures / "confidence_score_distributions.pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot()

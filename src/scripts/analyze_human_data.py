from chembench.analysis import load_all_reports
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import os
from paths import scripts, data, output, figures
from scipy.stats import pearsonr, spearmanr
from plotutils import range_frame
from utils import (
    obtain_chembench_repo,
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
)

plt.style.use(scripts / "lamalab.mplstyle")


def make_human_performance_plots():
    chembench = obtain_chembench_repo()
    paths = glob(os.path.join(chembench, "reports/humans/**/*.json"), recursive=True)
    users = pd.read_csv(os.path.join(chembench, "reports/humans/users.csv"))

    dirs = list(set([os.path.dirname(p) for p in paths]))
    all_results = []
    for d in dirs:
        try:
            results = load_all_reports(d, os.path.join(chembench, "data"))
            userid = Path(d).stem
            user_info = users[users["id"] == userid]
            experience = user_info["experience"].values[0]
            highest_education = user_info["highestEducation"].values[0]
            if len(results) < 5:
                continue
            results["userid"] = userid
            results["experience"] = experience
            results["highest_education"] = highest_education
            all_results.append(results)
        except Exception as e:
            print(e)
            continue

    number_humans = len(all_results)

    with open(output / "number_experts.txt", "w") as f:
        f.write(f"{str(int(number_humans))}" + "\endinput")

    long_df = pd.concat(all_results).reset_index(drop=True)
    long_df["time_s"] = long_df[("time", 0)]

    total_hours = long_df["time_s"].sum() / 3600
    with open(output / "total_hours.txt", "w") as f:
        f.write(f"\SI{{{str(int(total_hours))}}}{{\hour}}" + "\endinput")

    make_timing_plot(long_df)
    make_human_time_score_plot(long_df)


def make_timing_plot(long_df):
    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    sns.violinplot(data=long_df, x="all_correct", y="time_s", cut=0, ax=ax)
    sns.stripplot(
        data=long_df,
        x="all_correct",
        y="time_s",
        color="black",
        ax=ax,
        alpha=0.3,
        size=2,
    )

    ax.set_yscale("log")
    ax.set_ylabel("time / s")
    ax.set_xlabel("all correct")

    range_frame(
        ax,
        np.array([-0.5, 1.5]),
        np.array([long_df["time_s"].min(), long_df["time_s"].max()]),
    )

    fig.savefig(figures / "human_timing.pdf", bbox_inches="tight")


def make_human_time_score_plot(long_df):
    grouped_by_user = (
        long_df[["all_correct", "time_s", "experience", "userid"]]
        .groupby("userid")
        .mean()
    )
    grouped_by_user.dropna(inplace=True)
    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    ax.scatter(grouped_by_user["experience"], grouped_by_user["all_correct"])
    ax.set_xlabel("experience in chemistry / y")
    ax.set_ylabel("fraction of correct answers")

    range_frame(
        ax,
        np.array([0, grouped_by_user["experience"].max()]),
        np.array(
            [grouped_by_user["all_correct"].min(), grouped_by_user["all_correct"].max()]
        ),
    )
    fig.tight_layout()
    fig.savefig(figures / "experience_vs_correctness.pdf", bbox_inches="tight")

    spearman = spearmanr(grouped_by_user["experience"], grouped_by_user["all_correct"])

    with open(output / "spearman_experience_score.txt", "w") as f:
        f.write(str(np.round(spearman.statistic, 2)) + "\endinput")

    with open(output / "spearman_experience_score_p.txt", "w") as f:
        f.write(str(np.round(spearman.pvalue, 2)) + "\endinput")


if __name__ == "__main__":
    make_human_performance_plots()

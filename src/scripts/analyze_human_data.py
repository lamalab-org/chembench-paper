from chembench.analysis import load_all_reports
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import os
from paths import scripts, output, figures
from scipy.stats import spearmanr
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
        alpha=0.2,
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
    grouped_by_user = grouped_by_user[
        grouped_by_user["all_correct"] != 1
    ]  # exclude cheater
    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    ax.scatter(grouped_by_user["experience"], grouped_by_user["all_correct"])
    ax.set_xlabel("experience in chemistry / y")
    ax.set_ylabel("fraction correct")

    range_frame(
        ax,
        np.array([0, grouped_by_user["experience"].max()]),
        np.array(
            [grouped_by_user["all_correct"].min(), grouped_by_user["all_correct"].max()]
        ),
    )
    # fig.tight_layout()
    fig.savefig(figures / "experience_vs_correctness.pdf", bbox_inches="tight")

    spearman = spearmanr(grouped_by_user["experience"], grouped_by_user["all_correct"])

    with open(output / "spearman_experience_score.txt", "w") as f:
        f.write(str(np.round(spearman.statistic, 2)) + "\endinput")

    with open(output / "spearman_experience_score_p.txt", "w") as f:
        f.write(str(np.round(spearman.pvalue, 2)) + "\endinput")

    # find educational background for each user and write this to files
    edu_dict = {}
    for user in grouped_by_user.index:
        edu_dict[user] = long_df[long_df["userid"] == user]["highest_education"].values[
            0
        ]

    with open(output / "num_human_phd.txt", "w") as f:
        f.write(str(list(edu_dict.values()).count("doctorate")) + "\endinput")

    with open(output / "num_human_master.txt", "w") as f:
        f.write(str(list(edu_dict.values()).count("MSc")) + "\endinput")

    with open(output / "num_human_bachelor.txt", "w") as f:
        f.write(str(list(edu_dict.values()).count("BSc")) + "\endinput")

    with open(output / "num_human_highschool.txt", "w") as f:
        f.write(str(list(edu_dict.values()).count("high-school")) + "\endinput")

    with open(output / "num_human_postdoc.txt", "w") as f:
        f.write(str(list(edu_dict.values()).count("post-doctorate")) + "\endinput")

    with open(output / "num_users_with_education_info.txt", "w") as f:
        # take those user for which the education info is one of the above categories
        users_with_education_info = [
            k
            for k, v in edu_dict.items()
            if v in ["doctorate", "MSc", "BSc", "high-school", "post-doctorate"]
        ]
        f.write(str(len(users_with_education_info)) + "\endinput")


if __name__ == "__main__":
    make_human_performance_plots()

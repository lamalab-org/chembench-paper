from chembench.analysis import load_all_reports, all_correct
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
from typing import Union
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

# tool 127 
# no tool 121

plt.style.use(scripts / "lamalab.mplstyle")

def get_joint_frame(response_file: Union[Path, str], questions_file: Union[Path, str]):
    """From the database, we pull the two main tables: questions and responses and save them into csvs.

    This function will return a pandas dataframe with the two tables merged.

    Args:
        response_file: path to responses.csv
        questions_file: path to questions.csv

    Returns:
        merged: pandas dataframe with the two tables merged
    """
    questions_frame = pd.read_csv(questions_file)
    response_frame = pd.read_csv(response_file)

    merged = pd.merge(questions_frame, response_frame, left_on="id", right_on="questionId")

    return merged

# responses_20240918_161121.csv

def make_human_performance_plots():
    chembench = obtain_chembench_repo()
    paths_tool = glob(os.path.join(chembench, "reports/humans/reports/tool-allowed/**/*.json"), recursive=True)
    paths_no_tool = glob(os.path.join(chembench, "reports/humans/reports/tool-disallowed/**/*.json"), recursive=True)
    users = pd.read_csv(os.path.join(chembench, "reports/humans/users_20240918_161121.csv"))

    dirs_tool = list(set([os.path.dirname(p) for p in paths_tool]))
    dirs_no_tool = list(set([os.path.dirname(p) for p in paths_no_tool]))
    all_results = []
    for d in dirs_tool + dirs_no_tool:
        try:
            results = load_all_reports(d, os.path.join(chembench, "data"))
            if len(results) > 80: 
                userid = Path(d).stem
                user_info = users[users["id"] == userid]
                experience = user_info["experience"].values[0]
                highest_education = user_info["highestEducation"].values[0]
                if len(results) < 5:
                    continue
                results["userid"] = userid
                results['num_results'] = len(results)
                results["experience"] = experience
                results["highest_education"] = highest_education
                if "tool-allowed" in d:
                    results["tool_allowed"] = True
                else:
                    results["tool_allowed"] = False
                all_results.append(results)
            else: 
                print(f'Skipping {d} due to too few results')
        except Exception as e:
            print(e)
            continue

    number_humans = len(users)

    with open(output / "number_experts.txt", "w") as f:
        f.write(f"{str(int(number_humans))}" + "\endinput")

    long_df = pd.concat(all_results).reset_index(drop=True)
    long_df['all_correct'] = long_df.apply(all_correct, axis=1)
    long_df["time_in_s"] = long_df[("time_s", 0)]

    total_hours = long_df["time_in_s"].sum() / 3600
    with open(output / "total_hours.txt", "w") as f:
        f.write(f"\SI{{{str(int(total_hours))}}}{{\hour}}" + "\endinput")

    make_timing_plot(long_df)
    make_human_time_score_plot(long_df)


def make_timing_plot(long_df):
    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    #sns.violinplot(data=long_df, x="all_correct", y="time_in_s", cut=0, ax=ax)
    ax.set_yscale("log")
    ax = sns.violinplot(data=long_df, x="all_correct", y="time_in_s", hue="tool_allowed", split=True, inner="quart", ax=ax)
  
    #ax.set_yscale("log")
    ax.set_ylabel("time / s")
    ax.set_xlabel("all correct")

    range_frame(
        ax,
        np.array([-1.5, 1.5]),
        np.array([long_df["time_in_s"].min(), long_df["time_in_s"].max()]),
    )

    sns.move_legend(ax, "upper left", title='tools allowed', bbox_to_anchor=(-0, 1.2))

    fig.tight_layout()

    fig.savefig(figures / "human_timing.pdf", bbox_inches="tight")


def make_human_time_score_plot(long_df):
    with_tools = long_df[long_df["tool_allowed"] == True]
    without_tools = long_df[long_df["tool_allowed"] == False]
    grouped_by_user_with_tools = (
        with_tools[["all_correct", "time_in_s", "experience", "userid"]]
        .groupby("userid")
        .mean()
    )
    grouped_by_user_without_tools = (
        without_tools[["all_correct", "time_in_s", "experience", "userid"]]
        .groupby("userid")
        .mean()
    )
    grouped_by_user_without_tools.dropna(inplace=True)
    grouped_by_user_with_tools.dropna(inplace=True)
    grouped_by_user = pd.concat([grouped_by_user_without_tools, grouped_by_user_with_tools])
    fig, ax = plt.subplots(
        1, 1, figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
    ax.scatter(grouped_by_user_without_tools["experience"], grouped_by_user_without_tools["all_correct"], label="without tools")
    ax.scatter(grouped_by_user_with_tools["experience"], grouped_by_user_with_tools["all_correct"], label="with tools")
    ax.set_xlabel("experience in chemistry / y")
    ax.set_ylabel("fraction correct")
  

    range_frame(
        ax,
        np.array([0, grouped_by_user["experience"].max()]),
        np.array(
            [grouped_by_user["all_correct"].min(), grouped_by_user["all_correct"].max()]
        ),
    )
    
    ax.legend(loc="upper left", bbox_to_anchor=(-0.1, 1.2))
    
    fig.tight_layout()
    fig.savefig(figures / "experience_vs_correctness.pdf", bbox_inches="tight")

    spearman = spearmanr(grouped_by_user["experience"], grouped_by_user["all_correct"])
    spearman_with_tool = spearmanr(grouped_by_user_with_tools["experience"], grouped_by_user_with_tools["all_correct"])
    spearman_without_tool = spearmanr(grouped_by_user_without_tools["experience"], grouped_by_user_without_tools["all_correct"])

    with open(output / "spearman_experience_score_with_tool.txt", "w") as f:
        f.write(str(np.round(spearman_with_tool.statistic, 2)) + "\endinput")

    with open(output / "spearman_experience_score_without_tool.txt", "w") as f:
        f.write(str(np.round(spearman_without_tool.statistic, 2)) + "\endinput")

    with open(output / "spearman_experience_score_with_tool_p.txt", "w") as f:
        f.write(str(np.round(spearman_with_tool.pvalue, 2)) + "\endinput")

    with open(output / "spearman_experience_score_without_tool_p.txt", "w") as f:
        f.write(str(np.round(spearman_without_tool.pvalue, 2)) + "\endinput")

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

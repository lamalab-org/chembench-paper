from chembench.analysis import load_all_reports
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import os
from paths import scripts, data, output
from plotutils import range_frame
from utils import (
    obtain_chembench_repo,
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
)

plt.style.use(scripts / "lamalab.mplstyle")


def make_human_performance_plots():
    chembench = obtain_chembench_repo()
    print(chembench)
    paths = glob(os.path.join(chembench, "reports/humans/**/*.json"), recursive=True)
    print(paths)
    dirs = list(set([os.path.dirname(p) for p in paths]))
    all_results = []
    for d in dirs:
        try:
            results = load_all_reports(d, "../data/")
            if len(results) < 5:
                continue
            all_results.append(results)
        except Exception as e:
            print(e)
            continue

    number_humans = len(all_results)

    with open(output / "number_experts.txt", "w") as f:
        f.write(f"\SI{{{str(int(number_humans))}}}{{\hour}}")

    long_df = pd.concat(all_results).reset_index(drop=True)
    long_df["time_s"] = long_df[("time", 0)]

    total_hours = long_df["time_s"].sum() / 3600
    with open(output / "total_hours.txt", "w") as f:
        f.write(str(int(total_hours)))

    make_timing_plot(long_df)


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

    fig.savefig(output / "human_timing.pdf", bbox_inches="tight")


if __name__ == "__main__":
    make_human_performance_plots()

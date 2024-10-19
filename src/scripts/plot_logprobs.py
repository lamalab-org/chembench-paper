import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from paths import figures, scripts
from utils import (
    TWO_COL_WIDTH_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    obtain_chembench_repo,
)
from plotutils import model_color_map
plt.style.use(scripts / "lamalab.mplstyle")


rename_dict = {"gpt": "GPT-4o", "llama3": "Llama-3.1-8B-Instruct"}


def process_json_data(json_data):
    return pd.DataFrame(
        [
            {
                "question_name": score["question_name"],
                "all_correct": score["all_correct"],
                "linear_probability": score["linear_probability"],
            }
            for score in json_data["model_scores"]
        ]
    )


def create_calibration_plot(df, num_bins):
    df = df.dropna(subset=["linear_probability"])
    probabilities = df["linear_probability"].values
    labels = df["all_correct"].values

    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(probabilities, bins) - 1

    mean_predicted_probs = []
    std_dev_probs = []
    fraction_positives = []
    bin_counts = []

    for i in range(num_bins):
        bin_mask = bin_indices == i
        bin_probs = probabilities[bin_mask]
        bin_labels = labels[bin_mask]

        if len(bin_probs) > 0:
            mean_predicted_probs.append(np.mean(bin_probs))
            std_dev_probs.append(np.std(bin_probs))
            fraction_positives.append(np.mean(bin_labels))
            bin_counts.append(len(bin_probs))
        else:
            mean_predicted_probs.append(np.nan)
            std_dev_probs.append(np.nan)
            fraction_positives.append(np.nan)
            bin_counts.append(0)

    return mean_predicted_probs, std_dev_probs, fraction_positives, bin_counts


def make_plot_of_calibration(merged_dicts, num_bins, suffix: str = ""):
    fig, ax = plt.subplots(
        1,
        2,  # Two subplots for two models
        figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
        sharex=True,
    )
    plt.subplots_adjust(wspace=0.5)

    for i, (model, df) in enumerate(merged_dicts.items()):
        mean_predicted_probs, std_dev_probs, fraction_positives, bin_counts = (
            create_calibration_plot(df, num_bins)
        )

        # Calibration curve
        ax[i].errorbar(
            mean_predicted_probs,
            fraction_positives,
            yerr=std_dev_probs,
            color=model_color_map[rename_dict[model]],
            marker="o",
            linestyle="-",
            capsize=0,
            capthick=1,
            elinewidth=1,
            markersize=4,
        )
        ax[i].plot(
            mean_predicted_probs,
            fraction_positives,
            color=model_color_map[rename_dict[model]],
            marker="o",
        )

        # Perfect calibration line
        ax[i].plot([0, 1], [0, 1], linestyle="--", color="gray")

        ax[i].set_title(rename_dict[model])
        if i == 0:
            ax[i].set_ylabel("fraction of positives")
        ax[i].set_ylim(0, 1)
        ax[i].set_xlim(0, 1)

        # Add histogram
        bin_edges = np.linspace(0, 1, num_bins + 1)
        ax_twin = ax[i].twinx()
        ax_twin.bar(
            bin_edges[:-1],
            bin_counts,
            width=np.diff(bin_edges) - 0.2 * (np.diff(bin_edges)),
            alpha=0.3,
            color=model_color_map[rename_dict[model]],
            align="edge",
        )
        if i == 1:
            ax_twin.set_ylabel("Count")

        # Calculate and display ECE
        ece = np.mean(
            np.abs(np.array(fraction_positives) - np.array(mean_predicted_probs))
        )
        ax[i].text(
            0.05,
            0.95,
            f"ECE: {ece:.4f}",
            transform=ax[i].transAxes,
            verticalalignment="top",
        )

    # add xlabel
    fig.text(0.5, 0.01, "predicted probability", ha="center")

    fig.tight_layout()
    fig.set_facecolor("w")
    fig.savefig(
        os.path.join(figures, f"log_probs_calibration_plot_{suffix}.pdf"),
        bbox_inches="tight",
    )
    logger.info(f"Saved plot: calibration_plot_{suffix}.pdf")


def plot():
    chembench = obtain_chembench_repo()
    llama_json_path = os.path.join(
        chembench,
        "reports",
        "log_prob_llama3.1-8b_local",
        "log_probs.json",
    )
    gpt_json_path = os.path.join(
        chembench,
        "reports",
        "log_probs_gpt_4o",
        "log_probs.json",
    )

    json_file_paths = {
        "llama3": llama_json_path,
        "gpt": gpt_json_path,
    }

    # Add path to question_tags.csv file
    tags_file = os.path.join(
        chembench,
        "reports",
        "log_probs_gpt_4o",
        "question_tags.csv",
    )

    try:
        # Load question tags
        tags_df = pd.read_csv(tags_file)

        merged_dicts = {}
        for model, json_path in json_file_paths.items():
            with open(json_path, "r") as f:
                json_data = json.load(f)
            df = process_json_data(json_data)

            # Merge with tags and filter out "Chemical Preference" topic
            df = pd.merge(
                df,
                tags_df,
                left_on="question_name",
                right_on="question_name",
                how="left",
            )
            df = df[df["topic"] != "Chemical Preference"]

            df["model"] = rename_dict[model]
            merged_dicts[model] = df

        make_plot_of_calibration(merged_dicts, num_bins=5, suffix="overall_filtered")
        logger.info(
            "Successfully completed analysis and plotting (excluding Chemical Preference topic)"
        )
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        logger.exception("Error details:")


if __name__ == "__main__":
    plot()

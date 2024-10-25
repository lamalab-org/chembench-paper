import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from loguru import logger
from glob import glob
from typing import Union
from pathlib import Path
import seaborn as sns
from paths import scripts, output, figures, data, static
from plotutils import range_frame, model_color_map
from utils import (
    obtain_chembench_repo,
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH
)

chembench_repo = obtain_chembench_repo()
outpath = output / "model_confidence_performance"
os.makedirs(outpath, exist_ok=True)

BASE_PATH = chembench_repo

plt.style.use(scripts / "lamalab.mplstyle")

rename_dict = {
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "claude3": "Claude-3.5 (Sonnet)",
    "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct"
}

reports_dir = os.path.join(BASE_PATH, "reports")

confidence_files ={
    "gpt-4": os.path.join(reports_dir, "confidence_estimates", "confidence_estimates", "results_gpt-4.csv"),
    "gpt-4o": os.path.join(reports_dir, "confidence_estimates", "confidence_estimates", "results_gpt-4o.csv"),
    "claude3": os.path.join(reports_dir, "confidence_estimates", "confidence_estimates", "results_claude3.5.csv"),
    "llama3.1-8b-instruct": os.path.join(reports_dir, "confidence_estimat_local", "confidence_estimates", "results_llama3.1-8b-instruct.csv"),
}

def process_json_data(json_data):
    return pd.DataFrame([
        {
            'question_name': score['question_name'],
            'all_correct_': score['all_correct']
        } for score in json_data['model_scores']
    ])

def join_confidence_and_performance(json_file_paths):
    merged_dicts = {}
    for model, json_path in json_file_paths.items():
        confidence_file = confidence_files.get(model)
        if confidence_file is None:
            logger.warning(f"Confidence file not found for {model}: {confidence_file}")
            continue

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        performance_df = process_json_data(json_data)
        confidence_df = pd.read_csv(confidence_file)

        confidence_df = confidence_df[~confidence_df['name'].str.startswith('preference-preferences_')]

        confidence_df['model'] = rename_dict[model]

        df = pd.merge(confidence_df, performance_df, left_on='name', right_on='question_name')
        merged_dicts[model] = df

    return merged_dicts

def make_plot_of_confidence_vs_performance(merged_dicts):
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH), sharex=True, sharey=True)
    axs = axs.ravel()

    fig.patch.set_facecolor('white')

    for i, (model, df) in enumerate(merged_dicts.items()):
        ax = axs[i]
        ax.set_facecolor('white')

        df["all_correct_"] = df["all_correct_"].astype(float)

        average_performance = df.groupby("estimate")["all_correct_"].mean()
        stdev = df.groupby("estimate")["all_correct_"].apply(sem)

        ax.plot(
            average_performance.index,
            average_performance,
            color=model_color_map[rename_dict[model]],
            marker="o",
            label="Average Performance"
        )

        ax.errorbar(
            average_performance.index,
            average_performance,
            yerr=stdev,
            fmt="none",
            color=model_color_map[rename_dict[model]],
            alpha=0.3
        )

        counts, _ = np.histogram(df["estimate"], bins=5, range=(1, 5))
        max_count = counts.max()
        ax.bar(
            average_performance.index,
            counts / max_count * 0.3,
            alpha=0.3,
            color=model_color_map[rename_dict[model]],
            width=0.8,
           # label="Distribution"
        )

        # add diagonal line
        ax.plot([1, 5], [0, 1], color="black", linestyle="--", alpha=0.5)
        range_frame(ax, np.array([1,5]), np.array([0,1]))

        ax.set_title(rename_dict[model])

        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.legend(loc='upper left')

    fig.text(0.5, 0.00, 'confidence estimate', ha='center')
    fig.text(-0.01, 0.5, 'fraction correct', va='center', rotation='vertical')

    fig.tight_layout()

    plt.savefig(
        figures / "model_confidence_performance.pdf", format="pdf", bbox_inches="tight"
    )

    plt.close(fig)
    logger.info(f"Saved confidence vs performance plot as PDF in {static}")

if __name__ == "__main__":
    json_file_paths = {
        "gpt-4": os.path.join(chembench_repo, "reports", "gpt-4", "gpt-4.json"),
        "gpt-4o": os.path.join(chembench_repo, "reports", "gpt-4o", "gpt-4o.json"),
        "claude3": os.path.join(chembench_repo, "reports", "claude3.5", "claude3.5.json"),
        "llama3.1-8b-instruct": os.path.join(chembench_repo, "reports", "llama3.1-8b-instruct", "llama3.1-8b-instruct.json")
    }

    try:
        merged_dicts = join_confidence_and_performance(json_file_paths)
        print(merged_dicts)
        make_plot_of_confidence_vs_performance(merged_dicts)
        logger.info("Successfully completed analysis and saved plot as PDF")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        logger.exception("Error details:")


    df_questions = pd.read_pickle(data / "questions.pkl")


    subsets = [
        "is_point_group",
        "is_number_of_isomers",
        "is_number_nmr_peaks",
        "is_gfk",
        "is_dai",
        "is_pictograms",
        "is_name",
        "is_organic_reactivity",
        "is_electron_counts",
        "is_chemical_compatibility",
        "is_materials_compatibility",
        "is_oup",
        "is_olympiad",
        "is_toxicology",
        "is_polymer_chemistry",
    ]
    suffix = 'overall'
    for subset in subsets:
        for model, df in merged_dicts.items():
            df = pd.merge(df, df_questions, left_on='question_name', right_on='name')
            relevant_model_performance = df[df[subset]]
            correct = relevant_model_performance[
                relevant_model_performance["all_correct_"].astype(bool)
            ]
            incorrect = relevant_model_performance[
                ~relevant_model_performance["all_correct_"].astype(bool)
            ]
            num_correct = len(correct)
            num_incorrect = len(incorrect)

            average_confidence_correct = correct["estimate"].mean()
            average_confidence_incorrect = incorrect["estimate"].mean()

            with open(outpath / f"{model}_{subset}_num_correct_{suffix}.txt", "w") as f:
                f.write(f"{num_correct}" + "\endinput")

            with open(outpath / f"{model}_{subset}_num_incorrect_{suffix}.txt", "w") as f:
                f.write(f"{num_incorrect}" + "\endinput")

            with open(
                outpath / f"{model}_{subset}_average_confidence_correct_{suffix}.txt", "w"
            ) as f:
                rounded = np.round(average_confidence_correct, 2)
                f.write(f"{rounded}" + "\endinput")

            with open(
                outpath / f"{model}_{subset}_average_confidence_incorrect_{suffix}.txt", "w"
            ) as f:
                rounded = np.round(average_confidence_incorrect, 2)
                f.write(f"{rounded}" + "\endinput")

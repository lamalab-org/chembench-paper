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
from plotutils import range_frame
from utils import (
    obtain_chembench_repo,
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
)

chembench_repo = obtain_chembench_repo()

TWO_COL_WIDTH_INCH = 7.0
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = 5.5  

BASE_PATH = chembench_repo
DATA_PATH = os.path.join(BASE_PATH, "reports", "confidence_estimates", "confidence_estimates")

plt.style.use(scripts / "lamalab.mplstyle")

rename_dict = {
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "claude3.5": "Claude 3.5",
    "llama3.1-8b-instruct": "llama3-8b-8192"
}

model_color_map = {
    "GPT-4": '#FCCD2A',
    "GPT-4o": '#A04747',
    "Claude 3.5": '#347928',
    "llama3-8b-8192": '#F39C12'
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
        confidence_file = os.path.join(DATA_PATH, f"results_{model}.csv")
        if not os.path.exists(confidence_file):
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
            label="Estimate Distribution"
        )

        ax.set_ylim(0, 1)
        ax.set_xlim(1, 6)
        ax.set_title(rename_dict[model], fontsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.plot([1, 6], [0, 1], color='gray', linestyle='--', alpha=0.5, label="Ideal Correlation")

        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.legend(fontsize=8, loc='upper left')

    fig.text(0.5, 0.02, 'Estimate', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Fraction Correct', va='center', rotation='vertical', fontsize=14)

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    plt.savefig(
        static / "model_confidence_performance.pdf", format="pdf", dpi=300, bbox_inches="tight"
    )
    
    plt.close(fig)
    logger.info(f"Saved confidence vs performance plot as PDF in {static}")

if __name__ == "__main__":
    json_file_paths = {
        "gpt-4": os.path.join(chembench_repo, "reports", "gpt-4", "gpt-4.json"),
        "gpt-4o": os.path.join(chembench_repo, "reports", "gpt-4o", "gpt-4o.json"),
        "claude3.5": os.path.join(chembench_repo, "reports", "claude3.5", "claude3.5.json"),
        "llama3.1-8b-instruct": os.path.join(chembench_repo, "reports", "llama3-8b-instruct", "llama3-8b-instruct.json")
    }

    try:
        merged_dicts = join_confidence_and_performance(json_file_paths)
        make_plot_of_confidence_vs_performance(merged_dicts)
        logger.info("Successfully completed analysis and saved plot as PDF")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        logger.exception("Error details:")
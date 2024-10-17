import pickle
from paths import output, scripts, figures

import json
import os

from loguru import logger

from chembench.analysis import (
    all_correct,
    load_all_reports,
)
from chembench.types import PathType

from collections import defaultdict

from glob import glob
import os
from pathlib import Path
from utils import obtain_chembench_repo
from paths import output
from loguru import logger

import json
import numpy as np
import matplotlib.pyplot as plt
from plotutils import range_frame
from paths import output, scripts, figures
from loguru import logger
import pandas as pd

model_file_name_to_label = {
    # "claude2": "Claude 2",
    # "claude3": "Claude 3",
    "claude3.5": "Claude 3.5 Sonnet",
    # "command-r+": "Command R+",
    "galatica_120b": "Galactica 120B",
    # "gemini-pro": "Gemini Pro",
    # "gemma-1-1-7b-it": "Gemma 1.1 7B Instruct",
    # "gemma-1-1-7b-it-T-one": "Gemma 1.1 7B Instruct T-One",
    # "gemma-2-9b-it": "Gemma 2 9B Instruct",
    # "gemma-2-9b-it-T-one": "Gemma 2 9B Instruct T-One",
    "gpt-3.5-turbo": "GPT-3.5-Turbo",
    # "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "o1": "OpenAI o1",
    # "llama2-70b-chat": "Llama 2 70B Chat",
    # "llama3-70b-instruct": "Llama 3 70B Instruct",
    # "llama3-70b-instruct-T-one": "Llama 3 70B Instruct T-One",
    # "llama3-8b-instruct": "Llama 3 8B Instruct",
    # "llama3-8b-instruct-T-one": "Llama 3 8B Instruct T-One",
    "llama3.1-405b-instruct": "Llama 3.1 405B Instruct",
    "llama3.1-70b-instruct": "Llama 3.1 70B Instruct",
    # "llama3.1-70b-instruct-T-one": "Llama 3.1 70B Instruct T-One",
    "llama3.1-8b-instruct": "Llama 3.1 8B Instruct",
    # "llama3.1-8b-instruct-T-one": "Llama 3.1 8B Instruct T-One",
    # "mixtral-8x7b-instruct": "Mixtral 8x7B Instruct",
    # "mixtral-8x7b-instruct-T-one": "Mixtral 8x7B Instruct T-One",
    "mistral-large-2-123b": "Mistral Large 2 123B",
    "paper-qa": "Paper QA",
    # "phi-3-medium-4k-instruct": "Phi 3 Medium 4K Instruct",
    # "random_baseline": "Random Baseline"
}

def get_human_scored_questions(human_reports_dir: PathType):
    """
    This function retrieves all human scored questions from a given directory.
    Args:
        human_reports_dir (Path, str): The directory where the human scored questions are stored.
    Returns:
        questions (dict): A dictionary where the keys are the names of the questions and the values are lists of directories where the questions are found.
    """
    questions = defaultdict(list)
    for file in Path(human_reports_dir).rglob("*.json"):
        questions[file.stem].append(file.parent)
    return dict(questions)

def get_human_scored_questions_with_at_least_n_scores(human_reports_dir: PathType, n: int):
    """
    Retrieve human scored questions with at least a specified number of scores.
    Args:
        human_reports_dir (PathType): The directory where the human scored questions are stored.
        n (int): The minimum number of scores required.
    Returns:
        List: A list of questions with at least n scores.
    """
    questions = get_human_scored_questions(human_reports_dir)
    return [k for k, v in questions.items() if len(v) >= n]

def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def collect_human_results(human_path, datafolder, min_reports=200) -> list:
    tool_allowed_path = os.path.join(human_path, "tool-allowed")
    tool_disallowed_path = os.path.join(human_path, "tool-disallowed")

    tools_dirs = set(list_directories(tool_allowed_path))
    notools_dirs = set(list_directories(tool_disallowed_path))

    common_dirs = tools_dirs.intersection(notools_dirs)

    # Read JSON files from common directories
    human_scores = []
    for common_dir in common_dirs:
        tools_paths = os.path.join(tool_allowed_path, common_dir)
        notools_paths = os.path.join(tool_allowed_path, common_dir)

        tools_df = load_all_reports(tools_paths, datafolder)
        notools_df = load_all_reports(notools_paths, datafolder)

        # df = pd.concat([tools_df, notools_df], ignore_index=True)
        # if len(df) < min_reports:
            # continue

        df = notools_df

        model_scores = []
        all_correct_count = 0
        for i, row in df.iterrows():
            question_name = row[("name", 0)]
            metric = int(all_correct(row))
            all_correct_count += metric
            metrics = row["metrics"].to_dict()
            metrics["question_name"] = question_name
            metrics["all_correct"] = metric
            metrics["keywords"] = row[("keywords", 0)]
            model_scores.append(metrics)

        fraction_correct = all_correct_count / len(df) if len(df) > 0 else 0
        if len(df) == 0:
            raise ValueError(f"Error")
        human_scores.append({
            "fraction_correct": fraction_correct,
            "model_scores": model_scores,
        })

    return human_scores

def combine_scores_for_model(folder, datafolder, human_baseline_folder=None, min_human_responses: int = 4):
    try:
        df = load_all_reports(folder, datafolder)
        if df.empty:
            raise ValueError("No data loaded from reports")

        if human_baseline_folder is not None:
            relevant_questions = get_human_scored_questions_with_at_least_n_scores(human_baseline_folder, min_human_responses)
            df = df[df[("name", 0)].isin(relevant_questions)]

EXPECTED_HUMAN_BASELINE_REPORTS = 248
EXPECTED_GENERAL_REPORTS = 2854

if human_baseline_folder:
    if len(df) != EXPECTED_HUMAN_BASELINE_REPORTS:
        logger.error(f"ERROR: some reports are missing with human baseline")
elif len(df) != EXPECTED_GENERAL_REPORTS:
    logger.error(f"ERROR: some general reports are missing")

        logger.info(f"Loaded {len(df)} rows of data")
        model_scores = []
        all_correct_count = 0
        for i, row in df.iterrows():
            question_name = row[("name", 0)]
            metric = int(all_correct(row))
            all_correct_count += metric
            metrics = row["metrics"].to_dict()
            metrics["question_name"] = question_name
            metrics["all_correct"] = metric
            metrics["keywords"] = row[("keywords", 0)]
            model_scores.append(metrics)

        fraction_correct = all_correct_count / len(df) if len(df) > 0 else 0

        return {
            "fraction_correct": fraction_correct,
            "model_scores": model_scores,
        }

    except Exception as e:
        logger.error(f"Error in combine_scores_for_model: {str(e)}")
        raise ValueError
        return {"error": str(e), "fraction_correct": 0, "model_scores": []}

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    reports = glob(
        os.path.join(chembench_repo, "reports",
                     "**", "reports", "**", "*.json")
    )
    overall_files = glob(
        os.path.join(chembench_repo, "reports", "**", "*.json")
    )

    models = list(set([Path(p).parent for p in reports]))
    logger.info(f"Found {len(models)} reports")
    logger.debug(f"First 5 reports: {models[:5]}")
    outpath = os.path.join(output, "overall_model_scores")
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    datafolder = os.path.join(chembench_repo, "data")
    human_baseline_folder = os.path.join(chembench_repo, "reports", "humans")

    model_overall_scores = []
    overall_scores = []
    overall_human_scores = []
    model_scores_human_subset = []
    for i, file in enumerate(models):
        try:
            p = Path(file).parts[-3]
            logger.info(f"Running for model {p}")
            human_results = combine_scores_for_model(file.parent, datafolder, human_baseline_folder)
            model_scores_human_subset.append({
                "model": p,
                "fraction_correct": human_results["fraction_correct"]
            })
            model_results = combine_scores_for_model(file.parent, datafolder)
            model_overall_scores.append({
                "model": p,
                "fraction_correct": model_results["fraction_correct"]
            })
            overall_scores.append({
                "model": p,
                "scores": model_results,
            })
            overall_human_scores.append({
                "model": p,
                "scores": human_results,
            })
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            raise ValueError

    model_overall_scores_dict = {item["model"]: item["fraction_correct"] for item in model_overall_scores}
    model_scores_human_subset_dict = {item["model"]: item["fraction_correct"] for item in model_scores_human_subset}
    
    model_overall_scores = [(model_file_name_to_label[k], v) for k, v in model_overall_scores_dict.items() if k in model_file_name_to_label]
    model_scores_human_subset = [(model_file_name_to_label[k], v) for k, v in model_scores_human_subset_dict.items() if k in model_file_name_to_label]

    human_path = os.path.join(chembench_repo, "reports", "humans", "reports")

    logger.info("Collecting human with tool disallowed results")
    human_subset = collect_human_results(human_path, datafolder)
    
    # Add human scores to the lists
    model_overall_scores.append(("Humans", human_subset))
    model_scores_human_subset.append(("Humans", human_subset))

    logger.info("Saving results into .pkl files")
    with open(os.path.join(output, "t_overall_scores.pkl"), 'wb') as f:
        pickle.dump(overall_scores, f)

    with open(os.path.join(output, "t_overall_human_scores.pkl"), 'wb') as f:
        pickle.dump(overall_human_scores, f)
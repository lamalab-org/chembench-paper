import json
import os

from loguru import logger

from chembench.analysis import (
    all_correct,
    get_human_scored_questions_with_at_least_n_scores,
    load_all_reports,
)

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

plt.style.use(scripts / "lamalab.mplstyle")

human_dir = output / "human_scores"
model_subset_dir = output / "human_subset_model_scores"
model_dir = output / "overall_model_scores"

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

def make_table(results, output_tex_file) -> None:
    models = []
    overall_performance = []
    calculation = []
    knowledge = []
    reasoning = []
    intuition = []
    easy = []
    medium = []
    advanced = []

    for model, values in results.items():
        if model not in model_file_name_to_label:
            continue
        models.append(value_) if (value_ := model_file_name_to_label.get(model)) is not None else print(f"Model {model} not in the dictionary")
        overall_performance.append(values["fraction_correct"])
        total_questions = len(values["model_scores"])
        
        total_calculation = 0
        total_knowledge = 0
        total_reasoning = 0
        total_intuition = 0
        total_easy = 0
        total_medium = 0
        total_advanced = 0

        count_calculation = 0
        count_knowledge = 0
        count_reasoning = 0 
        count_intuition = 0
        count_easy = 0
        count_medium = 0
        count_advanced = 0
        for value in values["model_scores"]:
            print(value)
            keywords = value["keywords"]
            correct = value ["all_correct"]
            if "requires-calculation" in keywords:
                total_calculation += 1
                if correct:
                    count_calculation +=1
            if "requires-knowledge" in keywords:
                total_knowledge += 1
                if correct:
                    count_knowledge += 1
            if "requires-reasoning" in keywords:
                total_reasoning += 1
                if correct:
                    count_reasoning += 1
            if "requires-intuition" in keywords:
                total_intuition += 1
                if correct:
                    count_intuition += 1
            if "difficulty-basic" in keywords:
                total_easy += 1
                if correct:
                    count_easy += 1
            elif "difficulty-advanced" in keywords:
                total_advanced += 1
                if correct:
                    count_advanced += 1
            else:
                total_medium += 1
                if correct:
                    count_medium += 1
        
        calculation.append(count_calculation/total_calculation)
        knowledge.append(count_knowledge/total_knowledge)
        reasoning.append(count_reasoning/total_reasoning)
        intuition.append(count_intuition/total_intuition)
        easy.append(count_easy/total_easy)
        medium.append(count_medium/total_medium)
        advanced.append(count_advanced/total_advanced)


    df = pd.DataFrame({
        "Model": models,
        "Calculation": calculation,
        "Knowledge": knowledge,
        "Reasoning": reasoning,
        "Intuition": intuition,
        "Basic": easy,
        "Intermediate": medium,
        "Advanced": advanced,
        "Overall Accuracy": overall_performance,
    })

    df.columns = pd.MultiIndex.from_tuples([
        ("Model", ""),
        ("Requires", "Calculation"),
        ("Requires", "Knowledge"),
        ("Requires", "Reasoning"),
        ("Requires", "Intuition"),
        ("Difficulty", "Basic"),
        ("Difficulty", "Intermediate"),
        ("Difficulty", "Advanced"),
        ("", "Overall Accuracy"),
    ])
    
    latex_table = df.to_latex(index=False, multirow=True)
    
    with open(output_tex_file, 'w') as f:
        f.write(latex_table)

    return


def combine_scores_for_model(folder, datafolder, human_baseline_folder=None, min_human_responses: int = 4):
    try:
        df = load_all_reports(folder, datafolder)
        if df.empty:
            raise ValueError("No data loaded from reports")

        if human_baseline_folder is not None:
            relevant_questions = get_human_scored_questions_with_at_least_n_scores(human_baseline_folder, min_human_responses)
            df = df[df[("name", 0)].isin(relevant_questions)]

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
        return {"error": str(e), "fraction_correct": 0, "model_scores": []}

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

        df = pd.concat([tools_df, notools_df], ignore_index=True)
        if len(df) < min_reports:
            continue

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

        human_scores.append({
            "fraction_correct": fraction_correct,
            "model_scores": model_scores,
        })

    return human_scores

def plot_performance(
    model_scores,
    outname,
    human_scores=None,
):
    model_scores = sorted(model_scores, key=lambda x: x[1])
    print(model_scores)
    fig, ax = plt.subplots()
    ax.hlines(
        np.arange(len(model_scores)),
        0,
        [s[1] for s in model_scores],
        label="Model scores",
        color="#007acc",
        alpha=0.2,
        linewidth=5,
    )
    ax.plot(
        [s[1] for s in model_scores],
        np.arange(len(model_scores)),
        "o",
        markersize=5,
        color="#007acc",
        alpha=0.6,
    )

    scores = [s[1] for s in model_scores]

    range_frame(ax, np.array([0, max(scores)]), np.arange(len(model_scores)), pad=0.1)
    ax.set_yticks(np.arange(len(model_scores)))
    ax.set_yticklabels([s[0] for s in model_scores])

    if human_scores is not None:
        ax.vlines(
            np.mean(human_scores),
            -1,
            len(model_scores),
            label="Human mean",
            color="#00B945",
            alpha=0.6,
        )
        ax.vlines(
            np.max(human_scores),
            -1,
            len(model_scores),
            label="Human max",
            color="#FF9500",
            alpha=0.6,
        )
        ax.vlines(
            np.min(human_scores),
            -1,
            len(model_scores),
            label="Human min",
            color="#845B97",
            alpha=0.6,
        )

        ax.text(
            np.mean(human_scores),
            len(model_scores) - 0.5,
            "average human score",
            rotation=45,
            c="#00B945",
        )

        ax.text(
            np.max(human_scores),
            len(model_scores) - 0.5,
            "highest human score",
            rotation=45,
            c="#FF9500",
        )

        ax.text(
            np.min(human_scores),
            len(model_scores) - 0.5,
            "lowest human score",
            rotation=45,
            c="#845B97",
        )

    ax.set_xlabel("Fraction of completely correct answers")
    # fig.tight_layout()
    fig.savefig(outname, bbox_inches="tight")


if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    reports = glob(
        os.path.join(chembench_repo, "reports",
                     "**", "reports", "**", "*.json")
    )
    overall_files = glob(
        os.path.join(chembench_repo, "reports", "**", "*.json")
    )

    # model_overall_scores = {}
    # overall_scores = {}
    # for file in overall_files:
    #     p = Path(file).parts[-2]
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #     model_overall_scores[p] = data["fraction_correct"]
    #     overall_scores[p] = data

    # model_overall_scores = [(model_file_name_to_label[k], v) for k, v in model_overall_scores.items() if k in model_file_name_to_label]


    models = list(set([Path(p).parent for p in reports]))
    logger.info(f"Found {len(models)} reports")
    logger.debug(f"First 5 reports: {models[:5]}")
    outpath = os.path.join(output, "overall_model_scores")
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    datafolder = os.path.join(chembench_repo, "data")
    human_baseline_folder = os.path.join(chembench_repo, "reports", "humans")

    model_overall_scores = {}
    overall_scores = {}
    model_scores_human_subset = {}
    for i, file in enumerate(models):
        try:
            p = Path(file).parts[-3]
            human_results = combine_scores_for_model(file, datafolder, human_baseline_folder)
            model_scores_human_subset[p] = human_results["fraction_correct"]
            model_results = combine_scores_for_model(file, datafolder)
            model_overall_scores[p] = model_results["fraction_correct"]
            overall_scores[p] = model_results
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            pass
    model_overall_scores = [(model_file_name_to_label[k], v) for k, v in model_overall_scores.items() if k in model_file_name_to_label]

    model_scores_human_subset = [
        (model_file_name_to_label[k], v) for k, v in model_scores_human_subset.items() if k in model_file_name_to_label
    ]
    human_path = os.path.join(chembench_repo, "reports", "humans", "reports")

    human_subset = collect_human_results(human_path, datafolder)
    
    human_scores = [i["fraction_correct"] for i in human_subset]

    plot_performance(
        model_overall_scores,
        figures / "overall_performance.pdf",
        human_scores
    )
    plot_performance(
        model_scores_human_subset,
        figures / "human_subset_performance.pdf",
        human_scores
    )
    
    overall_scores["humans"] = human_subset
    human_results["humans"] = human_subset

    make_table(overall_scores, output / "performance_table.tex")
    make_table(human_results, output / "performance_table_human_subset.tex")
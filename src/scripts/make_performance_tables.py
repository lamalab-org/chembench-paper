
import os
import pickle

import json
from loguru import logger

from chembench.analysis import (
    all_correct,
    load_all_reports,
)
from chembench.types import PathType
from collections import defaultdict

from glob import glob
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
    "claude2": "Claude 2",
    "claude3": "Claude 3",
    "claude3.5": "Claude 3.5 Sonnet",
    "command-r+": "Command R+",
    "galatica_120b": "Galactica 120B",
    "gemini-pro": "Gemini Pro",
    "gemma-1-1-7b-it": "Gemma 1.1 7B",
    "gemma-2-9b-it": "Gemma 2 9B",
    "gpt-3.5-turbo": "GPT-3.5-Turbo",
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "Humans": "Humans",
    "o1": "OpenAI o1",
    "llama2-70b-chat": "Llama 2 70B Chat",
    "llama3-70b-instruct": "Llama 3 70B",
    "llama3-8b-instruct": "Llama 3 8B",
    "llama3.1-405b-instruct": "Llama 3.1 405B",
    "llama3.1-70b-instruct": "Llama 3.1 70B",
    "llama3.1-8b-instruct": "Llama 3.1 8B",
    "mixtral-8x7b-instruct": "Mixtral 8x7B",
    "mistral-large-2-123b": "Mistral Large 2 123B",
    "paper-qa": "Paper QA",
    "phi-3-medium-4k-instruct": "Phi 3 Medium 4K",
    "random_baseline": "Random Baseline"
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

def make_table(results, output_tex_file, human_subset = False) -> None:
    models = []
    overall_performance = []
    calculation = []
    knowledge = []
    reasoning = []
    intuition = []
    easy = []
    medium = []
    advanced = []

    for p in results:
        model = p["model"]
        values = p["scores"]
        logger.info(f"Working in model {model}")

        if model not in model_file_name_to_label:
            logger.error(f"Not found the name matching for model {model}")
            continue

        if model == "Humans":
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
            if not human_subset:
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
        if not human_subset:
            intuition.append(count_intuition/total_intuition)
        easy.append(count_easy/total_easy)
        medium.append(count_medium/total_medium)
        advanced.append(count_advanced/total_advanced)

    if not human_subset:
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
            ("Overall Accuracy", ""),
        ])
    else:
        df = pd.DataFrame({
            "Model": models,
            "Calculation": calculation,
            "Knowledge": knowledge,
            "Reasoning": reasoning,
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
            ("Difficulty", "Basic"),
            ("Difficulty", "Intermediate"),
            ("Difficulty", "Advanced"),
            ("Overall Accuracy", ""),
        ])
    
    # Round all numeric columns to 2 decimal places (excluding non-numeric columns like "Model")
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].applymap(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))


   # Function to bold the best value (assuming higher is better)
    def bold_best(series):
        best = series.astype(float).max()  # Modify as per your criteria (e.g., use min() for smaller is better)
        return series.apply(lambda x: f"\\textbf{{{x}}}" if float(x) == best else x)
    
    # Apply the bolding to relevant numeric columns (excluding 'Model')
    if not human_subset:
        columns_to_bold = [
            ('Requires', 'Calculation'),
            ('Requires', 'Knowledge'),
            ('Requires', 'Reasoning'),
            ('Requires', 'Intuition'),
            ('Difficulty', 'Basic'),
            ('Difficulty', 'Intermediate'),
            ('Difficulty', 'Advanced'),
            ('Overall Accuracy', '')
        ]
    else:
        columns_to_bold = [
            ('Requires', 'Calculation'),
            ('Requires', 'Knowledge'),
            ('Requires', 'Reasoning'),
            ('Difficulty', 'Basic'),
            ('Difficulty', 'Intermediate'),
            ('Difficulty', 'Advanced'),
            ('Overall Accuracy', '')
        ]
    
    df[columns_to_bold] = df[columns_to_bold].apply(bold_best)

    # Generate LaTeX table with bolded best values
    latex_table = df.to_latex(index=False, multirow=True, escape=False)
    
    if not human_subset:
        # Manually modify the LaTeX code for centering multi-level headers
        latex_table = latex_table.replace(
            r"Model & \multicolumn{4}{r}{Requires} & \multicolumn{3}{r}{Difficulty} & Overall Accuracy \\",
            r"\multirow{3}{*}{Model} & \multicolumn{4}{c}{\textbf{Requires}} & \multicolumn{3}{c}{\textbf{Difficulty}} & \multirow{3}{*}{\textbf{Overall Accuracy}}\\"
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-8}\\"
        )
        latex_table = latex_table.replace(
            r"\begin{tabular}{lllllllll}",
            r"\begin{tabular}{ccccccccc}"
        )
    else:
        # Manually modify the LaTeX code for centering multi-level headers
        latex_table = latex_table.replace(
            r"Model & \multicolumn{3}{r}{Requires} & \multicolumn{3}{r}{Difficulty} & Overall Accuracy \\",
            r"\multirow{3}{*}{Model} & \multicolumn{3}{c}{\textbf{Requires}} & \multicolumn{3}{c}{\textbf{Difficulty}} & \multirow{3}{*}{\textbf{Overall Accuracy}}\\"
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}\\"
        )
        latex_table = latex_table.replace(
            r"\begin{tabular}{llllllll}",
            r"\begin{tabular}{cccccccc}"
        )

    # Wrap the table with \resizebox
    latex_table = (
        r"\begin{table}[htbp]\centering" + "\n" +
        r"\resizebox{\textwidth}{!}{" + "\n" +
        latex_table + "\n" +
        r"}" + "\n" +
        r"\end{table}"
    )
    
    with open(output_tex_file, 'w') as f:
        f.write(latex_table)

    return


if __name__ == "__main__": 

    # Load t_overall_scores.pkl
    with open(os.path.join(output, "t_overall_scores.pkl"), 'rb') as f:
        overall_scores = pickle.load(f)

    # Load t_overall_human_scores.pkl
    with open(os.path.join(output, "t_overall_human_scores.pkl"), 'rb') as f:
        overall_human_scores = pickle.load(f)

    make_table(overall_scores, output / "performance_table.tex")
    make_table(overall_human_scores, output / "performance_table_human_subset.tex", human_subset = True)
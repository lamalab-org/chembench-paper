import pickle
from typing import Optional
from loguru import logger

import natsort
from natsort import (
    natsort_key,
    natsorted
)

from paths import output

from paths import data
import pandas as pd


def compile_df(results: dict, human_subset=False) -> pd.DataFrame:
    models = []
    overall_performance = []
    calculation = []
    knowledge = []
    reasoning = []
    intuition = []
    easy = []
    medium = []
    advanced = []

    for model, df in results.items():
        logger.info(f"Working on model {model}")

        models.append(model)
        overall_performance.append(df["all_correct_"].mean())
        total_questions = len(df)

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
        for i, row in df.iterrows():
            keywords = row["keywords_0"]
            correct = row["all_correct_"]
            if "requires-calculation" in keywords:
                total_calculation += 1
                if correct:
                    count_calculation += 1
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

        calculation.append(count_calculation / total_calculation)
        knowledge.append(count_knowledge / total_knowledge)
        reasoning.append(count_reasoning / total_reasoning)
        if not human_subset:
            intuition.append(count_intuition / total_intuition)
        easy.append(count_easy / total_easy)
        medium.append(count_medium / total_medium)
        advanced.append(count_advanced / total_advanced)

    if not human_subset:
        df = pd.DataFrame(
            {
                "Model": models,
                "Calculation": calculation,
                "Knowledge": knowledge,
                "Reasoning": reasoning,
                "Intuition": intuition,
                "Basic": easy,
                "Intermediate": medium,
                "Advanced": advanced,
                "Overall Accuracy": overall_performance,
            }
        )

        df.columns = pd.MultiIndex.from_tuples(
            [
                ("Model", ""),
                ("Requires", "Calculation"),
                ("Requires", "Knowledge"),
                ("Requires", "Reasoning"),
                ("Requires", "Intuition"),
                ("Difficulty", "Basic"),
                ("Difficulty", "Intermediate"),
                ("Difficulty", "Advanced"),
                ("Overall Accuracy", ""),
            ]
        )
    else:
        df = pd.DataFrame(
            {
                "Model": models,
                "Calculation": calculation,
                "Knowledge": knowledge,
                "Reasoning": reasoning,
                "Basic": easy,
                "Intermediate": medium,
                "Advanced": advanced,
                "Overall Accuracy": overall_performance,
            }
        )

        df.columns = pd.MultiIndex.from_tuples(
            [
                ("Model", ""),
                ("Requires", "Calculation"),
                ("Requires", "Knowledge"),
                ("Requires", "Reasoning"),
                ("Difficulty", "Basic"),
                ("Difficulty", "Intermediate"),
                ("Difficulty", "Advanced"),
                ("Overall Accuracy", ""),
            ]
        )
    return df


def make_table(
    results,
    output_tex_file,
    human_subset=False,
    additional_df: Optional[pd.DataFrame] = None,
) -> None:
    df = compile_df(results, human_subset=human_subset)
    if additional_df is not None:
        df = pd.concat([df, additional_df], ignore_index=True)
    # Round all numeric columns to 2 decimal places (excluding non-numeric columns like "Model")
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].applymap(
        lambda x: f"{x:.2f}"
    )
    # filter out rows where ("Model", "") starts with a number
    df = df[~df[("Model", "")].str.match(r"^\d")]
    # Sort the DataFrame alphabetically by the column "Models"
    df = df.reindex(natsort.natsorted(df.index, key=lambda x: df.loc[x, 'Model'], alg=natsort.IC))

    # Function to bold the best value (assuming higher is better)
    def bold_best(series):
        best = series.astype(
            float
        ).max()  # Modify as per your criteria (e.g., use min() for smaller is better)
        return series.apply(lambda x: f"\\textbf{{{x}}}" if float(x) == best else x)

    # Apply the bolding to relevant numeric columns (excluding 'Model')
    if not human_subset:
        columns_to_bold = [
            ("Requires", "Calculation"),
            ("Requires", "Knowledge"),
            ("Requires", "Reasoning"),
            ("Requires", "Intuition"),
            ("Difficulty", "Basic"),
            ("Difficulty", "Intermediate"),
            ("Difficulty", "Advanced"),
            ("Overall Accuracy", ""),
        ]
    else:
        columns_to_bold = [
            ("Requires", "Calculation"),
            ("Requires", "Knowledge"),
            ("Requires", "Reasoning"),
            ("Difficulty", "Basic"),
            ("Difficulty", "Intermediate"),
            ("Difficulty", "Advanced"),
            ("Overall Accuracy", ""),
        ]

    df[columns_to_bold] = df[columns_to_bold].apply(bold_best)

    # Generate LaTeX table with bolded best values
    latex_table = df.to_latex(index=False, multirow=True, escape=False)

    if not human_subset:
        # Manually modify the LaTeX code for centering multi-level headers
        latex_table = latex_table.replace(
            r"Model & \multicolumn{4}{r}{Requires} & \multicolumn{3}{r}{Difficulty} & Overall Accuracy \\",
            r"\multirow{3}{*}{Model} & \multicolumn{4}{c}{\textbf{Requires}} & \multicolumn{3}{c}{\textbf{Difficulty}} & \multirow{3}{*}{\textbf{Overall Accuracy}}\\"
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-8}\\",
        )
        latex_table = latex_table.replace(
            r"\begin{tabular}{lllllllll}", r"\begin{tabular}{ccccccccc}"
        )
    else:
        # Manually modify the LaTeX code for centering multi-level headers
        latex_table = latex_table.replace(
            r"Model & \multicolumn{3}{r}{Requires} & \multicolumn{3}{r}{Difficulty} & Overall Accuracy \\",
            r"\multirow{3}{*}{Model} & \multicolumn{3}{c}{\textbf{Requires}} & \multicolumn{3}{c}{\textbf{Difficulty}} & \multirow{3}{*}{\textbf{Overall Accuracy}}\\"
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}\\",
        )
        latex_table = latex_table.replace(
            r"\begin{tabular}{llllllll}", r"\begin{tabular}{cccccccc}"
        )

    with open(output_tex_file, "w") as f:
        f.write(latex_table)

    return


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as f:
        overall_scores = pickle.load(f)

    with open(data / "humans_as_models_scores_combined.pkl", "rb") as f:
        human_scores = pickle.load(f)["raw_scores"]

    human_results = compile_df(human_scores, human_subset=True)
    # now, average over all rows to get one row. The "model" column is string, so it will be ignored
    # drop the "model" column
    human_results = human_results.drop(columns=("Model", ""))
    human_results = human_results.mean()
    # now add the model column back and call it "Human"
    human_results = human_results.to_frame().T
    human_results.insert(0, "Model", "Human")

    make_table(overall_scores["overall"], output / "performance_table.tex")
    make_table(
        overall_scores["human_aligned_combined"],
        output / "performance_table_human_subset.tex",
        human_subset=True,
        additional_df=human_results,
    )

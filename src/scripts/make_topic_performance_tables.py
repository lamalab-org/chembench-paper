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

SHORT_KEYWORDS = {
    "analytical-chemistry": "AC",
    "biochemistry": "BC",
    "inorganic-chemistry": "IC",
    "toxicity/safety": "T/S",
    "organic-chemistry": "OC",
    "physical-chemistry": "PC",
    "photochemistry": "PhC",
    "solid-state-chemistry": "SSC",
    "technical-chemistry": "TC",
    "polymer-chemistry": "PolyC",
    "computational-chemistry": "CC",
    "quantum-chemistry": "QC",
    "electrochemistry": "EC",
    "chemical-preference": "ChemPref",
    "general-chemistry": "GC",
    "lab-observations": "LabObs",
    "materials-science": "MS",
    "pharmacology": "Pharm",
}

FULL_NAMES_LEGEND = {
    "AC": "Analytical Chemistry",
    "BC": "Biochemistry",
    "IC": "Inorganic Chemistry",
    "T/S": "Toxicity/Safety",
    "OC": "Organic Chemistry",
    "PC": "Physical Chemistry",
    "PhC": "Photochemistry",
    "SSC": "Solid State Chemistry",
    "TC": "Technical Chemistry",
    "PolyC": "Polymer Chemistry",
    "CC": "Computational Chemistry",
    "QC": "Quantum Chemistry",
    "EC": "Electrochemistry",
    "ChemPref": "Chemical Preference",
    "GC": "General Chemistry",
    "LabObs": "Lab Observations",
    "MS": "Materials Science",
    "Pharm": "Pharmacology",
    "RK": "Knowledge",
    "RR": "Reasoning",
    "RI": "Intuition",
    "RC": "Calculation",
}

def merge_categories(df: pd.DataFrame):
    category_mapping = {
        "GC": ["GC"],
        "TC": ["TC"],
        "ChemPref": ["ChemPref"],
        "OC": ["BC", "Pharm", "OC", "PhC"],
        "MS": ["SSC", "PolyC", "MS"],
        "PC": ["CC", "QC", "EC", "PC"],
        "AC": ["AC", "LabObs"],
        "IC": ["IC"],
    }
    for key, values in category_mapping.items():
        df.loc[df["topic"].isin(values), "topic"] = key

    return df

def replace_topic(x):
    for key, value in SHORT_KEYWORDS.items():
        x = x.replace(key, value)
    return x

def annotate_dataset(df):
    if "keywords" in df.columns:
        df["topic"] = df["keywords"].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    else:
        raise KeyError("Keywords column must be present in the dataframe")

    df["topic"] = df["topic"].apply(lambda x: "toxicity/safety" if "toxic" in x or "safety" in x else x)
    df["topic"] = [SHORT_KEYWORDS[x] if x in SHORT_KEYWORDS else x for x in df["topic"]]
    df["topic"] = df["topic"].apply(replace_topic)
    df = merge_categories(df)
    df["topic"] = df["topic"].apply(lambda x: FULL_NAMES_LEGEND[x] if x in FULL_NAMES_LEGEND else x)

    return df

def compile_df(results: dict, human_subset=False) -> pd.DataFrame:
    models = []
    overall_performance = []
    analytical = []
    chemical_preference = []
    general = []
    inorganic = []
    materials_science = []
    organic = []
    physical = []
    technical = []
    toxicity_safety = []

    for model, df in results.items():
        logger.info(f"Working on model {model}")

        models.append(model)
        df = annotate_dataset(df)
        overall_performance.append(df["all_correct_"].mean())
        total_questions = len(df)

        total_analytical = 0
        total_chemical_preference = 0
        total_general = 0
        total_inorganic = 0
        total_materials_science = 0
        total_organic = 0
        total_physical = 0
        total_technical = 0
        total_toxicity_safety = 0

        count_analytical = 0
        count_chemical_preference = 0
        count_general = 0
        count_inorganic = 0
        count_materials_science = 0
        count_organic = 0
        count_physical = 0
        count_technical = 0
        count_toxicity_safety = 0

        for i, row in df.iterrows():
            topic = row["topic"]
            correct = row["all_correct_"]
            if "GC" in topic:
                total_general += 1
                if correct:
                    count_general += 1
            if "TC" in topic:
                total_technical += 1
                if correct:
                    count_technical += 1
            if not human_subset:
                if "ChemPref" in topic:
                    total_chemical_preference += 1
                    if correct:
                        count_chemical_preference += 1
            if "OC" in topic:
                total_organic += 1
                if correct:
                    count_organic += 1
            if "MS" in topic:
                total_materials_science += 1
                if correct:
                    count_materials_science += 1
            if "PC" in topic:
                total_physical += 1
                if correct:
                    count_physical += 1
            if "AC" in topic:
                total_analytical += 1
                if correct:
                    count_analytical += 1
            if "Toxicity/Safety" in topic:
                total_toxicity_safety += 1
                if correct:
                    count_toxicity_safety += 1
            if "IC" in topic:
                total_inorganic += 1
                if correct:
                    count_inorganic += 1

        analytical.append(count_analytical / total_analytical)
        if not human_subset:
            chemical_preference.append(count_chemical_preference / total_chemical_preference)
        general.append(count_general / total_general)
        inorganic.append(count_inorganic / total_inorganic)
        materials_science.append(count_materials_science / total_materials_science)
        organic.append(count_organic / total_organic)
        physical.append(count_physical / total_physical)
        technical.append(count_technical / total_technical)
        toxicity_safety.append(count_toxicity_safety / total_toxicity_safety)


    if not human_subset:
        with open(output / "total_analytical.txt", "w") as f:
            percentage = count_analytical / total_analytical
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_analytical} Questions {percentage_str}" + "\endinput")

        with open(output / "total_chemical_preference.txt", "w") as f:
            percentage = count_chemical_preference / total_chemical_preference
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_chemical_preference} Questions {percentage_str}" + "\endinput")

        with open(output / "total_general.txt", "w") as f:
            percentage = count_general / total_general
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_general} Questions {percentage_str}" + "\endinput")

        with open(output / "total_inorganic.txt", "w") as f:
            percentage = count_inorganic / total_inorganic
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_inorganic} Questions {percentage_str}" + "\endinput")

        with open(output / "total_materials_science.txt", "w") as f:
            percentage = count_materials_science / total_materials_science
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_materials_science} Questions {percentage_str}" + "\endinput")

        with open(output / "total_organic.txt", "w") as f:
            percentage = count_organic / total_organic
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_organic} Questions {percentage_str}" + "\endinput")

        with open(output / "total_physical.txt", "w") as f:
            percentage = count_physical / total_physical
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_physical} Questions {percentage_str}" + "\endinput")

        with open(output / "total_technical.txt", "w") as f:
            percentage = count_technical / total_technical
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_technical} Questions {percentage_str}" + "\endinput")

        with open(output / "total_toxicity_safety.txt", "w") as f:
            percentage = count_toxicity_safety / total_toxicity_safety
            percentage_str = f"({percentage:.2%})".replace("%", "\%")
            f.write(f"{total_toxicity_safety} Questions {percentage_str}" + "\endinput")

        df = pd.DataFrame(
            {
                "Model": models,
                "Analytical": analytical,
                "Chemical Preference": chemical_preference,
                "General": general,
                "Inorganic": inorganic,
                "Materials Science": materials_science,
                "Organic": organic,
                "Physical": physical,
                "Technical": technical,
                "Toxicity/Safety": toxicity_safety,
                "Overall Accuracy": overall_performance,
            }
        )

        df.columns = [
            "Model",
            "Analytical",
            "Chemical Preference",
            "General",
            "Inorganic",
            "Materials Science",
            "Organic",
            "Physical",
            "Technical",
            "Toxicity/Safety",
            "Overall Accuracy",
        ]
    else:
        df = pd.DataFrame(
            {
                "Model": models,
                "Analytical": analytical,
                "General": general,
                "Inorganic": inorganic,
                "Materials Science": materials_science,
                "Organic": organic,
                "Physical": physical,
                "Technical": technical,
                "Toxicity/Safety": toxicity_safety,
                "Overall Accuracy": overall_performance,
            }
        )

        df.columns = [
            "Model",
            "Analytical",
            "General",
            "Inorganic",
            "Materials Science",
            "Organic",
            "Physical",
            "Technical",
            "Toxicity/Safety",
            "Overall Accuracy",
        ]

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
    # # filter out rows where ("Model", "") starts with a number
    # df = df[~df["Model"].str.match(r"^\d")]
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
            "Analytical",
            "Chemical Preference",
            "General",
            "Inorganic",
            "Materials Science",
            "Organic",
            "Physical",
            "Technical",
            "Toxicity/Safety",
            "Overall Accuracy"
        ]
    else:
        columns_to_bold = [
            "Analytical",
            "General",
            "Inorganic",
            "Materials Science",
            "Organic",
            "Physical",
            "Technical",
            "Toxicity/Safety",
            "Overall Accuracy"
        ]

    df[columns_to_bold] = df[columns_to_bold].apply(bold_best)

    # Generate LaTeX table with bolded best values
    latex_table = df.to_latex(index=False, multirow=True, escape=False)

    if not human_subset:
        latex_table = latex_table.replace(
            r"\begin{tabular}{lllllllll}", r"\begin{tabular}{ccccccccc}"
        )
    else:
        latex_table = latex_table.replace(
            r"\begin{tabular}{llllllllll}", r"\begin{tabular}{cccccccccc}"
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
    human_results = human_results.drop(columns="Model")
    human_results = human_results.mean()
    # now add the model column back and call it "Human"
    human_results = human_results.to_frame().T
    human_results.insert(0, "Model", "Human")

    make_table(overall_scores["overall"], output / "performance_topic_table.tex")
    make_table(
        overall_scores["human_aligned_combined"],
        output / "performance_topic_table_human_subset.tex",
        human_subset=True,
        additional_df=human_results,
    )

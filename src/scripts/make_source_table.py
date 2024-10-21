
import os
import json
import pickle
import pandas as pd
import numpy as np
from loguru import logger
import concurrent.futures
from pandas import json_normalize

from chembench.analysis import (
    construct_name_to_path_dict,
    is_semiautomatically_generated,
)
from utils import obtain_chembench_repo

from paths import tex, data

def craft_table(df):
    latex_table = df.to_latex(index=False)

    latex_table = latex_table.replace(
        r"\begin{tabular}{lr}",
        r"\begin{tabular}{cc}"
    )

    with open(output_file, 'w') as f:
        f.write(latex_table)

    return

def load_single_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
        return json_normalize(data)

def load_data(data_paths):
    results = []
    for path in data_paths:
        data = load_single_json(path)
        results.append(load_single_json(path))

    return pd.concat(results, ignore_index=True, sort=False)

def collect_data(datafolder):
    names = construct_name_to_path_dict(datafolder)
    data_paths = list(names.values())

    df = load_data(data_paths)

    if df.empty:
        raise ValueError("No data loaded from reports")
    logger.info(f"Loaded {len(df)} rows of data")
    
    df["semiautomatically"] = df.apply(is_semiautomatically_generated, axis=1)

    sources = {}
    for i, row in df.iterrows():
        try:
            if row["semiautomatically"]:
                if "Semiautomatically Generated" in sources:
                    sources["Semiautomatically Generated"] += 1
                else:
                    sources["Semiautomatically Generated"] = 1
            elif not pd.isna(row["meta.exam.country"]):
                if "Exam" in sources:
                    sources["Exam"] += 1
                else:
                    sources["Exam"] = 1
            elif not pd.isna(row["meta.urls"]):
                if "URL" in sources:
                    sources["URL"] += 1
                else:
                    sources["URL"] = 1
            elif not pd.isna(row["meta.textbook.title"]):
                if "Textbook" in sources:
                    sources["Textbook"] += 1
                else:
                    sources["Textbook"] = 1
            elif not pd.isna(row["meta.lecture.country"]):
                if "Lectures" in sources:
                    sources["Lectures"] += 1
                else:
                    sources["Lectures"] = 1
            elif not pd.isna(row["icho.country"]):
                if "IChO" in sources:
                    sources["IChO"] += 1
                else:
                    sources["IChO"] = 1
            else:
                if "No source" in sources:
                    sources["No source"] += 1
                else:
                    sources["No source"] = 1
        except Exception as e:
            if "URL" in sources:
                sources["URL"] += 1
            else:
                sources["URL"] = 1

    sources_df = pd.DataFrame(list(sources.items()), columns=['Source', 'Count'])
    sources_df = sources_df.sort_values(by='Count', ascending=False)

    return sources_df

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    datafolder = os.path.join(
        chembench_repo, "data"
    )
    output_file = (
        os.path.join(
            tex,
            "sources_table.tex"
        )
    )

    data = collect_data(datafolder)

    craft_table(data)
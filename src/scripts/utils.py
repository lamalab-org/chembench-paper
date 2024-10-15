import os
import pandas as pd
from paths import data, scripts
from dotenv import load_dotenv
import json
from pathlib import Path
from chembench.types import PathType
from scipy.constants import golden

ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2

ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

load_dotenv()
GH_PAT = os.getenv("GH_PAT")
print(GH_PAT)


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


def obtain_chembench_repo():
    # if in ../data the repository chem-bench is not found, clone it
    # if it is found, pull the latest version
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(data, "chem-bench-main")):
        os.chdir(data)
        zip_path = os.path.join(data, "main.zip")
        # Check if main.zip exists and remove it if it does
        if os.path.exists(zip_path):
            os.remove(zip_path)
        # download zip from https://github.com/lamalab-org/chem-bench/archive/refs/heads/humanbench.zip
        # using wget and then unzip it
        os.system(
            f"wget --header 'Authorization: token {GH_PAT}' https://github.com/lamalab-org/chem-bench/archive/refs/heads/main.zip"
        )
        os.system("unzip main.zip")

    else:
        os.chdir(os.path.join(data, "chem-bench-main"))
    os.chdir(cwd)
    return os.path.join(data, "chem-bench-main")

def replace_topic(x):
    for key, value in SHORT_KEYWORDS.items():
        x = x.replace(key, value)
    return x

def merge_categories(keywords: str):
    keywords = replace_topic(keywords)
    category_mapping = {
        "GC": ["GC"],
        "TC": ["TC"],
        "ChemPref": ["ChemPref"],
        "OC": ["BC", "Pharm", "OC", "PhC"],
        "MS": ["SSC", "PolyC", "MS"],
        "PC": ["CC", "QC", "EC", "PC"],
        "AC": ["AC", "LabObs"],
        "IC": ["IC"],
        "T/S": ["T/S"]
    }
    
    for keyword in keywords.split():
        for category, values in category_mapping.items():
            if keyword in values:
                return category
    
    return None


def clean_up(keywords):
    keywords = replace_topic(keywords)
    return merge_categories(keywords)


def keyword_filter(keyword_list: list[str]):
    """
    `exclude` defines words that are not main categories or can
    interfere with the classification

    `include` defines the words that are main categories
    across the questions topics and requires-<category>
    """
    exclude = [
        "coordination chemistry",
        "toxic gases",
        "assymetric chemistry",
        "thermodynamics",
        "nuclear-chemistry",
        "thermochemistry",
        "stereochemistry",
        "green chemistry",
        "mercury",
    ]
    include = [
        "chemistry",
        "safety",
        "toxic",
        "chemical-preference",
        "lab-observations",
        "materials-science",
        "pharmacology",
    ]
    return [keyword for keyword in keyword_list if keyword not in exclude and any(inc in keyword for inc in include)]

def filter_keywords(keywords):
    filtered_keywords = keyword_filter(keywords)
    q_keywords = "+".join(sorted(filtered_keywords)).split()[0]

    return q_keywords
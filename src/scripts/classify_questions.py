
from utils import (
    obtain_chembench_repo,
    filter_keywords,
    SHORT_KEYWORDS,
    clean_up,
    FULL_NAMES_LEGEND,
)

from chembench.types import PathType
from chembench.analysis import is_semiautomatically_generated

from loguru import logger
from paths import output, data
from pathlib import Path
from typing import List, Literal, Union
import pandas as pd
import os
import json


TOPICS = [
    "analytical-chemistry",
    "inorganic-chemistry",
    "toxicity/safety",
    "organic-chemistry",
    "physical-chemistry",
    "technical-chemistry",
    "chemical-preference",
    "general-chemistry",
    "materials-science",
]


def format_qa(row: pd.Series) -> str:
    """Format a question and answer pair."""
    template = f"""Question: {row['question']}
Answer: {row['target']}"""
    return template


def build_question_df(json_paths: List[Union[str, Path]]) -> pd.DataFrame:
    """Build a DataFrame of questions from a list of JSON files."""
    question_df = []
    for file in json_paths:
        folder = Path(file).parts[-2]
        stem = Path(file).stem
        with open(file, "r") as f:
            data = json.load(f)
        description = data["description"]
        keywords = filter_keywords(data["keywords"])
        q_topic = keywords
        keywords = " ,".join(data["keywords"])

        q_topic = "toxicity/safety" if "toxic" in q_topic or "safety" in q_topic else q_topic
        q_topic = SHORT_KEYWORDS[q_topic] if q_topic in SHORT_KEYWORDS else q_topic
        q_topic = clean_up(q_topic)
        q_topic = FULL_NAMES_LEGEND[q_topic] if q_topic in FULL_NAMES_LEGEND else q_topic

        for example in data["examples"]:
            question_df.append(
                {
                    "question": example["input"],
                    "description": description,
                    "keywords": keywords,
                    "source": file,
                    "is_classification": "target_scores" in example,
                    "target": (" ,".join([k for k, v in example["target_scores"].items() if int(v) > 0]) if "target_scores" in example else example["target"]),
                    "name": f'{folder}-{stem}-{data["name"]}',
                    "topic": q_topic,
                }
            )
    df = pd.DataFrame(question_df)
    df["is_semiautomatically_generated"] = df.apply(is_semiautomatically_generated, axis=1)
    df["formatted"] = df.apply(format_qa, axis=1)
    return df

def classify_inputs(df: pd.DataFrame) -> pd.DataFrame:
    output = []
    for q_index in range(len(df)):
        res = {
            "id": q_index,
            "question": df["formatted"].values[q_index],
            "topic": df["topic"].values[q_index],
            "is_classification": df["is_classification"].values[q_index],
            "is_semiautomatically_generated": df["is_semiautomatically_generated"].values[q_index],
            "description": df["description"].values[q_index],
            "keywords": df["keywords"].values[q_index],
            "source": df["source"].values[q_index],
            "target": df["target"].values[q_index],
            "name": df["name"].values[q_index],
            "formatted": df["formatted"].values[q_index]
            
        }
        output.append(res)

    return pd.DataFrame(output)



def classify_questions(question_dir: PathType, debug: bool = False) -> pd.DataFrame:
    df = build_question_df(list(Path(question_dir).rglob("*.json")))
    logger.info(f"Classifying {len(df)} questions")
    if debug:
        df = df.head(200)

    classified = classify_inputs(df)

    return classified

def classify():
    # Load the data
    chembench_repo = obtain_chembench_repo()

    # Load the data
    data_ = os.path.join(chembench_repo, "data")

    df = classify_questions(data_)  # set debug = True to speed up

    df.to_csv(data / "questions.csv")
    df.to_pickle(data / "questions.pkl")


if __name__ == "__main__":
    classify()


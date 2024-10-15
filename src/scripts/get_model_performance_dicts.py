import os
import pickle
from typing import Dict, Literal

import pandas as pd

from chembench.analysis import (
    load_all_reports,
    all_correct,
    merge_with_topic_info,
)
from utils import obtain_chembench_repo, obtain_human_relevant_questions
from paths import data


def combine_scores_for_model(
    folder: str,
    datafolder: str,
    human_subset: Literal[
        "tool-allowed", "tool-disallowed", "none", "combined"
    ] = "none",
) -> pd.DataFrame:
    df = load_all_reports(folder, datafolder)

    relevant_questions = obtain_human_relevant_questions(human_subset)
    df = df[df[("name", 0)].isin(relevant_questions)]

    df["all_correct"] = df.apply(all_correct, axis=1)
    return df


def combine_scores_for_all_models(
    models: Dict[str, str],
    datafolder: str,
    topic_frame: pd.DataFrame,
    human_subset: Literal[
        "tool-allowed", "tool-disallowed", "none", "combined"
    ] = "none",
) -> Dict[str, pd.DataFrame]:
    return {
        modelname: merge_with_topic_info(
            combine_scores_for_model(folder, datafolder, human_subset=human_subset),
            topic_frame,
        )
        for modelname, folder in models.items()
    }


def load_reports(
    topic_frame: pd.DataFrame,
    human_subset: Literal[
        "tool-allowed", "tool-disallowed", "none", "combined"
    ] = "none",
) -> Dict[str, pd.DataFrame]:
    chembench = obtain_chembench_repo()
    datafolder = os.path.join(chembench, "data")

    with open(data / "name_to_dir_map.pkl", "rb") as handle:
        name_dir_map = pickle.load(handle)

    return combine_scores_for_all_models(
        name_dir_map, datafolder, topic_frame, human_subset
    )


def main():
    topic_frame = pd.read_pickle(data / "questions.pkl")

    results = {
        "overall": load_reports(topic_frame),
        "human_aligned_no_tool": load_reports(
            topic_frame, human_subset="tool-disallowed"
        ),
        "human_aligned_tool": load_reports(topic_frame, human_subset="tool-allowed"),
        "human_aligned_combined": load_reports(topic_frame, human_subset="combined"),
    }

    with open(data / "model_score_dicts.pkl", "wb") as handle:
        pickle.dump(results, handle)


if __name__ == "__main__":
    main()

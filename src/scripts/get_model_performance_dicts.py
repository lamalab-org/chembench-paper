import os
import pickle
from typing import Dict, Optional

import pandas as pd

from chembench.analysis import (
    load_all_reports,
    all_correct,
    merge_with_topic_info,
)
from utils import obtain_chembench_repo
from paths import data

def combine_scores_for_model(
    folder: str,
    datafolder: str,
    human_baseline_folder: Optional[str] = None,
    min_human_responses: int = 17
) -> pd.DataFrame:
    df = load_all_reports(folder, datafolder)
    if human_baseline_folder:
        relevant_questions = get_human_scored_questions_with_at_least_n_scores(
            human_baseline_folder, min_human_responses
        )
        df = df[df[("name", 0)].isin(relevant_questions)]
    df["all_correct"] = df.apply(all_correct, axis=1)
    return df

def combine_scores_for_all_models(
    models: Dict[str, str],
    datafolder: str,
    topic_frame: pd.DataFrame,
    human_baseline_folder: Optional[str] = None,
    min_human_responses: int = 17
) -> Dict[str, pd.DataFrame]:
    return {
        modelname: merge_with_topic_info(
            combine_scores_for_model(
                folder, datafolder, human_baseline_folder, min_human_responses
            ),
            topic_frame
        )
        for modelname, folder in models.items()
    }

def load_reports(topic_frame: pd.DataFrame, human_aligned: bool = False) -> Dict[str, pd.DataFrame]:
    chembench = obtain_chembench_repo()
    datafolder = os.path.join(chembench, "data")
    
    with open(data / "name_to_dir_map.pkl", "rb") as handle:
        name_dir_map = pickle.load(handle)
    
    human_baseline_folder = os.path.join(chembench, "reports/humans") if human_aligned else None
    
    return combine_scores_for_all_models(
        name_dir_map,
        datafolder,
        topic_frame,
        human_baseline_folder
    )

def main():
    topic_frame = pd.read_pickle(data / "questions.pkl")
    
    results = {
        "overall": load_reports(topic_frame),
        "human_aligned": load_reports(topic_frame, human_aligned=True)
    }
    
    with open(data / "model_score_dicts.pkl", "wb") as handle:
        pickle.dump(results, handle)

if __name__ == "__main__":
    main()
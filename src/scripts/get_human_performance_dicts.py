from chembench.analysis import (
    load_all_reports,
    get_human_scored_questions_with_at_least_n_scores,
    all_correct,
    merge_with_topic_info,
)
from utils import obtain_chembench_repo
import os
from paths import data
import pickle
import pandas as pd
from glob import glob
from pathlib import Path


def combine_scores_for_model(
    folder, datafolder, human_baseline_folder=None, min_human_responses: int = 4
):

    df = load_all_reports(folder, datafolder)
    if human_baseline_folder is not None:
        relevant_questions = get_human_scored_questions_with_at_least_n_scores(
            human_baseline_folder, min_human_responses
        )
        df = df[df[("name", 0)].isin(relevant_questions)]
    df["all_correct"] = df.apply(all_correct, axis=1)

    return df


def find_all_human_scores():

    chembench_repo = obtain_chembench_repo()

    human_files_tools = os.listdir(os.path.join(chembench_repo, "reports", "humans", "reports", "tool-allowed"))
    human_files_tools = [
        os.path.join(chembench_repo, "reports", "humans", "reports", "tool-allowed", p) for p in human_files_tools
    ]
    human_files_tools = [p for p in human_files_tools if len(glob(os.path.join(p, "*.json"))) > 100]
    print(f"Found {len(human_files_tools)} files from tools human scorers")

    human_files_notools = os.listdir(os.path.join(chembench_repo, "reports", "humans", "reports", "tool-disallowed"))
    human_files_notools = [
        os.path.join(chembench_repo, "reports", "humans", "reports", "tool-allowed", p) for p in human_files_notools
    ]
    human_files_notools = [p for p in human_files_notools if len(glob(os.path.join(p, "*.json"))) > 100]
    print(f"Found {len(human_files_notools)} files from human scorers when no tools allowed")

    return human_files_tools, human_files_notools


def obtain_scores_for_folder(folder, topic_frame, tool=False):
    chembench = obtain_chembench_repo()

    if tool:
        human_baseline_folder = os.path.join(chembench, "reports", "humans", "reports", "tool-allowed")
    else:
        human_baseline_folder = os.path.join(chembench, "reports", "humans", "reports", "tool-disallowed")

    datafolder = os.path.join(chembench, "data")

    scores = combine_scores_for_model(folder, datafolder, human_baseline_folder)

    scores = merge_with_topic_info(scores, topic_frame)

    return scores


def score_all_humans():
    topic_frame = pd.read_pickle(data / "questions.pkl")

    tool_folders, folders = find_all_human_scores()

    all_scores = {}
    for folder in folders:
        try:
            score = obtain_scores_for_folder(folder, topic_frame)
            name = Path(folder).stem
            all_scores[name] = score
        except Exception:
            pass

    for folder in tool_folders:
        try:
            score = obtain_scores_for_folder(folder, topic_frame, tool=True)
            name = "tool" + Path(folder).stem
            all_scores[name] = score
        except Exception:
            pass

    results = {}
    results["raw_scores"] = all_scores

    aggregrated_by_topic = {}
    grouped_scores = []
    for name, score in all_scores.items():
        grouped = score.groupby("topic")["all_correct_"].mean()
        aggregrated_by_topic[name] = grouped
        grouped_scores.append(grouped)

    grouped_scores_frame = pd.concat(grouped_scores).to_frame()
    mean_over_all_models = grouped_scores_frame.groupby("topic").mean()

    results["topic_mean"] = mean_over_all_models
    with open(data / "humans_as_models_scores.pkl", "wb") as handle:
        pickle.dump(results, handle)


if __name__ == "__main__":
    score_all_humans()

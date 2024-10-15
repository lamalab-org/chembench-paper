from chembench.analysis import (
    load_all_reports,
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
import pickle 

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


def find_all_human_scores(tools_allowed=True):

    chembench_repo = obtain_chembench_repo()

    if tools_allowed:
        human_files = os.listdir(os.path.join(chembench_repo, "reports", "humans", "reports", "tool-allowed"))
        human_files = [
            os.path.join(chembench_repo, "reports", "humans",  "reports", "tool-allowed", p) for p in human_files
        ]

        human_files = [p for p in human_files if len(glob(os.path.join(p, "*.json"))) == 127]
    else:
        human_files = os.listdir(os.path.join(chembench_repo, "reports", "humans", "reports", "tool-disallowed"))
        human_files = [
            os.path.join(chembench_repo, "reports", "humans",  "reports", "tool-disallowed", p) for p in human_files
        ]        

        human_files = [p for p in human_files if len(glob(os.path.join(p, "*.json"))) == 121]
    print(f"Found {len(human_files)} files from human scorers")
    return human_files


def obtain_scores_for_folder(folder, topic_frame, tools_allowed=True):
    chembench = obtain_chembench_repo()
    # Make all of these paths absolute
    if tools_allowed:
        human_baseline_folder = os.path.join(chembench, "reports/humans/reports/tool-allowed")
    else:
        human_baseline_folder = os.path.join(chembench, "reports/humans/reports/tool-disallowed")
    datafolder = os.path.join(chembench, "data")

    # Filter to only include files which have more than 100 json files in them
    scores = combine_scores_for_model(folder, datafolder, human_baseline_folder)

    scores = merge_with_topic_info(scores, topic_frame)

    return scores


def summarize_scores(all_scores):
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

    return results

def score_all_humans():
    topic_frame = pd.read_pickle(data / "questions.pkl")

    folders_w_tools = find_all_human_scores(tools_allowed=True)

    folders_wo_tools = find_all_human_scores(tools_allowed=False)

    all_scores_w_tool = {}
    for folder in folders_w_tools:
        try:
            score = obtain_scores_for_folder(folder, topic_frame, tools_allowed=True)
            name = Path(folder).stem
            all_scores_w_tool[name] = score
        except Exception:
            pass

    all_scores_wo_tool = {}
    for folder in folders_wo_tools:
        try:
            score = obtain_scores_for_folder(folder, topic_frame, tools_allowed=False)
            name = Path(folder).stem
            all_scores_wo_tool[name] = score
        except Exception:
            pass
    
    tool_scores = summarize_scores(all_scores_w_tool)
    no_tool_scores = summarize_scores(all_scores_wo_tool)

    with open(data / "humans_as_models_scores_tools.pkl", "wb") as handle:
        pickle.dump(tool_scores, handle)

    with open(data / "humans_as_models_scores_no_tools.pkl", "wb") as handle:
        pickle.dump(no_tool_scores, handle)

    combined = {}

    for k, v in all_scores_w_tool.items():
        combined_df = pd.concat([v, all_scores_wo_tool[k]]).reset_index(drop=True)
        combined[k] = combined_df

    combined = summarize_scores(combined)

    with open(data / "humans_as_models_scores_combined.pkl", "wb") as handle:
        pickle.dump(combined, handle)


if __name__ == "__main__":
    score_all_humans()

from glob import glob
from paths import output, data
from utils import obtain_chembench_repo
import numpy as np
import json
import os
import pickle


def obtain_human_scores():
    chembench_repo = obtain_chembench_repo()

    human_reports_dir = os.path.join(chembench_repo, "reports", "humans")

    human_scores_with_tools_files = glob(
        os.path.join(human_reports_dir, "*_h_tool*.json")
    )

    human_scores_without_tools_files = glob(
        os.path.join(human_reports_dir, "*_h_notool*.json")
    )

    human_scores_with_tools, human_scores_without_tools = [], []

    for file in human_scores_with_tools_files:
        with open(file, "r") as handle:
            d = json.load(handle)
            human_scores_with_tools.append(d["fraction_correct"])

    for file in human_scores_without_tools_files:
        with open(file, "r") as handle:
            d = json.load(handle)
            human_scores_without_tools.append(d["fraction_correct"])

    human_scores_with_tools = np.array(human_scores_with_tools)
    human_scores_without_tools = np.array(human_scores_without_tools)

    all_human_scores = np.concatenate(
        [human_scores_with_tools, human_scores_without_tools]
    )

    return human_scores_with_tools, human_scores_without_tools, all_human_scores

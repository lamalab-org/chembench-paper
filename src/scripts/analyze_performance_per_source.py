from paths import output
import pickle
import os
import numpy as np


def obtain_score_for_subset(df, subset):
    return df[df[subset]]["all_correct_"].mean()


subsets = ["is_point_group", "is_number_of_isomers", "is_number_nmr_peaks"]


def obtain_subset_scores(data_dict, outdir):
    for subset in subsets:
        for model, scores in data_dict.items():
            score = obtain_score_for_subset(data_dict[model])

            with open(os.path.join(outdir, f"{subset}_{model}.txt"), "w") as handle:
                handle.write(int(np.round(score * 100, 0)) + "\endinput")


def obtain_subset_scores_humans(data_dict, outdir):
    for subset in subsets:
        subset_scores = []
        for human, scores in data_dict.items():
            score = obtain_score_for_subset(data_dict[human])
            subset_scores.append(score)
        mean_subset_score = np.mean(subset_scores)
        with open(os.path.join(outdir, f"{subset}.txt"), "w") as handle:
            handle.write(int(np.round(mean_subset_score * 100, 0)) + "\endinput")


if __name__ == "__main__":
    with open(output, "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)

    outdir = os.path.join(output, "subset_scores")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    obtain_score_for_subset(model_scores["human_aligned"], outdir)

    with open(output, "humans_as_models_scores.pkl", "rb") as handle:
        human_scores = pickle.load(handle)

    outdir_humans = os.path.join(output, "human_subset_scores")
    if not os.path.exists(outdir_humans):
        os.makedirs(outdir)

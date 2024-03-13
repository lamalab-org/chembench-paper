from paths import output, data
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
                handle.write(np.round(score, 2) + "\endinput")


if __name__ == "__main__":
    with open(output, "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)

    outdir = os.path.join(output, "subset_scores")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    obtain_score_for_subset(model_scores, outdir)

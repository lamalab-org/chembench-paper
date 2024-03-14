from paths import output
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from paths import scripts, figures

plt.style.use(scripts / "lamalab.org")


def obtain_score_for_subset(df, subset):
    return df[df[subset]]["all_correct_"].mean()


subsets = [
    "is_point_group",
    "is_number_of_isomers",
    "is_number_nmr_peaks",
    "is_gfk",
    "is_dai",
    "is_pictograms",
    "is_name",
    "is_smiles_name",
    "is_organic_reactivity",
    "is_electron_counts",
    "is_chemical_compatibility",
    "is_materials_compatibility",
    "is_oup",
    "is_olympiad",
    "is_toxicology",
    "is_polymer_chemistry",
]


def obtain_subset_scores(data_dict, outdir):
    all_scores = []
    for subset in subsets:
        for model, scores in data_dict.items():
            score = obtain_score_for_subset(data_dict[model])

            with open(os.path.join(outdir, f"{subset}_{model}.txt"), "w") as handle:
                handle.write(int(np.round(score * 100, 0)) + "\endinput")

            all_scores.append({"model": model, "score": score, "subset": subset})

    return all_scores


def obtain_subset_scores_humans(data_dict, outdir):
    all_scores = []
    for subset in subsets:
        subset_scores = []
        for human, scores in data_dict.items():
            score = obtain_score_for_subset(data_dict[human])
            subset_scores.append(score)
        mean_subset_score = np.mean(subset_scores)
        with open(os.path.join(outdir, f"{subset}.txt"), "w") as handle:
            handle.write(int(np.round(mean_subset_score * 100, 0)) + "\endinput")
    all_scores.append({"model": "human", "score": score, "subset": subset})

    return all_scores


# todo: perhaps make a heatmap with performance and models

if __name__ == "__main__":
    with open(output, "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)

    outdir = os.path.join(output, "subset_scores")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_scores = obtain_subset_scores(model_scores["human_aligned"], outdir)

    with open(output, "humans_as_models_scores.pkl", "rb") as handle:
        human_scores = pickle.load(handle)

    outdir_humans = os.path.join(output, "human_subset_scores")
    if not os.path.exists(outdir_humans):
        os.makedirs(outdir)

    human_scores = obtain_subset_scores_humans(human_scores["raw_scores"])

    all_scores = pd.DataFrame(model_scores + human_scores)

    score_heatmap = all_scores.pivot_table(
        index="model", columns="subset", values="score", aggfunc="mean"
    )

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(score_heatmap, ax=ax)
    fig.savefig(figures / "performance_per_topic.pdf", bbox_inches="tight")

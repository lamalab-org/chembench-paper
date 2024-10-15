from paths import output
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from paths import scripts, figures, data
from loguru import logger
from definitions import MODELS_TO_PLOT

plt.style.use(scripts / "lamalab.mplstyle")


def obtain_score_for_subset(df, subset):
    return df[df[subset]]["all_correct_"].astype(int).mean()


subsets = [
    "is_point_group",
    "is_number_of_isomers",
    "is_number_nmr_peaks",
    "is_gfk",
    "is_dai",
    "is_pictograms",
    "is_name",
    "is_organic_reactivity",
    "is_electron_counts",
    "is_chemical_compatibility",
    "is_materials_compatibility",
    "is_oup",
    "is_olympiad",
    "is_toxicology",
    "is_polymer_chemistry",
    "is_classification",
    "is_number",
    "is_preference",
]

subset_clean_names = [
    "point group",
    "number of isomers",
    "number of NMR signals",
    "GFK (chemical safety)",
    "DAI (daily allowed intake)",
    "GHS pictograms",
    "name to SMILES",
    "organic reactivity",
    "electron counts",
    "chemical compatibility",
    "materials compatibility",
    "textbook",
    "chemistry olympiad",
    "toxicology",
    "polymer chemistry",
    "classification task",
    "numerical task",
    "chemical preference",
]


rename_dict = dict(zip(subsets, subset_clean_names))


def obtain_subset_scores(data_dict, outdir):
    all_scores = []
    for subset in subsets:
        for model, scores in data_dict.items():
            try:
                score = obtain_score_for_subset(data_dict[model], subset)

                with open(os.path.join(outdir, f"{subset}_{model}.txt"), "w") as handle:
                    handle.write(str(int(np.round(score * 100, 0))) + "\endinput")

                all_scores.append({"model": model, "score": score, "subset": subset})
            except Exception as e:
                logger.warning(f"Failed for {model} {subset} due to {e}")
                pass

    return all_scores


def obtain_subset_scores_humans(data_dict, outdir):
    all_scores = []
    for subset in subsets:
        subset_scores = []
        try:
            for human, scores in data_dict.items():
                score = obtain_score_for_subset(data_dict[human], subset)
                subset_scores.append(score)
        except Exception as e:
            logger.warning(f"Failed for {subset} due to {e}")
        try:
            logger.info(f"Human subset scores: {subset_scores} for {subset}")
            mean_subset_score = np.nanmean(subset_scores)
            with open(os.path.join(outdir, f"{subset}.txt"), "w") as handle:
                handle.write(
                    str(int(np.round(mean_subset_score * 100, 0))) + "\endinput"
                )
            all_scores.append(
                {"model": "human", "score": mean_subset_score, "subset": subset}
            )
        except Exception as e:
            logger.warning(f"Failed for {subset} due to {e}")

    print(all_scores)
    return all_scores


# todo: perhaps make a heatmap with performance and models

if __name__ == "__main__":
    with open(os.path.join(data, "model_score_dicts.pkl"), "rb") as handle:
        model_scores = pickle.load(handle)

    outdir = os.path.join(output, "subset_scores")

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # human aligned
    model_scores = obtain_subset_scores(model_scores["human_aligned_no_tool"], outdir)

    with open(
        os.path.join(data, "humans_as_models_scores_no_tools.pkl"), "rb"
    ) as handle:
        human_scores = pickle.load(handle)

    outdir_humans = os.path.join(output, "human_subset_scores")
    if not os.path.exists(outdir_humans):
        os.makedirs(outdir, exist_ok=True)

    human_scores = obtain_subset_scores_humans(
        human_scores["raw_scores"], outdir_humans
    )

    all_scores = pd.DataFrame(model_scores + human_scores)
    all_scores["subset"] = all_scores["subset"].map(rename_dict)
    all_scores = all_scores[all_scores["model"].isin(MODELS_TO_PLOT + ["human"])]
    score_heatmap = all_scores.pivot_table(
        index="model", columns="subset", values="score", aggfunc="mean", fill_value=0
    )
    print(score_heatmap)

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(score_heatmap, ax=ax, xticklabels=True, yticklabels=True, cmap="RdBu_r")
    # fig.tight_layout()
    fig.savefig(figures / "performance_per_topic_tiny.pdf", bbox_inches="tight")

    # overall
    with open(os.path.join(data, "model_score_dicts.pkl"), "rb") as handle:
        model_scores = pickle.load(handle)
    model_scores = obtain_subset_scores(model_scores["overall"], outdir)
    all_scores = pd.DataFrame(model_scores)
    all_scores = all_scores[all_scores["model"].isin(MODELS_TO_PLOT + ["human"])]
    all_scores["subset"] = all_scores["subset"].map(rename_dict)
    score_heatmap = all_scores.pivot_table(
        index="model", columns="subset", values="score", aggfunc="mean", fill_value=0
    )
    print(score_heatmap)

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(score_heatmap, ax=ax, xticklabels=True, yticklabels=True, cmap="RdBu_r")
    # fig.tight_layout()
    fig.savefig(figures / "performance_per_topic.pdf", bbox_inches="tight")

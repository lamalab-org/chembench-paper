import pickle as pkl
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from functools import lru_cache
from paths import scripts, data, figures, output
from scipy.stats import spearmanr
from plotutils import model_color_map, range_frame
from utils import ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, TWO_COL_WIDTH_INCH
from complexity import complexity_from_smiles
import pandas as pd
import os
from pymatgen.core import Composition

from collect_human_scores import obtain_human_scores
from definitions import MODELS_TO_PLOT

human_scores_with_tools, human_scores_without_tools, all_human_scores = (
    obtain_human_scores()
)

correlation_correlation_dir = output / "correlation_correlation"
correlation_correlation_dir.mkdir(exist_ok=True)

plt.style.use(scripts / "lamalab.mplstyle")

relevant_topics = ["is_number_nmr_peaks", "is_electron_counts", "is_number_of_isomers"]
mcq_topics = ["is_name"]

topic_to_representation = {
    "is_number_nmr_peaks": "smiles",
    "is_electron_counts": "molecular_formula",
    "is_point_group": "smiles",
    "is_number_of_isomers": "molecular_formula",
    "is_organic_reactivity": "smiles",
    "is_name": "smiles",
}

covariate_caption = {
    "num_atoms": "number of atoms",
    "complexity": "Böttcher complexity",
}


@lru_cache(maxsize=None)
def compute_molecular_mass(molecular_formula):
    try:
        composition = Composition(molecular_formula)
        return composition.weight
    except Exception as e:
        print(f"Error in computing molecular mass: {e}")
        return None


@lru_cache(maxsize=None)
def composition_to_num_atoms(composition):
    try:
        elements = re.findall(r"([A-Z][a-z]*)(\d*)", composition)
        return sum(int(count) if count else 1 for _, count in elements)
    except Exception as e:
        print(f"Error in computing number of atoms: {e}")
        return None


@lru_cache(maxsize=None)
def smiles_to_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


@lru_cache(maxsize=None)
def smiles_to_num_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()


def filter_data(questions, model, column, subset):
    return questions[subset][model][questions[subset][model][column]].copy()


def prepare_data(questions, model, column, subset="overall", type_mol="smiles"):
    questions_ = filter_data(questions, model, column, subset)
    print(
        f"Preparing data for {model} in {column}. Number of questions: {len(questions_)}"
    )

    if type_mol == "smiles":
        questions_["smiles"] = questions_["content"].str.extract(
            r"\[START_SMILES\](.*?)\[END_SMILES\]"
        )
        questions_ = questions_.dropna(subset=["smiles"])
        questions_["molecular_weight"] = questions_["smiles"].map(smiles_to_mass)
        questions_["num_atoms"] = questions_["smiles"].map(
            lambda x: Chem.MolFromSmiles(x).GetNumAtoms() if x else None
        )
        questions_["complexity"] = questions_["smiles"].map(
            lambda x: complexity_from_smiles(x) if x else None
        )
    elif type_mol == "molecular_formula":
        questions_["molecular_formula"] = questions_["content"].str.extract(
            r"\\ce\{(.*)\}"
        )
        questions_["molecular_formula_weight"] = questions_["molecular_formula"].map(
            lambda x: compute_molecular_mass(x) if x else None
        )
        questions_["num_atoms"] = questions_["molecular_formula"].map(
            lambda x: composition_to_num_atoms(x) if x else None
        )

    return questions_


def plot_correlation(ax, x, y, model, color, covariate, metric):
    ax.scatter(x, y, label=model, color=color, s=3, alpha=0.4)
    ax.set_title(model)

    spearman_corr = spearmanr(x, y)

    for suffix, value in [("", spearman_corr.statistic), ("_p", spearman_corr.pvalue)]:
        with open(
            os.path.join(
                correlation_correlation_dir,
                f"spearman_{covariate}_{model}_{metric}{suffix}.txt",
            ),
            "w",
        ) as handle:
            handle.write(f"{value:.2f}" + "\endinput")


def plot_correlations(questions, topics, metric, plot_function):
    # Create a dictionary for shorter model names
    short_names = {
        "GPT-3.5 Turbo": "GPT-3.5",
        "GPT-4o": "GPT-4o",
        "o1": "o1",
        "Claude-3.5 (Sonnet)": "Claude-3.5",
        "Llama-3.1-405B-Instruct": "Llama-3.1-405B",
        "Mistral-Large-2": "Mistral-Large-2",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
        "PaperQA2": "PaperQA2",
    }

    for topic in topics:
        type_mol = topic_to_representation[topic]
        covariates = (
            ["num_atoms", "complexity"] if type_mol == "smiles" else ["num_atoms"]
        )

        all_data = pd.concat(
            [
                prepare_data(questions, model, topic, type_mol=type_mol).assign(
                    model=model
                )
                for model in tqdm(MODELS_TO_PLOT)
            ]
        ).reset_index(drop=True)

        metric_data = pd.concat(
            [
                questions["overall"][model][questions["overall"][model][topic]][metric]
                for model in MODELS_TO_PLOT
            ]
        ).dropna()

        if metric_data.empty:
            print(
                f"No data available for topic {topic} and metric {metric}. Skipping..."
            )
            continue

        metric_without_outliers = metric_data

        for covariate in covariates:
            n_models = len(MODELS_TO_PLOT)
            n_rows, n_cols = 2, (n_models + 1) // 2

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(TWO_COL_WIDTH_INCH, 1.8 * ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
                sharex="all",
                sharey="all",
            )
            axes = axes.flatten()

            covariate_range = all_data[covariate]
            if covariate_range.empty:
                print(f"No data available for covariate {covariate}. Skipping...")
                plt.close(fig)
                continue

            upperlim = 200 if covariate == "complexity" else covariate_range.max()
            covariate_range_without_outliers = covariate_range[
                covariate_range <= upperlim
            ]

            for i, model in enumerate(MODELS_TO_PLOT):
                questions_ = all_data[all_data["model"] == model]
                if questions_.empty:
                    print(f"No data available for model {model}. Skipping...")
                    continue

                plot_function(
                    axes[i],
                    questions_[covariate],
                    questions_[metric],
                    short_names[model],
                    model_color_map[model],
                    covariate,
                    topic,
                )

                if (
                    not covariate_range_without_outliers.empty
                    and not metric_without_outliers.empty
                ):
                    range_frame(
                        axes[i],
                        covariate_range_without_outliers,
                        metric_without_outliers.values,
                        pad=0.2,
                    )
                else:
                    print(f"Not enough data to create range frame for model {model}.")

                # Remove title and add a text label instead
                axes[i].set_title("")
                axes[i].text(
                    0.5,
                    0.95,
                    short_names[model],
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[i].transAxes,
                )

            for j in range(i + 1, n_rows * n_cols):
                fig.delaxes(axes[j])

            fig.text(
                0.5,
                0.02,
                covariate_caption[covariate],
                ha="center",
                va="center",
            )
            fig.text(
                0.0,
                0.5,
                metric.replace("_", " "),
                ha="center",
                va="center",
                rotation="vertical",
            )

            fig.tight_layout()
            # plt.subplots_adjust(
            #     top=0.95, bottom=0.1, left=0.1, right=0.98, hspace=0.4, wspace=0.2
            # )

            fig.savefig(
                figures / f"correlation_plot_{topic}_{covariate}.pdf",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


def plot_correlations_num_atoms(questions):
    plot_correlations(questions, relevant_topics, "metrics_mae", plot_correlation)


def plot_mcq_correlations(questions):
    plot_correlations(questions, mcq_topics, "metrics_hamming", plot_correlation)


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as f:
        questions = pkl.load(f)

    with open(data / "humans_as_models_scores_combined.pkl", "rb") as f:
        human_scores = pkl.load(f)["raw_scores"]

    all_human_scores = pd.concat([v for k, v in human_scores.items()]).reset_index(
        drop=True
    )
    all_human_scores["model"] = "human"

    questions["overall"]["human"] = all_human_scores

    plot_correlations_num_atoms(questions)
    plot_mcq_correlations(questions)

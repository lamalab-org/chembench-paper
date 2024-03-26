import pickle as pkl
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import re
from functools import lru_cache
from paths import scripts, data, figures
from plotutils import model_color_map, range_frame
from utils import (
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
)
from complexity import complexity_from_smiles
import pandas as pd

plt.style.use(scripts / "lamalab.mplstyle")
relevant_models = ["gpt4", "claude3", "galactica_120b", "human"]

model_rename_dict = {
    "gpt4": "GPT-4",
    "claude3": "Claude 3",
    "llama70b": "Llama",
    "galactica_120b": "Galactica",
    "human": "Human",
}

relevant_topics = [
    "is_number_nmr_peaks",
    "is_electron_counts",
    "is_number_of_isomers",
]  # ["is_point_group",  "is_number_nmr_peaks"]

mcq_topics = ["is_organic_reactivity", "is_name"]  # , "is_smiles_name"]

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


@lru_cache()
def compute_molecular_mass(molecular_formula):
    # Dictionary of atomic masses
    atomic_masses = {
        "H": 1.00784,
        "C": 12.0107,
        "N": 14.0067,
        "O": 15.999,
        "P": 30.973762,
        "S": 32.065,
        "Cl": 35.453,
        "Mg": 24.305,
        "Na": 22.98,
        "B": 10.811,
        "F": 18.998,
        "Ta": 180.947,
        "Zn": 65.38,
        "Ag": 107.8682,
        "Au": 196.966569,
        "Al": 26.981538,
        "As": 74.92160,
        "Ba": 137.327,
        "Fe": 55.845,
        "I": 126.90447,
        "K": 39.0983,
        "Li": 6.94,
        "Mn": 54.938044,
        "Si": 28.0855,
        "Br": 79.904,
        "Ca": 40.078,
        "Cu": 63.546,
        "Hg": 200.592,
        "Ni": 58.6934,
        "Pb": 207.2,
    }

    # Parse the molecular formula
    elements = re.findall(r"([A-Z][a-z]*)(\d*)", molecular_formula)

    # Calculate the molecular mass
    molecular_mass = 0.0
    for element, count in elements:
        count = int(count) if count else 1
        molecular_mass += atomic_masses[element] * count

    return molecular_mass


@lru_cache()
def composition_to_num_atoms(composition):
    elements = re.findall(r"([A-Z][a-z]*)(\d*)", composition)
    num_atoms = 0
    for element, count in elements:
        count = int(count) if count else 1
        num_atoms += count
    return num_atoms


@lru_cache()
def smiles_to_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


@lru_cache()
def smiles_to_num_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()


def prepare_data(
    questions: dict,
    model: str,
    column: str = "is_point_group",
    subset: str = "overall",
    type_mol: str = "smiles",
):
    questions_ = questions[subset][model].copy()
    # filter by column=True
    questions_ = questions_[questions_[column]]
    # extract type_mol
    if type_mol == "smiles":
        questions_["smiles"] = questions_["question"].str.extract(
            r"\[START_SMILES\](.*?)\[END_SMILES\]"
        )
        questions_ = questions_.dropna(subset=["smiles"])
        questions_["molecular_weight"] = questions_["smiles"].map(smiles_to_mass)
        questions_["num_atoms"] = questions_["smiles"].apply(
            lambda x: (Chem.MolFromSmiles(x).GetNumAtoms() if x is not None else None)
        )
        questions_["complexity"] = questions_["smiles"].apply(
            lambda x: complexity_from_smiles(x) if x is not None else None
        )
    elif type_mol == "molecular_formula":
        questions_["molecular_formula"] = questions_["question"].str.extract(
            r"\\ce\{(.*)\}"
        )
        questions_["molecular_formula_weight"] = questions_["molecular_formula"].apply(
            lambda x: compute_molecular_mass(x) if x is not None else None
        )
        questions_["num_atoms"] = questions_["molecular_formula"].apply(
            lambda x: composition_to_num_atoms(x) if x is not None else None
        )

    return questions_


def plot_mcq_correlations(questions: dict):
    mcq_metric = "metrics_hamming"
    for topic in mcq_topics:
        type_mol = topic_to_representation[topic]
        covariates = (
            ["num_atoms"] if type_mol != "smiles" else ["num_atoms", "complexity"]
        )

        metric_for_subset = [
            questions["overall"][model][questions["overall"][model][topic]][
                "metrics_hamming"
            ]
            for model in relevant_models
        ]

        metric_without_outliers = pd.concat(metric_for_subset).dropna()
        metric_without_outliers = metric_without_outliers[
            metric_without_outliers < metric_without_outliers.quantile(0.95)
        ]

        for covariate in covariates:
            fig, ax = plt.subplots(
                1,
                len(relevant_models),
                figsize=(0.7 * TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
                sharex="row",
                sharey="row",
            )

            covariate_range = [
                prepare_data(questions, model, topic, type_mol=type_mol)[covariate]
                for model in relevant_models
            ]
            covariate_range = pd.concat(covariate_range).dropna()
            for i, model in enumerate(relevant_models):

                questions_ = prepare_data(questions, model, topic, type_mol=type_mol)

                print(f"Number of questions for {model}: {len(questions_)} in {topic}")
                ax[i].scatter(
                    questions_[covariate],
                    questions_[mcq_metric],
                    label=model_rename_dict[model],
                    color=model_color_map[model],
                    s=3,
                    alpha=0.4,
                )
                ax[i].title.set_text(model_rename_dict[model])

                range_frame(
                    ax[i],
                    covariate_range,
                    metric_without_outliers.values,
                )

            ax[0].set_ylabel(mcq_metric)
            # set xlabel in the middle
            fig.text(0.5, -0.01, covariate_caption[covariate], ha="center")

            fig.tight_layout()

            fig.savefig(
                figures / f"correlation_plot_{topic}_{covariate}.pdf",
                bbox_inches="tight",
            )


def plot_correlations_num_atoms(questions: dict):

    for topic in relevant_topics:
        type_mol = topic_to_representation[topic]
        covariates = (
            ["num_atoms"] if type_mol != "smiles" else ["num_atoms", "complexity"]
        )

        mae_for_subset = [
            questions["overall"][model][questions["overall"][model][topic]][
                "metrics_mae"
            ]
            for model in relevant_models
        ]

        mae_without_outliers = pd.concat(mae_for_subset).dropna()
        mae_without_outliers = mae_without_outliers[
            mae_without_outliers < mae_without_outliers.quantile(0.95)
        ]

        for covariate in covariates:
            fig, ax = plt.subplots(
                1,
                len(relevant_models),
                figsize=(0.7 * TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
                sharex="row",
                sharey="row",
            )

            covariate_range = [
                prepare_data(questions, model, topic, type_mol=type_mol)[covariate]
                for model in relevant_models
            ]
            covariate_range = pd.concat(covariate_range).dropna()
            for i, model in enumerate(relevant_models):

                questions_ = prepare_data(questions, model, topic, type_mol=type_mol)

                print(f"Number of questions for {model}: {len(questions_)} in {topic}")
                ax[i].scatter(
                    questions_[covariate],
                    questions_["metrics_mae"],
                    label=model_rename_dict[model],
                    color=model_color_map[model],
                    s=3,
                    alpha=0.4,
                )
                ax[i].title.set_text(model_rename_dict[model])

                range_frame(
                    ax[i],
                    covariate_range,
                    mae_without_outliers.values,
                )

            ax[0].set_ylabel("mean absolute error")
            # set xlabel in the middle
            fig.text(0.5, -0.01, covariate_caption[covariate], ha="center")

            fig.tight_layout()

            fig.savefig(
                figures / f"correlation_plot_{topic}_{covariate}.pdf",
                bbox_inches="tight",
            )


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as f:
        questions = pkl.load(f)

    with open(data / "humans_as_models_scores.pkl", "rb") as f:
        human_scores = pkl.load(f)

    human_scores = human_scores["raw_scores"]
    all_human_score_dicts = [v for k, v in human_scores.items()]
    all_human_scores = pd.concat(all_human_score_dicts)
    all_human_scores["model"] = "human"

    questions["overall"]["human"] = all_human_scores

    plot_correlations_num_atoms(questions)
    plot_mcq_correlations(questions)

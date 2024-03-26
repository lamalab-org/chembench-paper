import pickle as pkl
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import re

from paths import scripts, data, figures
from plotutils import model_color_map, range_frame
from utils import ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH

plt.style.use(scripts / "lamalab.mplstyle")
relevant_models = ["gpt4", "claude3", "llama70b"]

model_rename_dict = {"gpt4": "GPT-4", "claude3": "Claude 3", "llama70b": "Llama (70b)"}

relevant_topics = [
    "is_number_nmr_peaks",
]  # ["is_point_group",  "is_number_nmr_peaks"]


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


def composition_to_num_atoms(composition):
    elements = re.findall(r"([A-Z][a-z]*)(\d*)", composition)
    num_atoms = 0
    for element, count in elements:
        count = int(count) if count else 1
        num_atoms += count
    return num_atoms


def smiles_to_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


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
    questions = questions[subset][model]
    # filter by column=True
    questions = questions[questions[column]]
    # extract type_mol
    if type_mol == "smiles":
        questions["smiles"] = questions["question"].str.extract(
            r"\[START_SMILES\](.*)\[END_SMILES\]"
        )
        questions = questions.dropna(subset=["smiles"])
        questions["molecular_weight"] = questions["smiles"].map(smiles_to_mass)
        questions["num_atoms"] = questions["smiles"].apply(
            lambda x: (Chem.MolFromSmiles(x).GetNumAtoms() if x is not None else None)
        )
    elif type_mol == "molecular_formula":
        questions["molecular_formula"] = questions["question"].str.extract(
            r"\\ce\{(.*)\}"
        )
        questions["molecular_formula_weight"] = questions["molecular_formula"].apply(
            lambda x: compute_molecular_mass(x) if x is not None else None
        )
        questions["num_atoms"] = questions["molecular_formula"].apply(
            lambda x: composition_to_num_atoms(x) if x is not None else None
        )

    return questions


def plot_correlations_num_atoms(questions: dict):

    for topic in relevant_topics:
        fig, ax = plt.subplots(
            1,
            len(relevant_models),
            figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
            sharex=True,
            sharey=True,
        )
        for i, model in enumerate(relevant_models):
            questions_ = prepare_data(questions, model, topic)
            ax[i].scatter(
                questions_["num_atoms"],
                questions_["metrics_mae"],
                label=model_rename_dict[model],
                color=model_color_map[model],
                s=3,
            )
            ax[i].title.set_text(model_rename_dict[model])

            range_frame(
                ax[i],
                questions_["num_atoms"].dropna().values,
                questions_["metrics_mae"].dropna().values,
            )

        ax[0].set_ylabel("mean absolute error")
        # set xlabel in the middle
        fig.text(0.6, -0.01, "number of atoms", ha="center")

        fig.tight_layout()

        fig.savefig(
            figures / f"correlation_plot_{topic}_num_atoms.pdf", bbox_inches="tight"
        )


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as f:
        questions = pkl.load(f)

    print(questions.keys())

    plot_correlations_num_atoms(questions)

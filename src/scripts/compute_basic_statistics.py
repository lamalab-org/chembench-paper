from utils import obtain_chembench_repo
from chembench.analysis import classify_questions
from paths import output
from glob import glob
import os
import pandas as pd


def load_data():
    # Load the data
    chembench_repo = obtain_chembench_repo()

    # Load the data
    data = f"{chembench_repo}/data"

    df = classify_questions(data, debug=True)

    total_number_of_questions = len(df)

    with open(output / "total_number_of_questions.txt", "w") as f:
        f.write(str(total_number_of_questions) + "\endinput")

    automatically_generated = df[df["is_semiautomatically_generated"]]

    with open(output / "automatically_generated.txt", "w") as f:
        f.write(str(len(automatically_generated)) + "\endinput")

    manually_generated = df[~df["is_semiautomatically_generated"]]
    with open(output / "manually_generated.txt", "w") as f:
        f.write(str(len(manually_generated)) + "\endinput")

    mcq_questions = df[df["is_classification"]]
    with open(output / "mcq_questions.txt", "w") as f:
        f.write(str(len(mcq_questions)) + "\endinput")

    non_mcq_questions = df[~df["is_classification"]]
    with open(output / "non_mcq_questions.txt", "w") as f:
        f.write(str(len(non_mcq_questions)) + "\endinput")

    df.to_csv(output / "questions.csv")
    df.to_pickle(output / "questions.pkl")

    dai_data = glob(
        os.path.join(chembench_repo, "data", "safety", "pubchem_data", "DAI*.json")
    )
    h_statements = glob(
        os.path.join(chembench_repo, "data", "safety", "pubchem_data", "h_state*.json")
    )
    pictograms = glob(
        os.path.join(
            chembench_repo, "data", "safety", "pubchem_data", "pictogram*.json"
        )
    )

    with open(output / "num_dai.txt", "w") as f:
        f.write(str(len(dai_data)) + "\endinput")

    with open(output / "num_h_statements.txt", "w") as f:
        f.write(str(len(h_statements)) + "\endinput")

    with open(output / "num_pictograms.txt", "w") as f:
        f.write(str(len(pictograms)) + "\endinput")

    # read number of questions in "tiny" subset
    df = pd.read_csv(os.path.join(chembench_repo, "reports", "humans", "questions.csv"))

    with open("num_tiny_questions.txt", "w") as f:
        f.write(str(len(df)) + "\endinput")


if __name__ == "__main__":
    load_data()

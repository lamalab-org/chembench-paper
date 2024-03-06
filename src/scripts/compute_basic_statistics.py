from utils import obtain_chembench_repo
from chembench.analysis import classify_questions
from paths import output


def load_data():
    # Load the data
    chembench_repo = obtain_chembench_repo()

    # Load the data
    data = f"{chembench_repo}/data"

    df = classify_questions(data, debug=True)

    total_number_of_questions = len(df)

    with open(output / "total_number_of_questions.txt", "w") as f:
        f.write(str(total_number_of_questions))

    automatically_generated = df[df["is_semiautomatically_generated"]]

    with open(output / "automatically_generated.txt", "w") as f:
        f.write(str(len(automatically_generated)))

    manually_generated = df[~df["is_semiautomatically_generated"]]
    with open(output / "manually_generated.txt", "w") as f:
        f.write(str(len(manually_generated)))
 
    df.to_csv(output / "questions.csv")
    df.to_pickle(output / "questions.pkl")


if __name__ == "__main__":
    load_data()

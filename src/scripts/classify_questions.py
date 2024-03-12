from utils import obtain_chembench_repo
from chembench.analysis import classify_questions
from paths import output, data


def classify():
    # Load the data
    chembench_repo = obtain_chembench_repo()

    # Load the data
    data_ = f"{chembench_repo}/data"

    df = classify_questions(data_)  # set debug = True to speed up

    df.to_csv(data / "questions.csv")
    df.to_pickle(data / "questions.pkl")


if __name__ == "__main__":
    classify()

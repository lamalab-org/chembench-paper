from utils import obtain_chembench_repo
from paths import output, data
import os
import pandas as pd 

def classify():
    # Load the data
    chembench_repo = obtain_chembench_repo()

    # Load the data
    data_ = os.path.join(chembench_repo, "data")

    df_categories = pd.read_csv(os.path.join(chembench_repo, "scripts", "classified_questions.csv"))    

    def update_path(path):
        # the dataframe contains paths like '../data/questions/00000000-0000-0000-0000-000000000000.json'
        # we want to get the path like chembench_repo/data/questions/00000000-0000-0000-0000-000000000000.json
        path_without_dots = path.replace("../data", "")
        return os.path.join(data_, path_without_dots)
    
    df_categories['path'] = df_categories['path'].apply(update_path)
    
    # now, keep those rows where the path exists
    df = df_categories[df_categories['path'].apply(lambda x: os.path.exists(x))].copy()

    df.to_csv(data / "questions.csv")
    df.to_pickle(data / "questions.pkl")


if __name__ == "__main__":
    classify()

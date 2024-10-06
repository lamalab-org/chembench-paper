from utils import obtain_chembench_repo
from paths import output, data
import os
import pandas as pd 
import json
from chembench.task import Task
from chembench.analysis import add_group_info, is_semiautomatically_generated
def classify():
    # Load the data
    chembench_repo = obtain_chembench_repo()

    # Load the data
    data_ = os.path.join(chembench_repo, "data")

    df_categories = pd.read_csv(os.path.join(chembench_repo, "scripts", "classified_questions.csv"))    

    def update_path(path):
        # the dataframe contains paths like '../data/questions/00000000-0000-0000-0000-000000000000.json'
        # we want to get the path like chembench_repo/data/questions/00000000-0000-0000-0000-000000000000.json
        
        path_without_dots = path.replace("../data/", "")
        return os.path.join(data_, path_without_dots)
    
    df_categories['name'] = df_categories['index']
    df_categories['path'] = df_categories['question'].apply(update_path)
    df_categories = df_categories[df_categories['path'].apply(lambda x: os.path.exists(x))].copy()
    df_categories['is_semiautomatically_generated'] = df_categories.apply(is_semiautomatically_generated, axis=1)

    #import pdb; pdb.set_trace()
    is_mcq = []
    keywords = []
    for path in df_categories['path']:
        task = Task.from_json(path)
        is_mcq.append('multiple_choice_grade' in task._metrics)
        keywords.append(task._keywords)

    df_categories['is_classification'] = is_mcq
    df_categories['keywords'] = keywords

    # now, keep those rows where the path exists
    
    df = add_group_info(df_categories)
    

    df.to_csv(data / "questions.csv")
    df.to_pickle(data / "questions.pkl")


if __name__ == "__main__":
    classify()


import os
import json

from paths import output
import pandas as pd
import pickle

def load_model_refusal(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def list_of_dicts_to_latex_table(data, output_file):
    df = pd.DataFrame(data)
    latex_table = df.to_latex(index=False)

    with open(output_file, 'w') as f:
        f.write(latex_table)

if __name__ == "__main__":
    refusal_results = load_model_refusal(
        os.path.join(
            data, 
            'model_refusal_and_extraction_count.pkl'
        )
    )

    refusal_results = sorted(refusal_results, key=lambda x: x["model"])

    list_of_dicts_to_latex_table(
        model_refusal, 
        os.path.join(
            output, 
            'model_refusal_table.tex'
        )
    )
    
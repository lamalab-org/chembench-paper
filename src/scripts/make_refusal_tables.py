
import os
import json

from paths import  data, output
import pandas as pd
import pickle

model_file_name_to_label = {
    "claude2": "Claude 2",
    "claude3": "Claude 3",
    "claude3.5": "Claude 3.5 Sonnet",
    "command-r+": "Command R+",
    "galactica_120b": "Galactica 120B",
    "gemini-pro": "Gemini Pro",
    "gemma-1-1-7b-it": "Gemma 1.1 7B",
    "gemma-1-1-7b-it-T-one": "Gemma 1.1 7B Temp=1",
    "gemma-2-9b-it": "Gemma 2 9B",
    "gemma-2-9b-it-T-one": "Gemma 2 9B Temp=1",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "o1": "o1",
    "llama2-70b-chat": "Llama 2 70B Chat",
    "llama3-70b-instruct": "Llama 3 70B",
    "llama3-70b-instruct-T-one": "Llama 3 70B Temp=1",
    "llama3-8b-instruct": "Llama 3 8B",
    "llama3-8b-instruct-T-one": "Llama 3 8B Temp=1",
    "llama3.1-405b-instruct": "Llama 3.1 405B",
    "llama3.1-70b-instruct": "Llama 3.1 70B",
    "llama3.1-70b-instruct-T-one": "Llama 3.1 70B Temp=1",
    "llama3.1-8b-instruct": "Llama 3.1 8B",
    "llama3.1-8b-instruct-T-one": "Llama 3.1 8B Temp=1",
    "mixtral-8x7b-instruct": "Mixtral 8x7B",
    "mixtral-8x7b-instruct-T-one": "Mixtral 8x7B Temp=1",
    "mistral-large-2-123b": "Mistral Large 2 123B",
    "paper-qa": "Paper QA",
    "phi-3-medium-4k-instruct": "Phi 3 Medium 4K",
}

def load_model_refusal(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def list_of_dicts_to_latex_table(data, output_file):
    data = [item for item in data if item['model'] != 'random_baseline']
    data = [item for item in data if not '_' in item['model'] and not '=' in item['model']]
    for item in data:
        item['model'] = model_file_name_to_label.get(item['model'], item['model'])

    df = pd.DataFrame(data)
    latex_table = df.to_latex(index=False)

    latex_table = latex_table.replace(
        r"\begin{tabular}{lrrrr}",
        r"\begin{tabular}{lcccc}"
    )
    # Manually modify the LaTeX code for centering multi-level headers
    latex_table = latex_table.replace(
        r"model & questions_refused & refusal_fraction & questions_extracted & extractions_fraction \\",
        r"""\multirow{3}{*}{Model} & \multicolumn{2}{c}{\textbf{Refusal}} & \multicolumn{2}{c}{\textbf{LLM Extraction}}\\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}\\
& Nº of Questions & Fraction & Nº of Questions & Fraction\\"""
    )

    with open(output_file, 'w') as f:
        f.write(latex_table)

if __name__ == "__main__":
    refusal_results = load_model_refusal(
        os.path.join(
            output,
            'model_refusal_and_extraction_count.pkl'
        )
    )

    refusal_results = sorted(refusal_results, key=lambda x: x["model"])

    list_of_dicts_to_latex_table(
        refusal_results,
        os.path.join(
            output,
            'model_refusal_table.tex'
        )
    )

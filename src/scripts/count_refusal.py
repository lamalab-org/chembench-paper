import os
import json

from loguru import logger

from chembench.analysis import (
    get_human_scored_questions_with_at_least_n_scores,
    load_all_reports,
)

from glob import glob
import os
from pathlib import Path
from utils import obtain_chembench_repo
from paths import output
from loguru import logger

from glob import glob
import os
import matplotlib.pyplot as plt
from paths import output, scripts
import math
from loguru import logger

plt.style.use(scripts / "lamalab.mplstyle")

human_dir = output / "human_scores"
model_subset_dir = output / "human_subset_model_scores"
model_dir = output / "overall_model_scores"

model_file_name_to_label = {
    "claude2": "Claude 2",
    "claude3": "Claude 3",
    "claude3.5": "Claude 3.5 Sonnet",
    "command-r+": "Command R+",
    "galatica_120b": "Galactica 120B",
    "gemini-pro": "Gemini Pro",
    "gemma-1-1-7b-it": "Gemma 1.1 7B Instruct",
    "gemma-1-1-7b-it-T-one": "Gemma 1.1 7B Instruct T-One",
    "gemma-2-9b-it": "Gemma 2 9B Instruct",
    "gemma-2-9b-it-T-one": "Gemma 2 9B Instruct T-One",
    "gpt-3.5-turbo": "GPT-3.5-Turbo",
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "o1": "OpenAI o1",
    "llama2-70b-chat": "Llama 2 70B Chat",
    "llama3-70b-instruct": "Llama 3 70B Instruct",
    "llama3-70b-instruct-T-one": "Llama 3 70B Instruct T-One",
    "llama3-8b-instruct": "Llama 3 8B Instruct",
    "llama3-8b-instruct-T-one": "Llama 3 8B Instruct T-One",
    "llama3.1-405b-instruct": "Llama 3.1 405B Instruct",
    "llama3.1-70b-instruct": "Llama 3.1 70B Instruct",
    "llama3.1-70b-instruct-T-one": "Llama 3.1 70B Instruct T-One",
    "llama3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "llama3.1-8b-instruct-T-one": "Llama 3.1 8B Instruct T-One",
    "mixtral-8x7b-instruct": "Mixtral 8x7B Instruct",
    "mixtral-8x7b-instruct-T-one": "Mixtral 8x7B Instruct T-One",
    "mistral-large-2-123b": "Mistral Large 2 123B",
    "paper-qa": "Paper QA",
    "phi-3-medium-4k-instruct": "Phi 3 Medium 4K Instruct",
    "random_baseline": "Random Baseline"
}

def count_refusal_model(folder, datafolder, human_baseline_folder=None, min_human_responses: int = 4):
    try:
        df = load_all_reports(folder, datafolder)
        if df.empty:
            raise ValueError("No data loaded from reports")

        if human_baseline_folder is not None:
            relevant_questions = get_human_scored_questions_with_at_least_n_scores(human_baseline_folder, min_human_responses)
            df = df[df[("name", 0)].isin(relevant_questions)]

        logger.info(f"Loaded {len(df)} rows of data")
        refusal_count = 0
        extraction_count = 0
        refusals = []
        extractions = []
        for i, row in df.iterrows():
            refusal = row[("triggered_refusal", 0)]
            llm_extraction = row[("llm_extraction_count", 0)]
            if refusal == True:
                refusal_count += 1
                refusals.append(
                    {
                        "question_name": row[("name", 0)],
                        "completion": row["output"]["text"]
                    }
                )
            if llm_extraction != 0:
                extraction_count += llm_extraction
                extractions.append(
                    {
                        "question_name": row[("name", 0)],
                        "completion": row["output"]["text"]
                    }
                )
                print(llm_extraction)

        fraction_refusal = refusal_count / len(df) if len(df) > 0 else 0
        fraction_llm_extraction = extraction_count / len(df) if len(df) > 0 else 0

        refusals_ = {
            "refusal_fraction": fraction_refusal,
            "questions_refused": refusal_count,
            "refused": refusals
        }

        extractions_ = {
            "extractions_fraction": fraction_llm_extraction,
            "questions_extracted": extraction_count,
            "extracted": extractions
        }

        return refusals_, extractions_


    except Exception as e:
        logger.error(f"Error in count_refusal_model: {str(e)}")
        return {"error": str(e), "fraction_correct": 0, "model_scores": []}

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    reports = glob(
        os.path.join(chembench_repo, "reports",
                     "mixtral-8x7b-instruct-T-one", "reports", "**", "*.json")
    )

    models = list(set([Path(p).parent for p in reports]))
    logger.info(f"Found {len(models)} reports")
    logger.debug(f"First 5 reports: {models[:5]}")
    outpath = os.path.join(output)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    datafolder = os.path.join(chembench_repo, "data")
    output_file_refusal = os.path.join(outpath, "count_refusal.txt")
    output_file_extraction = os.path.join(outpath, "count_extraction.txt")
    json_dir_refusals = os.path.join(outpath, "refusals_files")
    json_dir_extractions = os.path.join(outpath, "extractions_files")
    if not os.path.exists(json_dir_refusals):
        os.mkdir(json_dir_refusals)
    if not os.path.exists(json_dir_extractions):
        os.mkdir(json_dir_extractions)

    model_refusal = {}
    model_llm_extraction = {}
    for file in models:
        try:
            p = Path(file).parts[-3]
            logger.info(f"Running for model {p}")
            refusal, extraction = count_refusal_model(file, datafolder)
            model_refusal[p] = refusal["questions_refused"]
            model_llm_extraction[p] = extraction["questions_extracted"]
            outfile_refusal = os.path.join(json_dir_refusals, p + ".json")
            with open(outfile_refusal, "w") as f:
                json.dump(refusal, f, indent=2)
            outfile_extraction = os.path.join(json_dir_extractions, p + ".json")
            with open(outfile_extraction, "w") as f:
                json.dump(extraction, f, indent=2)
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            pass

    model_refusal = [
        (model_file_name_to_label[k], v) for k, v in model_refusal.items() if k in model_file_name_to_label
    ]

    with open(output_file_refusal, "w") as file:
        for item in model_refusal:
            file.write(f"{item}\n")

    model_llm_extraction = [
        (model_file_name_to_label[k], v) for k, v in model_llm_extraction.items() if k in model_file_name_to_label
    ]

    with open(output_file_extraction, "w") as file:
        for item in model_llm_extraction:
            file.write(f"{item}\n")
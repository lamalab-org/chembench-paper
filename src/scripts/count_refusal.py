import os
import json

from loguru import logger

import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from fastcore.basics import basic_repr
from loguru import logger

from chembench.analysis import (
    get_human_scored_questions_with_at_least_n_scores,
    load_all_reports,
    construct_name_to_path_dict,
)

from chembench.constant import (
    FLOATQ_REGEX_TEMPLATE_1,
    MCQ_REGEX_TEMPLATE_1,
    MCQ_REGEX_TEMPLATE_2,
    NUM_REGEX,
)
from chembench.extractor import LLMExtractor
from chembench.metrics import _AGGREGATION_MAP, _METRIC_MAP, classification_scores
from chembench.report import Report
from chembench.settings import MODEL_POST_PROCESS_ENV
from chembench.task import Task
from chembench.utils import (
    LangChainChatModelWrapper,
    create_multiple_choice_confidence_prompt,
    create_multiple_choice_prompt,
    create_multiple_choice_prompt_completion,
    create_multiple_choice_regex,
    detect_refusal,
    general_prompt,
    general_prompt_completion,
    general_prompt_confidence,
    passthrough,
    post_process_prompts,
    prepare_general_answer,
    prepare_mcq_answer,
    remove_ce,
    remove_math,
    remove_pu,
    remove_rxnsmiles,
    remove_smiles,
    run_regex,
)
from chembench.watermark import get_watermark

from chembench.extractor import LLMExtractor

from glob import glob
import os
from pathlib import Path
from utils import obtain_chembench_repo
from paths import output
from loguru import logger

import matplotlib.pyplot as plt
from paths import output, scripts
import math
from loguru import logger
from dotenv import load_dotenv

load_dotenv("/.env", override=True)


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
}

def _prompts_with_choices_it(examples: List[dict]):
    prompts = []
    score_maps = []
    for example in examples:
        prompt, answer_to_score = create_multiple_choice_prompt(
            example,
            cot=False,
            random_seed=42,
            permute=False,
        )
        score_maps.append(answer_to_score)
    return prompts, score_maps

def _prompts_with_choices(examples: List[dict]):
    prompts = []
    score_maps = []
    for example in examples:
        prompt, answer_to_score = create_multiple_choice_prompt_completion(
            example,
            cot=False,
            random_seed=42,
            permute=False,
        )
        score_maps.append(answer_to_score)
    return prompts, score_maps

def extract_examples(name, datafolder):
    names = construct_name_to_path_dict(datafolder)
    if name in names:
        data_path = names[name]
        with open(data_path, "r") as file:
            data = json.load(file)

        return data["examples"]

    else:
        logger.error(f"Cannot find the name for {name}")
        return []

def check_llm_fallback(data, datafolder, model):
    name = data[("name", 0)]
    _examples = extract_examples(data[("name", 0)], datafolder)
    if model == "galatica_120b":
        _mcq_regex_template = MCQ_REGEX_TEMPLATE_2
        _floatq_regex_template = NUM_REGEX
    else:
        _mcq_regex_template = MCQ_REGEX_TEMPLATE_1
        _floatq_regex_template = FLOATQ_REGEX_TEMPLATE_1 

    completion_text = data["output"]["text"]
    

    if not np.isnan(data["metrics"]["hamming"]):
        _metrics = ['multiple_choice_grade']
        preferred_score = "multiple_choice_grade"
        if model == "galatica_120b":
            prompts, score_maps = _prompts_with_choices(_examples)
        else:
            prompts, score_maps = _prompts_with_choices_it(_examples)
    else:
        _metrics = ['exact_str_match', 'mae', 'mse']
        preferred_score = "mae"

    llm_extractor = LLMExtractor(
        model_type="anthropic",
        seed=42,
        permute_options=False,
        temperature=0,
    )
    llm_extractor.llm_extraction_count = 0
    for metric in _metrics:
        if metric == "multiple_choice_grade":
            score_map = score_maps[0]
            prepared_output = prepare_mcq_answer(
                completion_text,
                _mcq_regex_template,
                example=_examples[0],
                llm_extractor=llm_extractor,
            )
        else: 
            if "target_scores" in _examples[0]:
                answer = run_regex(
                    create_multiple_choice_regex(list(score_map.keys())),
                    prepare_mcq_answer(
                        completion_text,
                        _mcq_regex_template,
                        example=_examples[0],
                        llm_extractor=llm_extractor,
                    ),
                    return_first=False,
                )
            else:
                answer = prepare_general_answer(
                    completion_text,
                    pattern=_floatq_regex_template,
                    llm_extractor=llm_extractor,
                )
    
    return llm_extractor.llm_extraction_count

def count_refusal_error(folder, datafolder, model):

    logger.info(f"Working in model {model}")
    df = load_all_reports(folder, datafolder)
    if df.empty:
        raise ValueError("No data loaded from reports")


    logger.info(f"Loaded {len(df)} rows of data")
    refusal_count = 0
    extraction_count = 0
    refusals = []
    extractions = []

    for i, row in df.iterrows():
        completion_text = row["output"]["text"]
        refusal = detect_refusal(
                completion_text,
                model_based_refusal_detection=False,
            )
        # If refusal, count and save the question
        # Then go to the next loop item since we assume no answer is provided
        if refusal == True:
            refusal_count += 1
            refusals.append(
                {
                    "question_name": row[("name", 0)],
                    "completion": row["output"]["text"]
                }
            )
            continue
        else:
            fallback = check_llm_fallback(row, datafolder, model)
            if fallback > 0:
                extraction_count += fallback

                extractions.append(
                    {
                        "question_name": row[("name", 0)],
                        "completion": row["output"]["text"]
                    }
                )

    fraction_refusal = refusal_count / len(df) if len(df) > 0 else 0
    fraction_llm_extraction = extraction_count / len(df) if len(df) > 0 else 0
    
    logger.info(f"Fraction refusal for model {model}: {fraction_refusal} ({refusal_count} questions)")
    logger.info(f"Fraction llm fallback for model {model}: {fraction_llm_extraction} ({extraction_count} questions)")

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
                     "**", "reports", "**", "*.json")
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
            # refusal, extraction = count_refusal_model(file, datafolder)
            refusal, extraction = count_refusal_error(file, datafolder, p)
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
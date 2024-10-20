import os
import json
from typing import List
import pickle
import numpy as np
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger
from dotenv import load_dotenv

from chembench.analysis import load_all_reports, construct_name_to_path_dict
from chembench.extractor import LLMExtractor
from chembench.utils import (
    detect_refusal,
    prepare_mcq_answer,
    prepare_general_answer,
    run_regex,
    create_multiple_choice_regex,
    create_multiple_choice_prompt,
    create_multiple_choice_prompt_completion
)
from chembench.constant import (
    FLOATQ_REGEX_TEMPLATE_1,
    MCQ_REGEX_TEMPLATE_1,
    MCQ_REGEX_TEMPLATE_2,
    NUM_REGEX,
)
from utils import obtain_chembench_repo

load_dotenv("/.env", override=True)

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

def check_llm_fallback(data, datafolder, model):
    name = data[("name", 0)]
    _examples = extract_examples(name, datafolder)
    if model == "galactica_120b":
        _mcq_regex_template = MCQ_REGEX_TEMPLATE_2
        _floatq_regex_template = NUM_REGEX
    else:
        _mcq_regex_template = MCQ_REGEX_TEMPLATE_1
        _floatq_regex_template = FLOATQ_REGEX_TEMPLATE_1

    completion_text = data["output"]["text"]

    if not np.isnan(data["metrics"]["hamming"]):
        _metrics = ['multiple_choice_grade']
        preferred_score = "multiple_choice_grade"
        if model == "galactica_120b":
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

    # Drop the 'targets_' column if it exists
    if "targets_" in df.columns:
        df = df.drop(columns=["targets_"])

    logger.info(f"Loaded {len(df)} rows of data")
    refusal_count = 0
    extraction_count = 0
    refusals = []
    extractions = []

    for _, row in df.iterrows():
        completion_text = row["output"]["text"]
        question_name = row[("name", 0)]

        refusal = detect_refusal(
            completion_text,
            model_based_refusal_detection=False,
        )
        if refusal:
            refusal_count += 1
            refusals.append({
                "question_name": question_name,
                "completion": completion_text
            })
        else:
            fallback = check_llm_fallback(row, datafolder, model)
            if fallback > 0:
                extraction_count += fallback
                extractions.append({
                    "question_name": question_name,
                    "completion": completion_text
                })

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

def process_model(file, datafolder, json_dir_refusals, json_dir_extractions):
    try:
        p = Path(file).parts[-3]
        logger.info(f"Running for model {p}")
        refusal, extraction = count_refusal_error(file, datafolder, p)
        result = {
            "model": p,
            "questions_refused": refusal["questions_refused"],
            "refusal_fraction": refusal["refusal_fraction"],
            "questions_extracted": extraction["questions_extracted"],
            "extractions_fraction": extraction["extractions_fraction"]
        }

        outfile_refusal = os.path.join(json_dir_refusals, f"{p}.json")
        with open(outfile_refusal, "w") as f:
            json.dump(refusal, f, indent=2)

        outfile_extraction = os.path.join(json_dir_extractions, f"{p}.json")
        with open(outfile_extraction, "w") as f:
            json.dump(extraction, f, indent=2)

        return result
    except Exception as e:
        logger.error(f"Error processing {file}: {e}")
        return None

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    reports = glob(os.path.join(chembench_repo, "reports", "**", "reports", "**", "*.json"))

    models = list(set([Path(p).parent for p in reports]))
    logger.info(f"Found {len(models)} reports")
    logger.debug(f"First 5 reports: {models[:5]}")

    outpath = os.path.join(chembench_repo, "output")
    os.makedirs(outpath, exist_ok=True)

    datafolder = os.path.join(chembench_repo, "data")
    json_dir_refusals = os.path.join(outpath, "refusals_files")
    json_dir_extractions = os.path.join(outpath, "extractions_files")
    os.makedirs(json_dir_refusals, exist_ok=True)
    os.makedirs(json_dir_extractions, exist_ok=True)

    results = []
    with ProcessPoolExecutor() as executor:
        future_to_model = {executor.submit(process_model, file, datafolder, json_dir_refusals, json_dir_extractions): file for file in models}
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.error(f'{model} generated an exception: {exc}')

    with open(os.path.join(outpath, 'model_refusal_and_extraction_count.pkl'), 'wb') as f:
        pickle.dump(results, f)

    logger.info("Processing completed successfully.")

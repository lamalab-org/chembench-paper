from chembench.analysis import (
    load_all_reports,
    get_human_scored_questions_with_at_least_n_scores,
    all_correct,
    merge_with_topic_info,
)
from utils import obtain_chembench_repo
import os
from paths import data
import pickle
import pandas as pd


def combine_scores_for_model(
    folder, datafolder, human_baseline_folder=None, min_human_responses: int = 4
):

    df = load_all_reports(folder, datafolder)
    if human_baseline_folder is not None:
        relevant_questions = get_human_scored_questions_with_at_least_n_scores(
            human_baseline_folder, min_human_responses
        )
        df = df[df[("name", 0)].isin(relevant_questions)]
    df["all_correct"] = df.apply(all_correct, axis=1)

    return df


def load_human_aligned_reports(topic_frame):
    chembench = obtain_chembench_repo()
    human_baseline_folder = os.path.join(chembench, "reports/humans")
    datafolder = os.path.join(chembench, "data")

    claude2 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/claude2/reports/da6e43ef-5ab4-4f08-96b4-21509167a176"
        ),
        datafolder,
        human_baseline_folder,
    )

    claude3 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/claude3/reports/702e03be-5cd8-4451-b52c-8d7b9b694304"
        ),
        datafolder,
        human_baseline_folder,
    )
    
    claude35 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/claude3.5/reports/77835118-e4a4-415a-9356-ce8787a6f60b"
        ),
        datafolder,
        human_baseline_folder,
    )

    commandrplus = combine_scores_for_model(
        os.path.join(
            chembench, "reports/command-r+/reports/4ed6f819-39b4-4d22-8c1f-f3bcfca1fece"
        ),
        datafolder,
        human_baseline_folder,
    )

    galactica_120b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/galatica_120b/reports/6e8c6bc5-c0e5-4c6f-a89f-cffec13acbbe",
        ),
        datafolder,
        human_baseline_folder,
    )

    gemini_pro = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gemini-pro/reports/063b389f-e024-48d3-b292-927715dc0cdc"
        ),
        datafolder,
        human_baseline_folder,
    )

    gemma11_7b = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gemma-1-1-7b-it/reports/085a6e2e-56aa-46ca-8aa0-516bfde12912"
        ),
        datafolder,
        human_baseline_folder,
    )

    gemma11_7b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gemma-1-1-7b-it-T-one/reports/1b58e91b-0ffa-4af2-ac9c-c685b992a9af"
        ),
        datafolder,
        human_baseline_folder,
    )

    gemma2_9b = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gemma-2-9b-it/reports/c56f9729-3176-4177-9f18-76da33468d56"
        ),
        datafolder,
        human_baseline_folder,
    )

    gemma2_9b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gemma-2-9b-it-T-one/reports/600260b4-bfc9-449f-b3d5-03429ef71ed3"
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt35turbo = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-3.5-turbo/reports/2699315b-8d19-44e7-800c-fa8939cb3832",
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt4 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gpt-4/reports/efcbbadd-0448-4c8e-bfb9-0cf813b5a43a"
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt4o = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gpt-4o/reports/bd07947a-4422-4b2a-ab1b-5c113e95a572"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama2_70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama2-70b-chat/reports/87d2bdd8-7593-436a-bfc4-08c68babec38",
        ),
        datafolder,
        human_baseline_folder,
    )

    llama3_8b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-8b-instruct/reports/5f704f4c-b4af-4483-b305-c03d93345d52",
        ),
        datafolder,
        human_baseline_folder,
    )

    llama3_8b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-8b-instruct-T-one/reports/f1440201-3cdf-4998-9da7-8021c5c93969"
        ),
        datafolder,
        human_baseline_folder,
    )
    
    llama3_70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-70b-instruct/reports/39bbc48e-9b82-4760-bead-49123fa3bca3"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama3_70b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-70b-instruct-T-one/reports/061bad64-75f2-47fb-8b7c-caf60bd53e91"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama31_8b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-8b-instruct/reports/dc7a9240-699e-4ec8-a8c8-f910f8cc72f2"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama31_8b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-8b-instruct-T-one/reports/230e05b8-b18c-423e-a997-25e9ea70aba5"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama31_70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-70b-instruct/reports/69bd37fb-7acf-48c9-b025-3a24cdb0a276"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama31_70b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-70b-instruct-T-one/reports/263801d1-6c6d-4569-94ed-aef35dd8b7fc"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama31_405b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-405b-instruct/reports/00c9856f-4df7-44b1-ab0d-1872879fce47"
        ),
        datafolder,
        human_baseline_folder,
    )

    mistral_large = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mistral-large-2-123b/reports/25dcad40-8cbe-42ee-abe6-c19726c1c532"
        ),
        datafolder,
        human_baseline_folder,
    )

    mixtral = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mixtral-8x7b-instruct/reports/0f52e819-f0bf-4a7b-a36b-fb1c78dc5d53"
        ),
        datafolder,
        human_baseline_folder,
    )

    mixtral_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mixtral-8x7b-instruct-T-one/reports/54610d99-f51c-411f-bea8-ee57edcfff8b"
        ),
        datafolder,
        human_baseline_folder,
    )

    paperqa = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/paper-qa/reports/collected_reports"
        ),
        datafolder,
        human_baseline_folder,
    )

    phi = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/phi-3-medium-4k-instruct/reports/6ef70831-54d2-4a16-b41e-56437100dbe7"
        ),
        datafolder,
        human_baseline_folder,
    )

    random_baseline = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/random_baseline/reports/cdfc0dcf-32c0-456b-8c9d-8e1159a6262c_random"
        ),
        datafolder,
        human_baseline_folder,
    )

    claude2 = merge_with_topic_info(claude2, topic_frame)
    claude3 = merge_with_topic_info(claude3, topic_frame)
    claude35 = merge_with_topic_info(claude35, topic_frame)
    commandrplus = merge_with_topic_info(commandrplus, topic_frame)
    galactica_120b = merge_with_topic_info(galactica_120b, topic_frame)
    gemini_pro = merge_with_topic_info(gemini_pro, topic_frame)
    gemma11_7b = merge_with_topic_info(gemma11_7b, topic_frame)
    gemma11_7b_temp1 = merge_with_topic_info(gemma11_7b_temp1, topic_frame)
    gemma2_9b = merge_with_topic_info(gemma2_9b, topic_frame)
    gemma2_9b_temp1 = merge_with_topic_info(gemma2_9b_temp1, topic_frame)
    gpt35turbo = merge_with_topic_info(gpt35turbo, topic_frame)
    gpt4 = merge_with_topic_info(gpt4, topic_frame)
    gpt4o = merge_with_topic_info(gpt4o, topic_frame)
    llama2_70b = merge_with_topic_info(llama2_70b, topic_frame)
    llama3_8b = merge_with_topic_info(llama3_8b, topic_frame)
    llama3_8b_temp1 = merge_with_topic_info(llama3_8b_temp1, topic_frame)
    llama3_70b = merge_with_topic_info(llama3_70b, topic_frame)
    llama3_70b_temp1 = merge_with_topic_info(llama3_70b_temp1, topic_frame)
    llama31_8b = merge_with_topic_info(llama31_8b, topic_frame)
    llama31_8b_temp1 = merge_with_topic_info(llama31_8b_temp1, topic_frame)
    llama31_70b = merge_with_topic_info(llama31_70b, topic_frame)
    llama31_70b_temp1 = merge_with_topic_info(llama31_70b_temp1, topic_frame)
    llama31_405b = merge_with_topic_info(llama31_405b, topic_frame)
    mistral_large = merge_with_topic_info(mistral_large, topic_frame)
    mixtral = merge_with_topic_info(mixtral, topic_frame)
    mixtral_temp1 = merge_with_topic_info(mixtral_temp1, topic_frame)
    paperqa = merge_with_topic_info(paperqa, topic_frame)
    phi = merge_with_topic_info(phi, topic_frame)
    random_baseline = merge_with_topic_info(random_baseline, topic_frame)

    return {
        "claude2": claude2,
        "claude3": claude3,
        "claude35": claude35,
        "commandrplus": commandrplus,
        "galactica_120b": galactica_120b,
        "gemini_pro": gemini_pro,
        "gemma11_7b": gemma11_7b,
        "gemma11_7b_temp1": gemma11_7b_temp1,
        "gemma2_9b": gemma2_9b,
        "gemma2_9b_temp1": gemma2_9b_temp1,
        "gpt35turbo": gpt35turbo,
        "gpt4": gpt4,
        "gpt4o": gpt4o,
        "llama2_70b": llama2_70b,
        "llama3_8b": llama3_8b,
        "llama3_8b_temp1": llama3_8b_temp1,
        "llama3_70b": llama3_70b,
        "llama3_70b_temp1": llama3_70b_temp1,
        "llama31_8b": llama31_8b,
        "llama31_8b_temp1": llama31_8b_temp1,
        "llama31_70b": llama31_70b,
        "llama31_70b_temp1": llama31_70b_temp1,
        "llama31_405b": llama31_405b,
        "mistral_large": mistral_large,
        "mixtral": mixtral,
        "mixtral_temp1": mixtral_temp1,
        "paperqa": paperqa,
        "phi": phi,
        "random_baseline": random_baseline,
    }


def load_reports(topic_frame):
    chembench = obtain_chembench_repo()

    claude2 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/claude2/reports/da6e43ef-5ab4-4f08-96b4-21509167a176"
        ),
        os.path.join(chembench, "data"),
    )
    
    claude3 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/claude3/reports/702e03be-5cd8-4451-b52c-8d7b9b694304"
        ),
        os.path.join(chembench, "data"),
    )
    
    claude35 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/claude3.5/reports/77835118-e4a4-415a-9356-ce8787a6f60b"
        ),
        os.path.join(chembench, "data"),
    )
    
    commandrplus = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/command-r+/reports/4ed6f819-39b4-4d22-8c1f-f3bcfca1fece"
        ),
        os.path.join(chembench, "data"),
    )
    
    galactica_120b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/galatica_120b/reports/6e8c6bc5-c0e5-4c6f-a89f-cffec13acbbe"
        ),
        os.path.join(chembench, "data"),
    )
    
    gemini_pro = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gemini-pro/reports/063b389f-e024-48d3-b292-927715dc0cdc"
        ),
        os.path.join(chembench, "data"),
    )
    
    gemma11_7b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gemma-1-1-7b-it/reports/085a6e2e-56aa-46ca-8aa0-516bfde12912"
        ),
        os.path.join(chembench, "data"),
    )
    
    gemma11_7b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gemma-1-1-7b-it-T-one/reports/1b58e91b-0ffa-4af2-ac9c-c685b992a9af"
        ),
        os.path.join(chembench, "data"),
    )
    
    gemma2_9b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gemma-2-9b-it/reports/c56f9729-3176-4177-9f18-76da33468d56"
        ),
        os.path.join(chembench, "data"),
    )
    
    gemma2_9b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gemma-2-9b-it-T-one/reports/600260b4-bfc9-449f-b3d5-03429ef71ed3"
        ),
        os.path.join(chembench, "data"),
    )
    
    gpt35turbo = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-3.5-turbo/reports/2699315b-8d19-44e7-800c-fa8939cb3832"
        ),
        os.path.join(chembench, "data"),
    )
    
    gpt4 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-4/reports/efcbbadd-0448-4c8e-bfb9-0cf813b5a43a"
        ),
        os.path.join(chembench, "data"),
    )
    
    gpt4o = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-4o/reports/bd07947a-4422-4b2a-ab1b-5c113e95a572"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama2_70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama2-70b-chat/reports/87d2bdd8-7593-436a-bfc4-08c68babec38"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama3_8b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-8b-instruct/reports/5f704f4c-b4af-4483-b305-c03d93345d52"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama3_8b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-8b-instruct-T-one/reports/f1440201-3cdf-4998-9da7-8021c5c93969"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama3_70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-70b-instruct/reports/39bbc48e-9b82-4760-bead-49123fa3bca3"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama3_70b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3-70b-instruct-T-one/reports/061bad64-75f2-47fb-8b7c-caf60bd53e91"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama31_8b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-8b-instruct/reports/dc7a9240-699e-4ec8-a8c8-f910f8cc72f2"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama31_8b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-8b-instruct-T-one/reports/230e05b8-b18c-423e-a997-25e9ea70aba5"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama31_70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-70b-instruct/reports/69bd37fb-7acf-48c9-b025-3a24cdb0a276"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama31_70b_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-70b-instruct-T-one/reports/263801d1-6c6d-4569-94ed-aef35dd8b7fc"
        ),
        os.path.join(chembench, "data"),
    )
    
    llama31_405b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama3.1-405b-instruct/reports/00c9856f-4df7-44b1-ab0d-1872879fce47"
        ),
        os.path.join(chembench, "data"),
    )
    
    mistral_large = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mistral-large-2-123b/reports/25dcad40-8cbe-42ee-abe6-c19726c1c532"
        ),
        os.path.join(chembench, "data"),
    )
    
    mixtral = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mixtral-8x7b-instruct/reports/0f52e819-f0bf-4a7b-a36b-fb1c78dc5d53"
        ),
        os.path.join(chembench, "data"),
    )
    
    mixtral_temp1 = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mixtral-8x7b-instruct-T-one/reports/54610d99-f51c-411f-bea8-ee57edcfff8b"
        ),
        os.path.join(chembench, "data"),
    )
    
    paperqa = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/paper-qa/reports/collected_reports"
        ),
        os.path.join(chembench, "data"),
    )
    
    phi = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/phi-3-medium-4k-instruct/reports/6ef70831-54d2-4a16-b41e-56437100dbe7"
        ),
        os.path.join(chembench, "data"),
    )

    random_baseline = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/random_baseline/reports/cdfc0dcf-32c0-456b-8c9d-8e1159a6262c_random"
        ),
        os.path.join(chembench, "data"),
    )

    # to make fair comparisons, we only look at the runs
    # that were done for all models
    # for this we get the set of names for each model
    # and then take the intersection of these sets
    # the run names can be accessed under ("name", 0) in the dataframe

    claude2_run_names = set(claude2[("name", 0)])
    claude3_run_names = set(claude3[("name", 0)])
    claude35_run_names = set(claude35[("name", 0)])
    commandrplus_run_names = set(commandrplus[("name", 0)])
    galactica_120b_run_names = set(galactica_120b[("name", 0)])
    gemini_pro_run_names = set(gemini_pro[("name", 0)])
    gemma11_7b_run_names = set(gemma11_7b[("name", 0)])
    gemma11_7b_temp1_run_names = set(gemma11_7b_temp1[("name", 0)])
    gemma2_9b_run_names = set(gemma2_9b[("name", 0)])
    gemma2_9b_temp1_run_names = set(gemma2_9b_temp1[("name", 0)])
    gpt35turbo_run_names = set(gpt35turbo[("name", 0)])
    gpt4_run_names = set(gpt4[("name", 0)])
    gpt4o_run_names = set(gpt4o[("name", 0)])
    llama2_70b_run_names = set(llama2_70b[("name", 0)])
    llama3_8b_run_names = set(llama3_8b[("name", 0)])
    llama3_8b_temp1_run_names = set(llama3_8b_temp1[("name", 0)])
    llama3_70b_run_names = set(llama3_70b[("name", 0)])
    llama3_70b_temp1_run_names = set(llama3_70b_temp1[("name", 0)])
    llama31_8b_run_names = set(llama31_8b[("name", 0)])
    llama31_8b_temp1_run_names = set(llama31_8b_temp1[("name", 0)])
    llama31_70b_run_names = set(llama31_70b[("name", 0)])
    llama31_70b_temp1_run_names = set(llama31_70b_temp1[("name", 0)])
    llama31_405b_run_names = set(llama31_405b[("name", 0)])
    mistral_large_run_names = set(mistral_large[("name", 0)])
    mixtral_run_names = set(mixtral[("name", 0)])
    mixtral_temp1_run_names = set(mixtral_temp1[("name", 0)])
    paperqa_run_names = set(paperqa[("name", 0)])
    phi_run_names = set(phi[("name", 0)])
    random_baseline_run_names = set(random_baseline[("name", 0)])

    print(
        "RUN COUNTS",
        len(claude2_run_names),
        len(claude3_run_names),
        len(claude35_run_names),
        len(commandrplus_run_names),
        len(galactica_120b_run_names),
        len(gemini_pro_run_names),
        len(gemma11_7b_run_names),
        len(gemma11_7b_temp1_run_names),
        len(gemma2_9b_run_names),
        len(gemma2_9b_temp1_run_names),
        len(gpt35turbo_run_names),
        len(gpt4_run_names),
        len(gpt4o_run_names),
        len(llama2_70b_run_names),
        len(llama3_8b_run_names),
        len(llama3_8b_temp1_run_names),
        len(llama3_70b_run_names),
        len(llama3_70b_temp1_run_names),
        len(llama31_8b_run_names),
        len(llama31_8b_temp1_run_names),
        len(llama31_70b_run_names),
        len(llama31_70b_temp1_run_names),
        len(llama31_405b_run_names),
        len(mistral_large_run_names),
        len(mixtral_run_names),
        len(mixtral_temp1_run_names),
        len(paperqa_run_names),
        len(phi_run_names),
        len(random_baseline_run_names),
    )
    # now we take the intersection of these sets
    # intersection = (
    #     claude2_run_names
    #     & claude2_react_run_names
    #     & claude2_zero_t_run_names
    #     & claude3_run_names
    #     & galactica_120b_run_names
    #     & gemini_pro_zero_t_run_names
    #     & gemini_pro_run_names
    #     & gpt35turbo_run_names
    #     & gpt35turbo_zero_t_run_names
    #     & gpt35turbo_react_run_names
    #     & gpt4_run_names
    #     & llama70b_run_names
    #     & mixtral_run_names
    #     & pplx7b_chat_run_names
    #     & pplx7b_online_run_names
    #     & random_baseline_run_names
    # )

    # # now we filter the dataframes to only contain the runs
    # # that are in the intersection
    # claude2 = claude2[claude2[("name", 0)].isin(intersection)]
    # claude2_react = claude2_react[claude2_react[("name", 0)].isin(intersection)]
    # claude2_zero_t = claude2_zero_t[claude2_zero_t[("name", 0)].isin(intersection)]
    # claude3 = claude3[claude3[("name", 0)].isin(intersection)]
    # galactica_120b = galactica_120b[galactica_120b[("name", 0)].isin(intersection)]
    # gemini_pro_zero_t = gemini_pro_zero_t[
    #     gemini_pro_zero_t[("name", 0)].isin(intersection)
    # ]
    # gemini_pro = gemini_pro[gemini_pro[("name", 0)].isin(intersection)]
    # gpt35turbo = gpt35turbo[gpt35turbo[("name", 0)].isin(intersection)]
    # gpt35turbo_zero_t = gpt35turbo_zero_t[
    #     gpt35turbo_zero_t[("name", 0)].isin(intersection)
    # ]
    # gpt35turbo_react = gpt35turbo_react[
    #     gpt35turbo_react[("name", 0)].isin(intersection)
    # ]
    # gpt4 = gpt4[gpt4[("name", 0)].isin(intersection)]
    # llama70b = llama70b[llama70b[("name", 0)].isin(intersection)]
    # mixtral = mixtral[mixtral[("name", 0)].isin(intersection)]
    # pplx7b_chat = pplx7b_chat[pplx7b_chat[("name", 0)].isin(intersection)]
    # pplx7b_online = pplx7b_online[pplx7b_online[("name", 0)].isin(intersection)]
    # random_baseline = random_baseline[random_baseline[("name", 0)].isin(intersection)]

    claude2 = merge_with_topic_info(claude2, topic_frame)
    claude3 = merge_with_topic_info(claude3, topic_frame)
    claude35 = merge_with_topic_info(claude35, topic_frame)
    commandrplus = merge_with_topic_info(commandrplus, topic_frame)
    galactica_120b = merge_with_topic_info(galactica_120b, topic_frame)
    gemini_pro = merge_with_topic_info(gemini_pro, topic_frame)
    gemma11_7b = merge_with_topic_info(gemma11_7b, topic_frame)
    gemma11_7b_temp1 = merge_with_topic_info(gemma11_7b_temp1, topic_frame)
    gemma2_9b = merge_with_topic_info(gemma2_9b, topic_frame)
    gemma2_9b_temp1 = merge_with_topic_info(gemma2_9b_temp1, topic_frame)
    gpt35turbo = merge_with_topic_info(gpt35turbo, topic_frame)
    gpt4 = merge_with_topic_info(gpt4, topic_frame)
    gpt4o = merge_with_topic_info(gpt4o, topic_frame)
    llama2_70b = merge_with_topic_info(llama2_70b, topic_frame)
    llama3_8b = merge_with_topic_info(llama3_8b, topic_frame)
    llama3_8b_temp1 = merge_with_topic_info(llama3_8b_temp1, topic_frame)
    llama3_70b = merge_with_topic_info(llama3_70b, topic_frame)
    llama3_70b_temp1 = merge_with_topic_info(llama3_70b_temp1, topic_frame)
    llama31_8b = merge_with_topic_info(llama31_8b, topic_frame)
    llama31_8b_temp1 = merge_with_topic_info(llama31_8b_temp1, topic_frame)
    llama31_70b = merge_with_topic_info(llama31_70b, topic_frame)
    llama31_70b_temp1 = merge_with_topic_info(llama31_70b_temp1, topic_frame)
    llama31_405b = merge_with_topic_info(llama31_405b, topic_frame)
    mistral_large = merge_with_topic_info(mistral_large, topic_frame)
    mixtral = merge_with_topic_info(mixtral, topic_frame)
    mixtral_temp1 = merge_with_topic_info(mixtral_temp1, topic_frame)
    paperqa = merge_with_topic_info(paperqa, topic_frame)
    phi = merge_with_topic_info(phi, topic_frame)
    random_baseline = merge_with_topic_info(random_baseline, topic_frame)

    return {
        "claude2": claude2,
        "claude3": claude3,
        "claude35": claude35,
        "commandrplus": commandrplus,
        "galactica_120b": galactica_120b,
        "gemini_pro": gemini_pro,
        "gemma11_7b": gemma11_7b,
        "gemma11_7b_temp1": gemma11_7b_temp1,
        "gemma2_9b": gemma2_9b,
        "gemma2_9b_temp1": gemma2_9b_temp1,
        "gpt35turbo": gpt35turbo,
        "gpt4": gpt4,
        "gpt4o": gpt4o,
        "llama2_70b": llama2_70b,
        "llama3_8b": llama3_8b,
        "llama3_8b_temp1": llama3_8b_temp1,
        "llama3_70b": llama3_70b,
        "llama3_70b_temp1": llama3_70b_temp1,
        "llama31_8b": llama31_8b,
        "llama31_8b_temp1": llama31_8b_temp1,
        "llama31_70b": llama31_70b,
        "llama31_70b_temp1": llama31_70b_temp1,
        "llama31_405b": llama31_405b,
        "mistral_large": mistral_large,
        "mixtral": mixtral,
        "mixtral_temp1": mixtral_temp1,
        "paperqa": paperqa,
        "phi": phi,
        "random_baseline": random_baseline,
    }

if __name__ == "__main__":
    topic_frame = pd.read_pickle(data / "questions.pkl")
    df_dict = load_reports(topic_frame)
    df_dict_human_aligned = load_human_aligned_reports(topic_frame)
    print("\n\n\n")
    columns_list = list(df_dict["claude2"].columns)
    print(columns_list)
    print("\n\n\n")

    results = {"overall": df_dict, "human_aligned": df_dict_human_aligned}

    with open(data / "model_score_dicts.pkl", "wb") as handle:
        pickle.dump(results, handle)

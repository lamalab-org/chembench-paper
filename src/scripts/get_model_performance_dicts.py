from chembench.analysis import (
    load_all_reports,
    get_human_scored_questions_with_at_least_n_scores,
    all_correct,
    merge_with_topic_info,
)
from utils import obtain_chembench_repo
import os
import matplotlib.pyplot as plt
from paths import output, scripts, data
import pickle
import pandas as pd

plt.style.use(scripts / "lamalab.mplstyle")


def combine_scores_for_model(
    folder, datafolder, human_baseline_folder=None, min_human_responses: int = 100
):

    df = load_all_reports(folder, datafolder)
    if human_baseline_folder is not None:
        relevant_questions = get_human_scored_questions_with_at_least_n_scores(
            human_baseline_folder, min_human_responses
        )
        df = df[df[("name", 0)].isin(relevant_questions)]
    df["all_correct"] = df.apply(all_correct, axis=1)

    return df


def load_human_aligned_reports(topics):
    chembench = obtain_chembench_repo()
    human_baseline_folder = os.path.join(chembench, "reports/humans")
    datafolder = os.path.join(chembench, "data")

    claude2 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/claude2/reports/7536581a-d92d-43d3-b946-39e4a7213b7f"
        ),
        datafolder,
        human_baseline_folder,
    )

    claude2_react = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/claude2-react/reports/700ed9b4-995c-4cfd-aa63-0a6e84b3a815",
        ),
        datafolder,
        human_baseline_folder,
    )

    claude2_zero_t = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/claude2-zero-T/reports/412657cc-7a11-4d73-80e9-03d6f05cd63e",
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

    galactica_120b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/galatica_120b/reports/d7ce25da-bbce-4f06-8a5b-43e6cfb01c30",
        ),
        datafolder,
        human_baseline_folder,
    )

    gemini_pro_zero_t = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gemin-pro-zero-T/reports/1e5457ad-96b5-4bc8-bd6c-bad3eb6deb7a",
        ),
        datafolder,
        human_baseline_folder,
    )

    gemini_pro = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gemini-pro/reports/ebde051c-6d66-456a-a207-a1c65eceaf40"
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt35turbo = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-3.5-turbo/reports/f76bf17d-3e12-47c5-b879-9cc0c78be989",
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt35turbo_zero_t = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-3.5-turbo-zero-T/reports/44cf2a6b-a7bc-43ee-8f16-9576d1547c76",
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt35turbo_react = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/gpt-35-turbo-react/reports/e4964803-79cb-44bc-b5b2-e22aa3f40607",
        ),
        datafolder,
        human_baseline_folder,
    )

    gpt4 = combine_scores_for_model(
        os.path.join(
            chembench, "reports/gpt-4/reports/76c5bdd4-e893-43d4-b37d-2ade66c20308"
        ),
        datafolder,
        human_baseline_folder,
    )

    llama70b = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/llama-2-70b-chat/reports/a2d0b2d7-2381-4e75-8c8b-8baadf054073",
        ),
        datafolder,
        human_baseline_folder,
    )

    mixtral = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/mixtral-8x7b-instruct/reports/8409294f-de26-4c05-bd50-2cfb2148ec65",
        ),
        datafolder,
        human_baseline_folder,
    )

    pplx7b_chat = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/pplx-7b-chat/reports/d63fb4e2-3dd6-432e-bbe7-8bc1c23115d3",
        ),
        datafolder,
        human_baseline_folder,
    )

    pplx7b_online = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/pplx-7b-online/reports/309c0ecb-bd79-406d-bd2d-b5d434053f2f",
        ),
        datafolder,
        human_baseline_folder,
    )

    random_baseline = combine_scores_for_model(
        os.path.join(
            chembench,
            "reports/random_baseline/reports/56255cd7-294b-4441-b389-e4a1eb044107_random",
        ),
        datafolder,
        human_baseline_folder,
    )

    claude2 = merge_with_topic_info(claude2, topic_frame)
    claude2_react = merge_with_topic_info(claude2_react, topic_frame)
    claude2_zero_t = merge_with_topic_info(claude2_zero_t, topic_frame)
    claude3 = merge_with_topic_info(claude3, topic_frame)
    galactica_120b = merge_with_topic_info(galactica_120b, topic_frame)
    gemini_pro_zero_t = merge_with_topic_info(gemini_pro_zero_t, topic_frame)
    gemini_pro = merge_with_topic_info(gemini_pro, topic_frame)
    gpt35turbo = merge_with_topic_info(gpt35turbo, topic_frame)
    gpt35turbo_zero_t = merge_with_topic_info(gpt35turbo_zero_t, topic_frame)
    gpt4 = merge_with_topic_info(gpt4, topic_frame)
    llama70b = merge_with_topic_info(llama70b, topic_frame)
    mixtral = merge_with_topic_info(mixtral, topic_frame)
    pplx7b_chat = merge_with_topic_info(pplx7b_chat, topic_frame)
    pplx7b_online = merge_with_topic_info(pplx7b_online, topic_frame)
    random_baseline = merge_with_topic_info(random_baseline, topic_frame)

    return {
        "claude2": claude2,
        "claude2_react": claude2_react,
        "claude2_zero_t": claude2_zero_t,
        "claude3": claude3,
        "galactica_120b": galactica_120b,
        "gemini_pro_zero_t": gemini_pro_zero_t,
        "gemini_pro": gemini_pro,
        "gpt35turbo": gpt35turbo,
        "gpt35turbo_zero_t": gpt35turbo_zero_t,
        "gpt35turbo_react": gpt35turbo_react,
        "gpt4": gpt4,
        "llama70b": llama70b,
        "mixtral": mixtral,
        "pplx7b_chat": pplx7b_chat,
        "pplx7b_online": pplx7b_online,
        "random_baseline": random_baseline,
    }


def load_reports():
    chembench = obtain_chembench_repo()
    claude2 = load_all_reports(
        os.path.join(
            chembench, "reports/claude2/reports/7536581a-d92d-43d3-b946-39e4a7213b7f"
        ),
        os.path.join(chembench, "data"),
    )

    claude2_react = load_all_reports(
        os.path.join(
            chembench,
            "reports/claude2-react/reports/700ed9b4-995c-4cfd-aa63-0a6e84b3a815",
        ),
        os.path.join(chembench, "data"),
    )

    claude2_zero_t = load_all_reports(
        os.path.join(
            chembench,
            "reports/claude2-zero-T/reports/412657cc-7a11-4d73-80e9-03d6f05cd63e",
        ),
        os.path.join(chembench, "data"),
    )

    claude3 = load_all_reports(
        os.path.join(
            chembench, "reports/claude3/reports/702e03be-5cd8-4451-b52c-8d7b9b694304"
        ),
        os.path.join(chembench, "data"),
    )

    galactica_120b = load_all_reports(
        os.path.join(
            chembench,
            "reports/galatica_120b/reports/d7ce25da-bbce-4f06-8a5b-43e6cfb01c30",
        ),
        os.path.join(chembench, "data"),
    )

    gemini_pro_zero_t = load_all_reports(
        os.path.join(
            chembench,
            "reports/gemin-pro-zero-T/reports/1e5457ad-96b5-4bc8-bd6c-bad3eb6deb7a",
        ),
        os.path.join(chembench, "data"),
    )

    gemini_pro = load_all_reports(
        os.path.join(
            chembench, "reports/gemini-pro/reports/ebde051c-6d66-456a-a207-a1c65eceaf40"
        ),
        os.path.join(chembench, "data"),
    )

    gpt35turbo = load_all_reports(
        os.path.join(
            chembench,
            "reports/gpt-3.5-turbo/reports/f76bf17d-3e12-47c5-b879-9cc0c78be989",
        ),
        os.path.join(chembench, "data"),
    )

    gpt35turbo_zero_t = load_all_reports(
        os.path.join(
            chembench,
            "reports/gpt-3.5-turbo-zero-T/reports/44cf2a6b-a7bc-43ee-8f16-9576d1547c76",
        ),
        os.path.join(chembench, "data"),
    )

    gpt35turbo_react = load_all_reports(
        os.path.join(
            chembench,
            "reports/gpt-35-turbo-react/reports/e4964803-79cb-44bc-b5b2-e22aa3f40607",
        ),
        os.path.join(chembench, "data"),
    )

    gpt4 = load_all_reports(
        os.path.join(
            chembench, "reports/gpt-4/reports/76c5bdd4-e893-43d4-b37d-2ade66c20308"
        ),
        os.path.join(chembench, "data"),
    )

    llama70b = load_all_reports(
        os.path.join(
            chembench,
            "reports/llama-2-70b-chat/reports/a2d0b2d7-2381-4e75-8c8b-8baadf054073",
        ),
        os.path.join(chembench, "data"),
    )

    mixtral = load_all_reports(
        os.path.join(
            chembench,
            "reports/mixtral-8x7b-instruct/reports/8409294f-de26-4c05-bd50-2cfb2148ec65",
        ),
        os.path.join(chembench, "data"),
    )

    pplx7b_chat = load_all_reports(
        os.path.join(
            chembench,
            "reports/pplx-7b-chat/reports/d63fb4e2-3dd6-432e-bbe7-8bc1c23115d3",
        ),
        os.path.join(chembench, "data"),
    )

    pplx7b_online = load_all_reports(
        os.path.join(
            chembench,
            "reports/pplx-7b-online/reports/309c0ecb-bd79-406d-bd2d-b5d434053f2f",
        ),
        os.path.join(chembench, "data"),
    )

    random_baseline = load_all_reports(
        os.path.join(
            chembench,
            "reports/random_baseline/reports/56255cd7-294b-4441-b389-e4a1eb044107_random",
        ),
        os.path.join(chembench, "data"),
    )

    # to make fair comparisons, we only look at the runs
    # that were done for all models
    # for this we get the set of names for each model
    # and then take the intersection of these sets
    # the run names can be accessed under ("name", 0) in the dataframe

    claude2_run_names = set(claude2[("name", 0)])
    claude2_react_run_names = set(claude2_react[("name", 0)])
    claude2_zero_t_run_names = set(claude2_zero_t[("name", 0)])
    claude3_run_names = set(claude3[("name", 0)])
    galactica_120b_run_names = set(galactica_120b[("name", 0)])
    gemini_pro_zero_t_run_names = set(gemini_pro_zero_t[("name", 0)])
    gemini_pro_run_names = set(gemini_pro[("name", 0)])
    gpt35turbo_run_names = set(gpt35turbo[("name", 0)])
    gpt35turbo_zero_t_run_names = set(gpt35turbo_zero_t[("name", 0)])
    gpt35turbo_react_run_names = set(gpt35turbo_react[("name", 0)])
    gpt4_run_names = set(gpt4[("name", 0)])
    llama70b_run_names = set(llama70b[("name", 0)])
    mixtral_run_names = set(mixtral[("name", 0)])
    pplx7b_chat_run_names = set(pplx7b_chat[("name", 0)])
    pplx7b_online_run_names = set(pplx7b_online[("name", 0)])
    random_baseline_run_names = set(random_baseline[("name", 0)])

    # now we take the intersection of these sets
    intersection = (
        claude2_run_names
        & claude2_react_run_names
        & claude2_zero_t_run_names
        & claude3_run_names
        & galactica_120b_run_names
        & gemini_pro_zero_t_run_names
        & gemini_pro_run_names
        & gpt35turbo_run_names
        & gpt35turbo_zero_t_run_names
        & gpt35turbo_react_run_names
        & gpt4_run_names
        & llama70b_run_names
        & mixtral_run_names
        & pplx7b_chat_run_names
        & pplx7b_online_run_names
        & random_baseline_run_names
    )

    with open(output / "intersection.txt", "w") as f:
        f.write(f"{len(intersection)}" + "\endinput")

    # now we filter the dataframes to only contain the runs
    # that are in the intersection
    claude2 = claude2[claude2[("name", 0)].isin(intersection)]
    claude2_react = claude2_react[claude2_react[("name", 0)].isin(intersection)]
    claude2_zero_t = claude2_zero_t[claude2_zero_t[("name", 0)].isin(intersection)]
    claude3 = claude3[claude3[("name", 0)].isin(intersection)]
    galactica_120b = galactica_120b[galactica_120b[("name", 0)].isin(intersection)]
    gemini_pro_zero_t = gemini_pro_zero_t[
        gemini_pro_zero_t[("name", 0)].isin(intersection)
    ]
    gemini_pro = gemini_pro[gemini_pro[("name", 0)].isin(intersection)]
    gpt35turbo = gpt35turbo[gpt35turbo[("name", 0)].isin(intersection)]
    gpt35turbo_zero_t = gpt35turbo_zero_t[
        gpt35turbo_zero_t[("name", 0)].isin(intersection)
    ]
    gpt35turbo_react = gpt35turbo_react[
        gpt35turbo_react[("name", 0)].isin(intersection)
    ]
    gpt4 = gpt4[gpt4[("name", 0)].isin(intersection)]
    llama70b = llama70b[llama70b[("name", 0)].isin(intersection)]
    mixtral = mixtral[mixtral[("name", 0)].isin(intersection)]
    pplx7b_chat = pplx7b_chat[pplx7b_chat[("name", 0)].isin(intersection)]
    pplx7b_online = pplx7b_online[pplx7b_online[("name", 0)].isin(intersection)]
    random_baseline = random_baseline[random_baseline[("name", 0)].isin(intersection)]

    claude2 = merge_with_topic_info(claude2, topic_frame)
    claude2_react = merge_with_topic_info(claude2_react, topic_frame)
    claude2_zero_t = merge_with_topic_info(claude2_zero_t, topic_frame)
    claude3 = merge_with_topic_info(claude3, topic_frame)
    galactica_120b = merge_with_topic_info(galactica_120b, topic_frame)
    gemini_pro_zero_t = merge_with_topic_info(gemini_pro_zero_t, topic_frame)
    gemini_pro = merge_with_topic_info(gemini_pro, topic_frame)
    gpt35turbo = merge_with_topic_info(gpt35turbo, topic_frame)
    gpt35turbo_zero_t = merge_with_topic_info(gpt35turbo_zero_t, topic_frame)
    gpt4 = merge_with_topic_info(gpt4, topic_frame)
    llama70b = merge_with_topic_info(llama70b, topic_frame)
    mixtral = merge_with_topic_info(mixtral, topic_frame)
    pplx7b_chat = merge_with_topic_info(pplx7b_chat, topic_frame)
    pplx7b_online = merge_with_topic_info(pplx7b_online, topic_frame)
    random_baseline = merge_with_topic_info(random_baseline, topic_frame)

    # return dictionary with the dataframes
    return {
        "claude2": claude2,
        "claude2_react": claude2_react,
        "claude2_zero_t": claude2_zero_t,
        "claude3": claude3,
        "galactica_120b": galactica_120b,
        "gemini_pro_zero_t": gemini_pro_zero_t,
        "gemini_pro": gemini_pro,
        "gpt35turbo": gpt35turbo,
        "gpt35turbo_zero_t": gpt35turbo_zero_t,
        "gpt35turbo_react": gpt35turbo_react,
        "gpt4": gpt4,
        "llama70b": llama70b,
        "mixtral": mixtral,
        "pplx7b_chat": pplx7b_chat,
        "pplx7b_online": pplx7b_online,
        "random_baseline": random_baseline,
    }


if __name__ == "__main__":
    topic_frame = pd.read_pickle(data / "questions.pkl")
    df_dict = load_reports()
    df_dict_human_aligned = load_human_aligned_reports()

    results = {"overall": df_dict, "human_aligned": df_dict_human_aligned}

    with open(data / "model_score_dicts.pkl", "wb") as handle:
        pickle.dump(results, handle)

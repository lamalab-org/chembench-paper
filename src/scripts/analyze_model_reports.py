import matplotlib.pyplot as plt
from plotutils import radar_factory, model_color_map
import numpy as np
import matplotlib.pyplot as plt
from paths import figures, data, scripts
import pickle

plt.style.use(scripts / "lamalab.mplstyle")


human_relevant_topics = [
    "General Chemistry",
    "Technical Chemistry",
    "Organic Chemistry",
    "Materials Science",
    "Physical Chemistry",
    "Analytical Chemistry",
    "Inorganic Chemistry",
    "Toxicity/Safety",
]


def make_overall_performance_radar_plot(df_dict, suffix, human_dicts=None):
    claude2_mean = df_dict["claude2"].groupby("topic")["all_correct_"].mean()
    claude3_mean = df_dict["claude3"].groupby("topic")["all_correct_"].mean()
    claude35_mean = df_dict["claude35"].groupby("topic")["all_correct_"].mean()
    commandrplus_mean = df_dict["commandrplus"].groupby("topic")["all_correct_"].mean()
    galactica_120b_mean = df_dict["galactica_120b"].groupby("topic")["all_correct_"].mean()
    gemini_pro_mean = df_dict["gemini_pro"].groupby("topic")["all_correct_"].mean()
    gemma11_7b_mean = df_dict["gemma11_7b"].groupby("topic")["all_correct_"].mean()
    gemma11_7b_temp1_mean = df_dict["gemma11_7b_temp1"].groupby("topic")["all_correct_"].mean()
    gemma2_9b_mean = df_dict["gemma2_9b"].groupby("topic")["all_correct_"].mean()
    gemma2_9b_temp1_mean = df_dict["gemma2_9b_temp1"].groupby("topic")["all_correct_"].mean()
    gpt35turbo_mean = df_dict["gpt35turbo"].groupby("topic")["all_correct_"].mean()
    gpt4_mean = df_dict["gpt4"].groupby("topic")["all_correct_"].mean()
    gpt4o_mean = df_dict["gpt4o"].groupby("topic")["all_correct_"].mean()
    llama2_70b_mean = df_dict["llama2_70b"].groupby("topic")["all_correct_"].mean()
    llama3_8b_mean = df_dict["llama3_8b"].groupby("topic")["all_correct_"].mean()
    llama3_8b_temp1_mean = df_dict["llama3_8b_temp1"].groupby("topic")["all_correct_"].mean()
    llama3_70b_mean = df_dict["llama3_70b"].groupby("topic")["all_correct_"].mean()
    llama3_70b_temp1_mean = df_dict["llama3_70b_temp1"].groupby("topic")["all_correct_"].mean()
    llama31_8b_mean = df_dict["llama31_8b"].groupby("topic")["all_correct_"].mean()
    llama31_8b_temp1_mean = df_dict["llama31_8b_temp1"].groupby("topic")["all_correct_"].mean()
    llama31_70b_mean = df_dict["llama31_70b"].groupby("topic")["all_correct_"].mean()
    llama31_70b_temp1_mean = df_dict["llama31_70b_temp1"].groupby("topic")["all_correct_"].mean()
    llama31_405b_mean = df_dict["llama31_405b"].groupby("topic")["all_correct_"].mean()
    mistral_large_mean = df_dict["mistral_large"].groupby("topic")["all_correct_"].mean()
    mixtral_mean = df_dict["mixtral"].groupby("topic")["all_correct_"].mean()
    mixtral_temp1_mean = df_dict["mixtral_temp1"].groupby("topic")["all_correct_"].mean()
    paperqa_mean = df_dict["paperqa"].groupby("topic")["all_correct_"].mean()
    phi_mean = df_dict["phi"].groupby("topic")["all_correct_"].mean()
    random_baseline_mean = df_dict["random_baseline"].groupby("topic")["all_correct_"].mean()

    if human_dicts is None:
        # Sort the data based on the maximum value in each group
        sorted_data = sorted(
            list(
                zip(
                    [
                        claude2_mean,
                        claude3_mean,
                        claude35_mean,
                        commandrplus_mean,
                        galactica_120b_mean,
                        gemini_pro_mean,
                        gemma11_7b_mean,
                        gemma11_7b_temp1_mean,
                        gemma2_9b_mean,
                        gemma2_9b_temp1_mean,
                        gpt35turbo_mean,
                        gpt4_mean,
                        gpt4o_mean,
                        llama2_70b_mean,
                        llama3_8b_mean,
                        llama3_8b_temp1_mean,
                        llama3_70b_mean,
                        llama3_70b_temp1_mean,
                        llama31_8b_mean,
                        llama31_8b_temp1_mean,
                        llama31_70b_mean,
                        llama31_70b_temp1_mean,
                        llama31_405b_mean,
                        mistral_large_mean,
                        mixtral_mean,
                        mixtral_temp1_mean,
                        paperqa_mean,
                        phi_mean,
                        random_baseline_mean,
                    ],
                    [
                        "Claude2",
                        "Claude3",
                        "Claude35",
                        "Command-R+",
                        "Galactica-120B",
                        "Gemini-Pro",
                        "Gemma-7B",
                        "Gemma-7B_Temp1",
                        "Gemma2-9B",
                        "Gemma2-9B_Temp1",
                        "GPT-3.5-Turbo",
                        "GPT-4",
                        "GPT-4o",
                        "Llama2-70B",
                        "Llama3-8B",
                        "Llama3-8B_Temp1",
                        "Llama3-70B",
                        "Llama3-70B_Temp1",
                        "Llama3.1-8B",
                        "Llama3.1-8B_Temp1",
                        "Llama3.1-70B",
                        "Llama3.1-70B_Temp1",
                        "Llama3.1-405B",
                        "Mistral-Large2-123B",
                        "Mixtral-8x7B",
                        "Mixtral-8x7B_Temp1",
                        "PaperQA",
                        "Phi3-Medium",
                        "Random Baseline",
                    ],
                    [
                        model_color_map["Claude2"],
                        model_color_map["Claude3"],
                        model_color_map["Claude35"],
                        model_color_map["Command-R+"],
                        model_color_map["Galactica-120B"],
                        model_color_map["Gemini-Pro"],
                        model_color_map["Gemma-7B"],
                        model_color_map["Gemma-7B_Temp1"],
                        model_color_map["Gemma2-9B"],
                        model_color_map["Gemma2-9B_Temp1"],
                        model_color_map["GPT-3.5-Turbo"],
                        model_color_map["GPT-4"],
                        model_color_map["GPT-4o"],
                        model_color_map["Llama2-70B"],
                        model_color_map["Llama3-8B"],
                        model_color_map["Llama3-8B_Temp1"],
                        model_color_map["Llama3-70B"],
                        model_color_map["Llama3-70B_Temp1"],
                        model_color_map["Llama3.1-8B"],
                        model_color_map["Llama3.1-8B_Temp1"],
                        model_color_map["Llama3.1-70B"],
                        model_color_map["Llama3.1-70B_Temp1"],
                        model_color_map["Llama3.1-405B"],
                        model_color_map["Mistral-Large2-123B"],
                        model_color_map["Mixtral-8x7B"],
                        model_color_map["Mixtral-8x7B_Temp1"],
                        model_color_map["PaperQA"],
                        model_color_map["Phi3-Medium"],
                        model_color_map["Random_Baseline"],
                    ],
                )
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
    else:
        # filter out topics that are not relevant for humans
        human_dicts = human_dicts.loc[human_relevant_topics]
        claude2_mean = claude2_mean.loc[human_relevant_topics]
        claude3_mean = claude3_mean.loc[human_relevant_topics]
        claude35_mean = claude35_mean.loc[human_relevant_topics]
        commandrplus_mean = commandrplus_mean.loc[human_relevant_topics]
        galactica_120b_mean = galactica_120b_mean.loc[human_relevant_topics]
        gemini_pro_mean = gemini_pro_mean.loc[human_relevant_topics]
        gemma11_7b_mean = gemma11_7b_mean.loc[human_relevant_topics]
        gemma11_7b_temp1_mean = gemma11_7b_temp1_mean.loc[human_relevant_topics]
        gemma2_9b_mean = gemma2_9b_mean.loc[human_relevant_topics]
        gemma2_9b_temp1_mean = gemma2_9b_temp1_mean.loc[human_relevant_topics]
        gpt35turbo_mean = gpt35turbo_mean.loc[human_relevant_topics]
        gpt4_mean = gpt4_mean.loc[human_relevant_topics]
        gpt4o_mean = gpt4o_mean.loc[human_relevant_topics]
        llama2_70b_mean = llama2_70b_mean.loc[human_relevant_topics]
        llama3_8b_mean = llama3_8b_mean.loc[human_relevant_topics]
        llama3_8b_temp1_mean = llama3_8b_temp1_mean.loc[human_relevant_topics]
        llama3_70b_mean = llama3_70b_mean.loc[human_relevant_topics]
        llama3_70b_temp1_mean = llama3_70b_temp1_mean.loc[human_relevant_topics]
        llama31_8b_mean = llama31_8b_mean.loc[human_relevant_topics]
        llama31_8b_temp1_mean = llama31_8b_temp1_mean.loc[human_relevant_topics]
        llama31_70b_mean = llama31_70b_mean.loc[human_relevant_topics]
        llama31_70b_temp1_mean = llama31_70b_temp1_mean.loc[human_relevant_topics]
        llama31_405b_mean = llama31_405b_mean.loc[human_relevant_topics]
        mistral_large_mean = mistral_large_mean.loc[human_relevant_topics]
        mixtral_mean = mixtral_mean.loc[human_relevant_topics]
        mixtral_temp1_mean = mixtral_temp1_mean.loc[human_relevant_topics]
        paperqa_mean = paperqa_mean.loc[human_relevant_topics]
        phi_mean = phi_mean.loc[human_relevant_topics]
        random_baseline_mean = random_baseline_mean.loc[human_relevant_topics]

        sorted_data = sorted(
            list(
                zip(
                    [
                        claude2_mean,
                        claude3_mean,
                        claude35_mean,
                        commandrplus_mean,
                        galactica_120b_mean,
                        gemini_pro_mean,
                        gemma11_7b_mean,
                        gemma11_7b_temp1_mean,
                        gemma2_9b_mean,
                        gemma2_9b_temp1_mean,
                        gpt35turbo_mean,
                        gpt4_mean,
                        gpt4o_mean,
                        llama2_70b_mean,
                        llama3_8b_mean,
                        llama3_8b_temp1_mean,
                        llama3_70b_mean,
                        llama3_70b_temp1_mean,
                        llama31_8b_mean,
                        llama31_8b_temp1_mean,
                        llama31_70b_mean,
                        llama31_70b_temp1_mean,
                        llama31_405b_mean,
                        mistral_large_mean,
                        mixtral_mean,
                        mixtral_temp1_mean,
                        paperqa_mean,
                        phi_mean,
                        random_baseline_mean,
                    ],
                    [
                        "Claude2",
                        "Claude3",
                        "Claude3.5",
                        "Command-R+",
                        "Galactica-120B",
                        "Gemini-Pro",
                        "Gemma-7B",
                        "Gemma-7B_Temp1",
                        "Gemma2-9B",
                        "Gemma2-9B_Temp1",
                        "GPT-3.5-Turbo",
                        "GPT-4",
                        "GPT-4o",
                        "Llama2-70B",
                        "Llama3-8B",
                        "Llama3-8B_Temp1",
                        "Llama3-70B",
                        "Llama3-70B_Temp1",
                        "Llama3.1-8B",
                        "Llama3.1-8B_Temp1",
                        "Llama3.1-70B",
                        "Llama3.1-70B_Temp1",
                        "Llama3.1-405B",
                        "Mistral-Large2-123B",
                        "Mixtral-8x7B",
                        "Mixtral-8x7B_Temp1",
                        "PaperQA",
                        "Phi3-Medium",
                        "Random Baseline",
                    ],
                    [
                        model_color_map["Claude2"],
                        model_color_map["Claude3"],
                        model_color_map["Claude35"],
                        model_color_map["Command-R+"],
                        model_color_map["Galactica-120B"],
                        model_color_map["Gemini-Pro"],
                        model_color_map["Gemma-7B"],
                        model_color_map["Gemma-7B_Temp1"],
                        model_color_map["Gemma2-9B"],
                        model_color_map["Gemma2-9B_Temp1"],
                        model_color_map["GPT-3.5-Turbo"],
                        model_color_map["GPT-4"],
                        model_color_map["GPT-4o"],
                        model_color_map["Llama2-70B"],
                        model_color_map["Llama3-8B"],
                        model_color_map["Llama3-8B_Temp1"],
                        model_color_map["Llama3-70B"],
                        model_color_map["Llama3-70B_Temp1"],
                        model_color_map["Llama3.1-8B"],
                        model_color_map["Llama3.1-8B_Temp1"],
                        model_color_map["Llama3.1-70B"],
                        model_color_map["Llama3.1-70B_Temp1"],
                        model_color_map["Llama3.1-405B"],
                        model_color_map["Mistral-Large2-123B"],
                        model_color_map["Mixtral-8x7B"],
                        model_color_map["Mixtral-8x7B_Temp1"],
                        model_color_map["PaperQA"],
                        model_color_map["Phi3-Medium"],
                        model_color_map["Random_Baseline"],
                    ],
                )
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
    theta = radar_factory(len(claude2_mean), frame="polygon")

    # Adjust the layout to leave space for labels
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection="radar"))
    for data_, label, color in sorted_data:
        # Plot the filled area
        print(data_, label, color)
        ax.fill(theta, data_, alpha=0.2, label=label, color=color)

        # Plot the line
        ax.plot(theta, data_, color=color)

    # Rotate labels
    ax.set_varlabels(claude2_mean.index)
    ax.tick_params(pad=30)
    plt.xticks(
        rotation=45, ha="center"
    )  # Adjust horizontal alignment for better readability

    # Remove unnecessary grid lines
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Improve legend placement
    leg = ax.legend(loc=(-0.1, 1.2), ncols=3)

    for lh in leg.legend_handles:
        lh.set_alpha(1)

    fig.tight_layout()

    fig.savefig(
        figures / f"all_questions_models_completely_correct_radar_{suffix}.pdf",
        bbox_inches="tight",
    )


def radarplot_requires_calculation(df_dict, human_dicts, suffix):
    # to get subset that requires calculation, we can filter
    # model_df.groupby('requires_calculation')['all_correct_'].mean()[1]
    # for each model

    df_dict["claude2"]["requires_calculation"] = df_dict["claude2"]["keywords"].str.contains("requires-calculation")
    claude2_with_calc = df_dict["claude2"][df_dict["claude2"]["requires_calculation"]]
    claude2_mean = claude2_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["claude3"]["requires_calculation"] = df_dict["claude3"]["keywords"].str.contains("requires-calculation")
    claude3_with_calc = df_dict["claude3"][df_dict["claude3"]["requires_calculation"]]
    claude3_mean = claude3_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["claude35"]["requires_calculation"] = df_dict["claude35"]["keywords"].str.contains("requires-calculation")
    claude35_with_calc = df_dict["claude35"][df_dict["claude35"]["requires_calculation"]]
    claude35_mean = claude35_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["commandrplus"]["requires_calculation"] = df_dict["commandrplus"]["keywords"].str.contains("requires-calculation")
    commandrplus_with_calc = df_dict["commandrplus"][df_dict["commandrplus"]["requires_calculation"]]
    commandrplus_mean = commandrplus_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["galactica_120b"]["requires_calculation"] = df_dict["galactica_120b"]["keywords"].str.contains("requires-calculation")
    galactica_120b_with_calc = df_dict["galactica_120b"][df_dict["galactica_120b"]["requires_calculation"]]
    galactica_120b_mean = galactica_120b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gemini_pro"]["requires_calculation"] = df_dict["gemini_pro"]["keywords"].str.contains("requires-calculation")
    gemini_pro_with_calc = df_dict["gemini_pro"][df_dict["gemini_pro"]["requires_calculation"]]
    gemini_pro_mean = gemini_pro_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gemma11_7b"]["requires_calculation"] = df_dict["gemma11_7b"]["keywords"].str.contains("requires-calculation")
    gemma11_7b_with_calc = df_dict["gemma11_7b"][df_dict["gemma11_7b"]["requires_calculation"]]
    gemma11_7b_mean = gemma11_7b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gemma11_7b_temp1"]["requires_calculation"] = df_dict["gemma11_7b_temp1"]["keywords"].str.contains("requires-calculation")
    gemma11_7b_temp1_with_calc = df_dict["gemma11_7b_temp1"][df_dict["gemma11_7b_temp1"]["requires_calculation"]]
    gemma11_7b_temp1_mean = gemma11_7b_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gemma2_9b"]["requires_calculation"] = df_dict["gemma2_9b"]["keywords"].str.contains("requires-calculation")
    gemma2_9b_with_calc = df_dict["gemma2_9b"][df_dict["gemma2_9b"]["requires_calculation"]]
    gemma2_9b_mean = gemma2_9b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gemma2_9b_temp1"]["requires_calculation"] = df_dict["gemma2_9b_temp1"]["keywords"].str.contains("requires-calculation")
    gemma2_9b_temp1_with_calc = df_dict["gemma2_9b_temp1"][df_dict["gemma2_9b_temp1"]["requires_calculation"]]
    gemma2_9b_temp1_mean = gemma2_9b_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gpt35turbo"]["requires_calculation"] = df_dict["gpt35turbo"]["keywords"].str.contains("requires-calculation")
    gpt35turbo_with_calc = df_dict["gpt35turbo"][df_dict["gpt35turbo"]["requires_calculation"]]
    gpt35turbo_mean = gpt35turbo_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gpt4"]["requires_calculation"] = df_dict["gpt4"]["keywords"].str.contains("requires-calculation")
    gpt4_with_calc = df_dict["gpt4"][df_dict["gpt4"]["requires_calculation"]]
    gpt4_mean = gpt4_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["gpt4o"]["requires_calculation"] = df_dict["gpt4o"]["keywords"].str.contains("requires-calculation")
    gpt4o_with_calc = df_dict["gpt4o"][df_dict["gpt4o"]["requires_calculation"]]
    gpt4o_mean = gpt4o_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama2_70b"]["requires_calculation"] = df_dict["llama2_70b"]["keywords"].str.contains("requires-calculation")
    llama2_70b_with_calc = df_dict["llama2_70b"][df_dict["llama2_70b"]["requires_calculation"]]
    llama2_70b_mean = llama2_70b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama3_8b"]["requires_calculation"] = df_dict["llama3_8b"]["keywords"].str.contains("requires-calculation")
    llama3_8b_with_calc = df_dict["llama3_8b"][df_dict["llama3_8b"]["requires_calculation"]]
    llama3_8b_mean = llama3_8b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama3_8b_temp1"]["requires_calculation"] = df_dict["llama3_8b_temp1"]["keywords"].str.contains("requires-calculation")
    llama3_8b_temp1_with_calc = df_dict["llama3_8b_temp1"][df_dict["llama3_8b_temp1"]["requires_calculation"]]
    llama3_8b_temp1_mean = llama3_8b_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama3_70b"]["requires_calculation"] = df_dict["llama3_70b"]["keywords"].str.contains("requires-calculation")
    llama3_70b_with_calc = df_dict["llama3_70b"][df_dict["llama3_70b"]["requires_calculation"]]
    llama3_70b_mean = llama3_70b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama3_70b_temp1"]["requires_calculation"] = df_dict["llama3_70b_temp1"]["keywords"].str.contains("requires-calculation")
    llama3_70b_temp1_with_calc = df_dict["llama3_70b_temp1"][df_dict["llama3_70b_temp1"]["requires_calculation"]]
    llama3_70b_temp1_mean = llama3_70b_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama31_8b"]["requires_calculation"] = df_dict["llama31_8b"]["keywords"].str.contains("requires-calculation")
    llama31_8b_with_calc = df_dict["llama31_8b"][df_dict["llama31_8b"]["requires_calculation"]]
    llama31_8b_mean = llama31_8b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama31_8b_temp1"]["requires_calculation"] = df_dict["llama31_8b_temp1"]["keywords"].str.contains("requires-calculation")
    llama31_8b_temp1_with_calc = df_dict["llama31_8b_temp1"][df_dict["llama31_8b_temp1"]["requires_calculation"]]
    llama31_8b_temp1_mean = llama31_8b_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama31_70b"]["requires_calculation"] = df_dict["llama31_70b"]["keywords"].str.contains("requires-calculation")
    llama31_70b_with_calc = df_dict["llama31_70b"][df_dict["llama31_70b"]["requires_calculation"]]
    llama31_70b_mean = llama31_70b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama31_70b_temp1"]["requires_calculation"] = df_dict["llama31_70b_temp1"]["keywords"].str.contains("requires-calculation")
    llama31_70b_temp1_with_calc = df_dict["llama31_70b_temp1"][df_dict["llama31_70b_temp1"]["requires_calculation"]]
    llama31_70b_temp1_mean = llama31_70b_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["llama31_405b"]["requires_calculation"] = df_dict["llama31_405b"]["keywords"].str.contains("requires-calculation")
    llama31_405b_with_calc = df_dict["llama31_405b"][df_dict["llama31_405b"]["requires_calculation"]]
    llama31_405b_mean = llama31_405b_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["mistral_large"]["requires_calculation"] = df_dict["mistral_large"]["keywords"].str.contains("requires-calculation")
    mistral_large_with_calc = df_dict["mistral_large"][df_dict["mistral_large"]["requires_calculation"]]
    mistral_large_mean = mistral_large_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["mixtral"]["requires_calculation"] = df_dict["mixtral"]["keywords"].str.contains("requires-calculation")
    mixtral_with_calc = df_dict["mixtral"][df_dict["mixtral"]["requires_calculation"]]
    mixtral_mean = mixtral_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["mixtral_temp1"]["requires_calculation"] = df_dict["mixtral_temp1"]["keywords"].str.contains("requires-calculation")
    mixtral_temp1_with_calc = df_dict["mixtral_temp1"][df_dict["mixtral_temp1"]["requires_calculation"]]
    mixtral_temp1_mean = mixtral_temp1_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["paperqa"]["requires_calculation"] = df_dict["paperqa"]["keywords"].str.contains("requires-calculation")
    paperqa_with_calc = df_dict["paperqa"][df_dict["paperqa"]["requires_calculation"]]
    paperqa_mean = paperqa_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["phi"]["requires_calculation"] = df_dict["phi"]["keywords"].str.contains("requires-calculation")
    phi_with_calc = df_dict["phi"][df_dict["phi"]["requires_calculation"]]
    phi_mean = phi_with_calc.groupby("topic")["all_correct_"].mean()
    
    df_dict["random_baseline"]["requires_calculation"] = df_dict["random_baseline"]["keywords"].str.contains("requires-calculation")
    random_baseline_with_calc = df_dict["random_baseline"][df_dict["random_baseline"]["requires_calculation"]]
    random_baseline_mean = random_baseline_with_calc.groupby("topic")["all_correct_"].mean()

    # Sort the data based on the maximum value in each group
    if human_dicts is None:
        # Sort the data based on the maximum value in each group
        sorted_data = sorted(
            zip(
                    [
                        claude2_mean,
                        claude3_mean,
                        claude35_mean,
                        commandrplus_mean,
                        galactica_120b_mean,
                        gemini_pro_mean,
                        gemma11_7b_mean,
                        gemma11_7b_temp1_mean,
                        gemma2_9b_mean,
                        gemma2_9b_temp1_mean,
                        gpt35turbo_mean,
                        gpt4_mean,
                        gpt4o_mean,
                        llama2_70b_mean,
                        llama3_8b_mean,
                        llama3_8b_temp1_mean,
                        llama3_70b_mean,
                        llama3_70b_temp1_mean,
                        llama31_8b_mean,
                        llama31_8b_temp1_mean,
                        llama31_70b_mean,
                        llama31_70b_temp1_mean,
                        llama31_405b_mean,
                        mistral_large_mean,
                        mixtral_mean,
                        mixtral_temp1_mean,
                        paperqa_mean,
                        phi_mean,
                        random_baseline_mean,
                    ],
                    [
                        "Claude2",
                        "Claude3",
                        "Claude3.5",
                        "Command-R+",
                        "Galactica-120B",
                        "Gemini-Pro",
                        "Gemma-7B",
                        "Gemma-7B_Temp1",
                        "Gemma2-9B",
                        "Gemma2-9B_Temp1",
                        "GPT-3.5-Turbo",
                        "GPT-4",
                        "GPT-4o",
                        "Llama2-70B",
                        "Llama3-8B",
                        "Llama3-8B_Temp1",
                        "Llama3-70B",
                        "Llama3-70B_Temp1",
                        "Llama3.1-8B",
                        "Llama3.1-8B_Temp1",
                        "Llama3.1-70B",
                        "Llama3.1-70B_Temp1",
                        "Llama3.1-405B",
                        "Mistral-Large2-123B",
                        "Mixtral-8x7B",
                        "Mixtral-8x7B_Temp1",
                        "PaperQA",
                        "Phi3-Medium",
                        "Random Baseline",
                    ],
                    [
                        model_color_map["Claude2"],
                        model_color_map["Claude3"],
                        model_color_map["Claude35"],
                        model_color_map["Command-R+"],
                        model_color_map["Galactica-120B"],
                        model_color_map["Gemini-Pro"],
                        model_color_map["Gemma-7B"],
                        model_color_map["Gemma-7B_Temp1"],
                        model_color_map["Gemma2-9B"],
                        model_color_map["Gemma2-9B_Temp1"],
                        model_color_map["GPT-3.5-Turbo"],
                        model_color_map["GPT-4"],
                        model_color_map["GPT-4o"],
                        model_color_map["Llama2-70B"],
                        model_color_map["Llama3-8B"],
                        model_color_map["Llama3-8B_Temp1"],
                        model_color_map["Llama3-70B"],
                        model_color_map["Llama3-70B_Temp1"],
                        model_color_map["Llama3.1-8B"],
                        model_color_map["Llama3.1-8B_Temp1"],
                        model_color_map["Llama3.1-70B"],
                        model_color_map["Llama3.1-70B_Temp1"],
                        model_color_map["Llama3.1-405B"],
                        model_color_map["Mistral-Large2-123B"],
                        model_color_map["Mixtral-8x7B"],
                        model_color_map["Mixtral-8x7B_Temp1"],
                        model_color_map["PaperQA"],
                        model_color_map["Phi3-Medium"],
                        model_color_map["Random_Baseline"],
                    ],
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
        print(sorted_data)
    else:
        sorted_data = sorted(
            zip(
                    [
                        claude2_mean,
                        claude3_mean,
                        claude35_mean,
                        commandrplus_mean,
                        galactica_120b_mean,
                        gemini_pro_mean,
                        gemma11_7b_mean,
                        gemma11_7b_temp1_mean,
                        gemma2_9b_mean,
                        gemma2_9b_temp1_mean,
                        gpt35turbo_mean,
                        gpt4_mean,
                        gpt4o_mean,
                        llama2_70b_mean,
                        llama3_8b_mean,
                        llama3_8b_temp1_mean,
                        llama3_70b_mean,
                        llama3_70b_temp1_mean,
                        llama31_8b_mean,
                        llama31_8b_temp1_mean,
                        llama31_70b_mean,
                        llama31_70b_temp1_mean,
                        llama31_405b_mean,
                        mistral_large_mean,
                        mixtral_mean,
                        mixtral_temp1_mean,
                        paperqa_mean,
                        phi_mean,
                        random_baseline_mean,
                    ],
                    [
                        "Claude2",
                        "Claude3",
                        "Claude3.5",
                        "Command-R+",
                        "Galactica-120B",
                        "Gemini-Pro",
                        "Gemma-7B",
                        "Gemma-7B_Temp1",
                        "Gemma2-9B",
                        "Gemma2-9B_Temp1",
                        "GPT-3.5-Turbo",
                        "GPT-4",
                        "GPT-4o",
                        "Llama2-70B",
                        "Llama3-8B",
                        "Llama3-8B_Temp1",
                        "Llama3-70B",
                        "Llama3-70B_Temp1",
                        "Llama3.1-8B",
                        "Llama3.1-8B_Temp1",
                        "Llama3.1-70B",
                        "Llama3.1-70B_Temp1",
                        "Llama3.1-405B",
                        "Mistral-Large2-123B",
                        "Mixtral-8x7B",
                        "Mixtral-8x7B_Temp1",
                        "PaperQA",
                        "Phi3-Medium",
                        "Random Baseline",
                    ],
                    [
                        model_color_map["Claude2"],
                        model_color_map["Claude3"],
                        model_color_map["Claude35"],
                        model_color_map["Command-R+"],
                        model_color_map["Galactica-120B"],
                        model_color_map["Gemini-Pro"],
                        model_color_map["Gemma-7B"],
                        model_color_map["Gemma-7B_Temp1"],
                        model_color_map["Gemma2-9B"],
                        model_color_map["Gemma2-9B_Temp1"],
                        model_color_map["GPT-3.5-Turbo"],
                        model_color_map["GPT-4"],
                        model_color_map["GPT-4o"],
                        model_color_map["Llama2-70B"],
                        model_color_map["Llama3-8B"],
                        model_color_map["Llama3-8B_Temp1"],
                        model_color_map["Llama3-70B"],
                        model_color_map["Llama3-70B_Temp1"],
                        model_color_map["Llama3.1-8B"],
                        model_color_map["Llama3.1-8B_Temp1"],
                        model_color_map["Llama3.1-70B"],
                        model_color_map["Llama3.1-70B_Temp1"],
                        model_color_map["Llama3.1-405B"],
                        model_color_map["Mistral-Large2-123B"],
                        model_color_map["Mixtral-8x7B"],
                        model_color_map["Mixtral-8x7B_Temp1"],
                        model_color_map["PaperQA"],
                        model_color_map["Phi3-Medium"],
                        model_color_map["Random_Baseline"],
                    ],
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
        print(sorted_data)

    theta = radar_factory(len(claude2_mean), frame="polygon")

    # Adjust the layout to leave space for labels
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection="radar"))

    for data, label, color in sorted_data:
        # Plot the filled area

        ax.fill(theta, data, alpha=0.2, label=label, color=color)

        # Plot the line
        ax.plot(theta, data, color=color)

    # Rotate labels
    ax.set_varlabels(claude2_mean.index)
    ax.tick_params(pad=30)
    plt.xticks(
        rotation=45, ha="center"
    )  # Adjust horizontal alignment for better readability

    # Remove unnecessary grid lines
    ax.set_yticklabels([])
    ax.set_yticks([])
    # Improve legend placement
    leg = ax.legend(loc=(-0.1, 1.2), ncols=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    fig.tight_layout()

    fig.savefig(
        figures / f"all_questions_models_requires_calculation_radar_{suffix}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        df_dicts = pickle.load(handle)

    with open(data / "humans_as_models_scores.pkl", "rb") as handle:
        human_dicts = pickle.load(handle)

    make_overall_performance_radar_plot(df_dicts["overall"], "overall")
    radarplot_requires_calculation(df_dicts["overall"], None, "overall")

    make_overall_performance_radar_plot(
        df_dicts["human_aligned"], "human", human_dicts["topic_mean"]
    )
    radarplot_requires_calculation(
        df_dicts["human_aligned"],
        human_dicts["topic_mean"],
        "human",
    )

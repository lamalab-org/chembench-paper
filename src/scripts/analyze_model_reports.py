import matplotlib.pyplot as plt
from plotutils import radar_factory, model_color_map
import numpy as np
import matplotlib.pyplot as plt
from paths import figures, data, scripts
import pickle

plt.style.use(scripts / "lamalab.mplstyle")


def make_overall_performance_radar_plot(df_dict, suffix, human_dicts=None):

    claude2_mean = df_dict["claude2"].groupby("topic")["all_correct_"].mean()
    claude2_react_mean = (
        df_dict["claude2_react"].groupby("topic")["all_correct_"].mean()
    )
    claude2_zero_t_mean = (
        df_dict["claude2_zero_t"].groupby("topic")["all_correct_"].mean()
    )
    claude3_mean = df_dict["claude3"].groupby("topic")["all_correct_"].mean()
    galactica_120b_mean = (
        df_dict["galactica_120b"].groupby("topic")["all_correct_"].mean()
    )
    gemini_pro_zero_t_mean = (
        df_dict["gemini_pro_zero_t"].groupby("topic")["all_correct_"].mean()
    )
    gemini_pro_mean = df_dict["gemini_pro"].groupby("topic")["all_correct_"].mean()
    gpt35turbo_mean = df_dict["gpt35turbo"].groupby("topic")["all_correct_"].mean()
    gpt35turbo_zero_t_mean = (
        df_dict["gpt35turbo_zero_t"].groupby("topic")["all_correct_"].mean()
    )
    gpt35turbo_react_mean = (
        df_dict["gpt35turbo_react"].groupby("topic")["all_correct_"].mean()
    )
    gpt4_mean = df_dict["gpt4"].groupby("topic")["all_correct_"].mean()
    # gpt4zero_t_mean = df_dict["gpt4zero_t"].groupby("topic")["all_correct_"].mean()
    llama70b_mean = df_dict["llama70b"].groupby("topic")["all_correct_"].mean()
    mixtral_mean = df_dict["mixtral"].groupby("topic")["all_correct_"].mean()
    pplx7b_chat_mean = df_dict["pplx7b_chat"].groupby("topic")["all_correct_"].mean()
    pplx7b_online_mean = (
        df_dict["pplx7b_online"].groupby("topic")["all_correct_"].mean()
    )
    random_baseline_mean = (
        df_dict["random_baseline"].groupby("topic")["all_correct_"].mean()
    )

    if human_dicts is None:
        # Sort the data based on the maximum value in each group
        sorted_data = sorted(
            list(
                zip(
                    [
                        gpt4_mean,
                        claude2_mean,
                        # claude2_react_mean,
                        claude3_mean,
                        gemini_pro_mean,
                        gpt35turbo_mean,
                        # gpt35turbo_react_mean,
                        # llama70b_mean,
                        # galactica_120b_mean,
                        # mixtral_mean,
                        pplx7b_chat_mean,
                        pplx7b_online_mean,
                        random_baseline_mean,
                    ],
                    [
                        "GPT-4",
                        "Claude2",
                        # "Claude2-ReAct",
                        "Claude3",
                        "Gemini-Pro",
                        "GPT-3.5-Turbo",
                        # "GPT-3.5-Turbo-ReAct",
                        # "LLAMA-2-70B",
                        # "Galactica-120B",
                        # "MixTRAL-8x7B",
                        "PPLX-7B-Chat",
                        "PPLX-7B-Online",
                        "Random Baseline",
                    ],
                    [
                        model_color_map["gpt4"],
                        model_color_map["claude2"],
                        # model_color_map["claude2_react"],
                        model_color_map["claude3"],
                        model_color_map["gemini_pro"],
                        model_color_map["gpt35turbo"],
                        # model_color_map["gpt35turbo_react"],
                        # model_color_map["llama70b"],
                        # model_color_map["galactica_120b"],
                        # model_color_map["mixtral"],
                        model_color_map["pplx7b_chat"],
                        model_color_map["pplx7b_online"],
                        model_color_map["random_baseline"],
                    ],
                )
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
    else:
        sorted_data = sorted(
            list(
                zip(
                    [
                        gpt4_mean,
                        claude2_mean,
                        # claude2_react_mean,
                        claude3_mean,
                        gemini_pro_mean,
                        gpt35turbo_mean,
                        gpt35turbo_react_mean,
                        # llama70b_mean,
                        # galactica_120b_mean,
                        # mixtral_mean,
                        pplx7b_chat_mean,
                        pplx7b_online_mean,
                        random_baseline_mean,
                        human_dicts,
                    ],
                    [
                        "GPT-4",
                        "Claude2",
                        # "Claude2-ReAct",
                        "Claude3",
                        "Gemini-Pro",
                        "GPT-3.5-Turbo",
                        # "GPT-3.5-Turbo-ReAct",
                        # "LLAMA-2-70B",
                        # "Galactica-120B",
                        # "MixTRAL-8x7B",
                        "PPLX-7B-Chat",
                        "PPLX-7B-Online",
                        "Random Baseline",
                        "Average Human",
                    ],
                    [
                        model_color_map["gpt4"],
                        model_color_map["claude2"],
                        # model_color_map["claude2_react"],
                        model_color_map["claude3"],
                        model_color_map["gemini_pro"],
                        model_color_map["gpt35turbo"],
                        # model_color_map["gpt35turbo_react"],
                        model_color_map["llama70b"],
                        # model_color_map["galactica_120b"],
                        # model_color_map["mixtral"],
                        model_color_map["pplx7b_chat"],
                        model_color_map["pplx7b_online"],
                        model_color_map["random_baseline"],
                        model_color_map["human"],
                    ],
                )
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
    theta = radar_factory(len(claude2_mean), frame="polygon")

    print(sorted_data)
    # Adjust the layout to leave space for labels
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection="radar"))
    for data_, label, color in sorted_data:
        # Plot the filled area
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
    ax.legend(loc=(-0.1, 1.2), ncols=4)

    fig.tight_layout()

    fig.savefig(
        figures / f"all_questions_models_completely_correct_radar_{suffix}.pdf",
        bbox_inches="tight",
    )


def radarplot_requires_calculation(df_dict, suffix):
    # to get subset that requires calculation, we can filter
    # model_df.groupby('requires_calculation')['all_correct_'].mean()[1]
    # for each model
    df_dict["claude2"]["requires_calculation"] = df_dict["claude2"][
        "keywords"
    ].str.contains("requires-calculation")

    claude_with_calc = df_dict["claude2"][df_dict["claude2"]["requires_calculation"]]
    claude2_mean = claude_with_calc.groupby("topic")["all_correct_"].mean()

    # claude2_react_mean = (
    #     df_dict["claude2_react"]
    #     .groupby("requires_calculation")["all_correct_"]
    #     .mean()[1]
    # )

    # claude2_zero_t_mean = (
    #     df_dict["claude2_zero_t"]
    #     .groupby("requires_calculation")["all_correct_"]
    #     .mean()[1]
    # )

    df_dict["claude3"]["requires_calculation"] = df_dict["claude3"][
        "keywords"
    ].str.contains("requires-calculation")

    claude3_with_calc = df_dict["claude3"][df_dict["claude3"]["requires_calculation"]]

    claude3_mean = claude3_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["galactica_120b"]["requires_calculation"] = df_dict["galactica_120b"][
        "keywords"
    ].str.contains("requires-calculation")

    galactica_with_calc = df_dict["galactica_120b"][
        df_dict["galactica_120b"]["requires_calculation"]
    ]

    galactica_120b_mean = galactica_with_calc.groupby("topic")["all_correct_"].mean()

    # gemini_pro_zero_t_mean = (
    #     df_dict["gemini_pro_zero_t"]
    #     .groupby("requires_calculation")["all_correct_"]
    #     .mean()[1]
    # )

    df_dict["gemini_pro"]["requires_calculation"] = df_dict["gemini_pro"][
        "keywords"
    ].str.contains("requires-calculation")

    gemini_with_calc = df_dict["gemini_pro"][
        df_dict["gemini_pro"]["requires_calculation"]
    ]

    gemini_pro_mean = gemini_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["gpt35turbo"]["requires_calculation"] = df_dict["gpt35turbo"][
        "keywords"
    ].str.contains("requires-calculation")

    gpt3_with_calc = df_dict["gpt35turbo"][
        df_dict["gpt35turbo"]["requires_calculation"]
    ]

    gpt35turbo_mean = gpt3_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["gpt35turbo_zero_t"]["requires_calculation"] = df_dict["gpt35turbo_zero_t"][
        "keywords"
    ].str.contains("requires-calculation")

    gpt35turbo_zero_t_with_calc = df_dict["gpt35turbo_zero_t"][
        df_dict["gpt35turbo_zero_t"]["requires_calculation"]
    ]

    gpt35turbo_zero_t_mean = gpt35turbo_zero_t_with_calc.groupby("topic")[
        "all_correct_"
    ].mean()

    df_dict["gpt35turbo_react"]["requires_calculation"] = df_dict["gpt35turbo_react"][
        "keywords"
    ].str.contains("requires-calculation")

    gpt35turbo_react_with_calc = df_dict["gpt35turbo_react"][
        df_dict["gpt35turbo_react"]["requires_calculation"]
    ]

    gpt35turbo_react_mean = gpt35turbo_react_with_calc.groupby("topic")[
        "all_correct_"
    ].mean()

    df_dict["gpt4"]["requires_calculation"] = df_dict["gpt4"]["keywords"].str.contains(
        "requires-calculation"
    )

    gpt4_with_calc = df_dict["gpt4"][df_dict["gpt4"]["requires_calculation"]]

    gpt4_mean = gpt4_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["llama70b"]["requires_calculation"] = df_dict["llama70b"][
        "keywords"
    ].str.contains("requires-calculation")

    llama_with_calc = df_dict["llama70b"][df_dict["llama70b"]["requires_calculation"]]

    llama70b_mean = llama_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["mixtral"]["requires_calculation"] = df_dict["mixtral"][
        "keywords"
    ].str.contains("requires-calculation")

    mixtral_with_calc = df_dict["mixtral"][df_dict["mixtral"]["requires_calculation"]]

    mixtral_mean = mixtral_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["pplx7b_chat"]["requires_calculation"] = df_dict["pplx7b_chat"][
        "keywords"
    ].str.contains("requires-calculation")

    pplx7b_chat_with_calc = df_dict["pplx7b_chat"][
        df_dict["pplx7b_chat"]["requires_calculation"]
    ]

    pplx7b_chat_mean = pplx7b_chat_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["pplx7b_online"]["requires_calculation"] = df_dict["pplx7b_online"][
        "keywords"
    ].str.contains("requires-calculation")

    pplx7b_online_with_calc = df_dict["pplx7b_online"][
        df_dict["pplx7b_online"]["requires_calculation"]
    ]

    pplx7b_online_mean = pplx7b_online_with_calc.groupby("topic")["all_correct_"].mean()

    df_dict["random_baseline"]["requires_calculation"] = df_dict["random_baseline"][
        "keywords"
    ].str.contains("requires-calculation")

    random_baseline_with_calc = df_dict["random_baseline"][
        df_dict["random_baseline"]["requires_calculation"]
    ]

    random_baseline_mean = random_baseline_with_calc.groupby("topic")[
        "all_correct_"
    ].mean()

    # Sort the data based on the maximum value in each group
    if human_dicts is None:
        # Sort the data based on the maximum value in each group
        sorted_data = sorted(
            zip(
                [
                    gpt4_mean,
                    claude2_mean,
                    # claude2_react_mean,
                    claude3_mean,
                    gemini_pro_mean,
                    gpt35turbo_mean,
                    # gpt35turbo_react_mean,
                    # llama70b_mean,
                    galactica_120b_mean,
                    mixtral_mean,
                    pplx7b_chat_mean,
                    pplx7b_online_mean,
                    random_baseline_mean,
                ],
                [
                    "GPT-4",
                    "Claude2",
                    # "Claude2-ReAct",
                    "Claude3",
                    "Gemini-Pro",
                    "GPT-3.5-Turbo",
                    # "GPT-3.5-Turbo-ReAct",
                    # "LLAMA-2-70B",
                    "Galactica-120B",
                    # "MixTRAL-8x7B",
                    "PPLX-7B-Chat",
                    "PPLX-7B-Online",
                    "Random Baseline",
                ],
                [
                    model_color_map["gpt4"],
                    model_color_map["claude2"],
                    # model_color_map["claude2_react"],
                    model_color_map["claude3"],
                    model_color_map["gemini_pro"],
                    model_color_map["gpt35turbo"],
                    # model_color_map["gpt35turbo_react"],
                    # model_color_map["llama70b"],
                    model_color_map["galactica_120b"],
                    # model_color_map["mixtral"],
                    model_color_map["pplx7b_chat"],
                    model_color_map["pplx7b_online"],
                    model_color_map["random_baseline"],
                ],
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )
    else:
        sorted_data = sorted(
            zip(
                [
                    gpt4_mean,
                    claude2_mean,
                    # claude2_react_mean,
                    claude3_mean,
                    gemini_pro_mean,
                    gpt35turbo_mean,
                    # gpt35turbo_react_mean,
                    # llama70b_mean,
                    galactica_120b_mean,
                    # mixtral_mean,
                    pplx7b_chat_mean,
                    pplx7b_online_mean,
                    random_baseline_mean,
                    # human_dicts,
                ],
                [
                    "GPT-4",
                    "Claude2",
                    # "Claude2-ReAct",
                    "Claude3",
                    "Gemini-Pro",
                    "GPT-3.5-Turbo",
                    # "GPT-3.5-Turbo-ReAct",
                    # "LLAMA-2-70B",
                    "Galactica-120B",
                    # "MixTRAL-8x7B",
                    "PPLX-7B-Chat",
                    "PPLX-7B-Online",
                    "Random Baseline",
                    # "Average Human",
                ],
                [
                    model_color_map["gpt4"],
                    model_color_map["claude2"],
                    # model_color_map["claude2_react"],
                    model_color_map["claude3"],
                    model_color_map["gemini_pro"],
                    model_color_map["gpt35turbo"],
                    # model_color_map["gpt35turbo_react"],
                    # model_color_map["llama70b"],
                    model_color_map["galactica_120b"],
                    # model_color_map["mixtral"],
                    model_color_map["pplx7b_chat"],
                    model_color_map["pplx7b_online"],
                    model_color_map["random_baseline"],
                    # model_color_map["human"],
                ],
            ),
            key=lambda x: x[0].sum().item(),
            reverse=True,
        )

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
    ax.legend(loc=(-0.1, 1.2), ncols=4)

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
    radarplot_requires_calculation(df_dicts["overall"], "overall")

    # make_overall_performance_radar_plot(
    #     df_dicts["human_aligned"], "human", human_dicts["topic_mean"]
    # )
    # radarplot_requires_calculation(
    #     df_dicts["human_aligned"], "human", human_dicts["topic_mean"]
    # )

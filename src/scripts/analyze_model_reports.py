import matplotlib.pyplot as plt
from chembench.analysis import (
    load_all_reports,
    get_human_scored_questions_with_at_least_n_scores,
    all_correct,
)
from utils import obtain_chembench_repo
import os
from plotutils import radar_factory, model_color_map
import numpy as np
import matplotlib.pyplot as plt
from paths import figures, output, scripts


def make_overall_performance_radar_plot(df_dict):

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
    gpt4zero_t_mean = df_dict["gpt4zero_t"].groupby("topic")["all_correct_"].mean()
    llama70b_mean = df_dict["llama70b"].groupby("topic")["all_correct_"].mean()
    mixtral_mean = df_dict["mixtral"].groupby("topic")["all_correct_"].mean()
    pplx7b_chat_mean = df_dict["pplx7b_chat"].groupby("topic")["all_correct_"].mean()
    pplx7b_online_mean = (
        df_dict["pplx7b_online"].groupby("topic")["all_correct_"].mean()
    )
    random_baseline_mean = (
        df_dict["random_baseline"].groupby("topic")["all_correct_"].mean()
    )

    # Sort the data based on the maximum value in each group
    sorted_data = sorted(
        zip(
            [
                gpt4_mean,
                claude2_mean,
                claude2_react_mean,
                claude3_mean,
                gemini_pro_mean,
                gpt35turbo_mean,
                gpt35turbo_react_mean,
                llama70b_mean,
                galactica_120b_mean,
                mixtral_mean,
                pplx7b_chat_mean,
                pplx7b_online_mean,
                random_baseline_mean,
            ],
            [
                "GPT-4",
                "Claude2",
                "Claude2-ReAct",
                "Claude3",
                "Gemini-Pro",
                "GPT-3.5-Turbo",
                "GPT-3.5-Turbo-ReAct",
                "LLAMA-2-70B",
                "Galactica-120B",
                "MixTRAL-8x7B",
                "PPLX-7B-Chat",
                "PPLX-7B-Online",
                "Random Baseline",
            ],
            [
                model_color_map["gpt4"],
                model_color_map["claude2"],
                model_color_map["claude2_react"],
                model_color_map["claude3"],
                model_color_map["gemini_pro"],
                model_color_map["gpt35turbo"],
                model_color_map["gpt35turbo_react"],
                model_color_map["llama70b"],
                model_color_map["galactica_120b"],
                model_color_map["mixtral"],
                model_color_map["pplx7b_chat"],
                model_color_map["pplx7b_online"],
                model_color_map["random_baseline"],
            ],
        ),
        key=lambda x: np.max(x[0]),
        reverse=True,
    )

    theta = radar_factory(len(claude2_mean), frame="polygon")

    # Adjust the layout to leave space for labels
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="radar"))

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
        figures / "all_questions_models_completely_correct_radar.pdf",
        bbox_inches="tight",
    )


def radarplot_requires_calculation(df_dict):
    # to get subset that requires calculation, we can filter
    # model_df.groupby('requires_calculation')['all_correct_'].mean()[1]
    # for each model
    claude2_mean = (
        df_dict["claude2"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    claude2_react_mean = (
        df_dict["claude2_react"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    claude2_zero_t_mean = (
        df_dict["claude2_zero_t"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    claude3_mean = (
        df_dict["claude3"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    galactica_120b_mean = (
        df_dict["galactica_120b"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    gemini_pro_zero_t_mean = (
        df_dict["gemini_pro_zero_t"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    gemini_pro_mean = (
        df_dict["gemini_pro"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    gpt35turbo_mean = (
        df_dict["gpt35turbo"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    gpt35turbo_zero_t_mean = (
        df_dict["gpt35turbo_zero_t"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    gpt35turbo_react_mean = (
        df_dict["gpt35turbo_react"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    gpt4_mean = (
        df_dict["gpt4"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    llama70b_mean = (
        df_dict["llama70b"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    mixtral_mean = (
        df_dict["mixtral"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    pplx7b_chat_mean = (
        df_dict["pplx7b_chat"].groupby("requires_calculation")["all_correct_"].mean()[1]
    )

    pplx7b_online_mean = (
        df_dict["pplx7b_online"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    random_baseline_mean = (
        df_dict["random_baseline"]
        .groupby("requires_calculation")["all_correct_"]
        .mean()[1]
    )

    # Sort the data based on the maximum value in each group
    sorted_data = sorted(
        zip(
            [
                gpt4_mean,
                claude2_mean,
                claude2_react_mean,
                claude3_mean,
                gemini_pro_mean,
                gpt35turbo_mean,
                gpt35turbo_react_mean,
                llama70b_mean,
                galactica_120b_mean,
                mixtral_mean,
                pplx7b_chat_mean,
                pplx7b_online_mean,
                random_baseline_mean,
            ],
            [
                "GPT-4",
                "Claude2",
                "Claude2-ReAct",
                "Claude3",
                "Gemini-Pro",
                "GPT-3.5-Turbo",
                "GPT-3.5-Turbo-ReAct",
                "LLAMA-2-70B",
                "Galactica-120B",
                "MixTRAL-8x7B",
                "PPLX-7B-Chat",
                "PPLX-7B-Online",
                "Random Baseline",
            ],
            [
                model_color_map["gpt4"],
                model_color_map["claude2"],
                model_color_map["claude2_react"],
                model_color_map["claude3"],
                model_color_map["gemini_pro"],
                model_color_map["gpt35turbo"],
                model_color_map["gpt35turbo_react"],
                model_color_map["llama70b"],
                model_color_map["galactica_120b"],
                model_color_map["mixtral"],
                model_color_map["pplx7b_chat"],
                model_color_map["pplx7b_online"],
                model_color_map["random_baseline"],
            ],
        ),
        key=lambda x: np.max(x[0]),
        reverse=True,
    )

    theta = radar_factory(len(claude2_mean), frame="polygon")

    # Adjust the layout to leave space for labels
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="radar"))

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
        figures / "all_questions_models_requires_calculation_radar.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    ...

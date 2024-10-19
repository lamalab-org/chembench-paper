
import os
import json
import pickle
from paths import figures, data, scripts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import cycle
from plotutils import range_frame

plt.style.use(scripts / "lamalab.mplstyle")

model_file_name_to_label = [
    # "Gemma-1.1-7B-it",
    "Gemma-2-9B-it",
    "Llama-3-8B-Instruct",
    "Llama-3-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-8x7b-Instruct",
]

def compute_scores_subset(data, title):
    results = []
    for model, scores in data.items():
        if model in model_file_name_to_label:
            all_correct_count = 0
            for row in scores["all_correct_"]:
                all_correct_count += int(row)

            t_zero = all_correct_count / len(scores["all_correct_"])
            all_correct_count = 0
            t1_model = model + "-T-one"
            print(t1_model, data.keys())
            if t1_model in data.keys():
                t1_scores = data[t1_model]
                # print(t1_scores)
                all_correct_count_t1 = 0
                for row_t1 in t1_scores["all_correct_"]:
                    all_correct_count_t1 += int(row_t1)

                t_one = all_correct_count_t1 / len(t1_scores["all_correct_"])

                results.append({
                    "Model": model,
                    "T=0": t_zero,
                    "T=1": t_one
                })
    df = pd.DataFrame(results)
    df_melted = df.melt(id_vars=["Model"], value_vars=["T=0", "T=1"],
                        var_name="Temperature", value_name="Score")

    with open("color_palette.json", "r") as f:
        model_color_map = json.load(f)

    unique_models = df_melted["Model"].unique()
    color_palette = {model: color for model, color in model_color_map.items() if model in unique_models}

    fig, ax = plt.subplots()
    sns.swarmplot(x="Temperature", y="Score", hue="Model", data=df_melted, dodge=False, palette=color_palette)

    sns.lineplot(x="Temperature", y="Score", hue="Model", data=df_melted,
        palette=color_palette, markers="", ci=None, legend=False)

    plt.xlabel("Temperature")
    plt.ylabel("Score")
    # Place the legend outside the plot to the right
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    # Adjust axis limits to ensure all data points are fully visible
    ax.set_ylim(df_melted["Score"].min() - 0.01, df_melted["Score"].max() + 0.01)
    range_frame(ax, np.array([0,1]), df_melted['Score'].values)

    plt.savefig(os.path.join(figures, title), bbox_inches="tight")

if __name__ ==  "__main__":

    with open(os.path.join(data, "model_score_dicts.pkl"), 'rb') as f:
        scores = pickle.load(f)

    compute_scores_subset(scores["overall"], "swarm_plot_overall.pdf")
    compute_scores_subset(scores["human_aligned_no_tool"], "swarm_plot_no_tools.pdf")
    compute_scores_subset(scores["human_aligned_tool"], "swarm_plot_tools.pdf")
    compute_scores_subset(scores["human_aligned_combined"], "swarm_plot_combined.pdf")

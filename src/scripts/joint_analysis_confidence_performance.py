from utils import (
    obtain_chembench_repo,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
)
from paths import data, figures, output
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotutils import model_color_map, range_frame
import numpy as np

# the confidence estimates are csv files with index, question name, score
# the model performance is in a pickle with in which we have a dictionary with the model name as key
# the value is a dataframe with the performance per question

outpath = output / "model_confidence_performance"
os.makedirs(outpath, exist_ok=True)

subsets = [
    "is_point_group",
    "is_number_of_isomers",
    "is_number_nmr_peaks",
    "is_gfk",
    "is_dai",
    "is_pictograms",
    "is_name",
    "is_smiles_name",
    "is_organic_reactivity",
    "is_electron_counts",
    "is_chemical_compatibility",
    "is_materials_compatibility",
    "is_oup",
    "is_olympiad",
    "is_toxicology",
    "is_polymer_chemistry",
]


def join_confidence_and_performance(performance_dict):
    chembench = obtain_chembench_repo()
    gpt = pd.read_csv(
        os.path.join(
            chembench,
            "reports",
            "confidence_estimates",
            "confidence_estimates",
            "results_gpt-4.csv",
        )
    )
    gpt["model"] = "GPT-4"

    gpt = pd.merge(gpt, performance_dict["GPT-4"], right_on="name", left_on="name")

    claude_2 = pd.read_csv(
        os.path.join(
            chembench,
            "reports",
            "confidence_estimates",
            "confidence_estimates",
            "results_claude2.csv",
        )
    )

    claude_2["model"] = "Claude 2"

    claude_2 = pd.merge(
        claude_2, performance_dict["Claude 2"], right_on="name", left_on="name"
    )

    claude_3 = pd.read_csv(
        os.path.join(
            chembench,
            "reports",
            "confidence_estimates",
            "confidence_estimates",
            "results_claude3.csv",
        )
    )

    claude_3["model"] = "Claude 3"

    claude_3 = pd.merge(
        claude_3, performance_dict["Claude 3"], right_on="name", left_on="name"
    )

    return {"gpt4": gpt, "claude2": claude_2, "claude3": claude_3}


def make_plot_of_confidence_vs_performance(merged_dicts, suffix: str = ""):
    fig, ax = plt.subplots(
        3, 1, figsize=(ONE_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    for i, (model, df) in enumerate(merged_dicts.items()):
        sns.swarmplot(
            data=df,
            x="estimate",
            y="all_correct",
            ax=ax[i],
            color=model_color_map[model],
        )

    range_frame(ax[0], np.array([0.5, 5.5]), np.array([0, 1]))

    # set shared y axis label
    fig.text(0.01, 0.5, "completely correct", va="center", rotation="vertical")

    ax[-1].set_xlabel("confidence estimate")

    fig.tight_layout()
    fig.savefig(
        figures / f"confidence_vs_performance_{suffix}.pdf", bbox_inches="tight"
    )


def make_subset_analysis(merged_dict, subset, suffix: str = ""):
    for model, df in merged_dict.items():
        relevant_model_performance = df[df[subset]]
        correct = relevant_model_performance[
            relevant_model_performance["all_correct"].astype(bool)
        ]
        incorrect = relevant_model_performance[
            ~relevant_model_performance["all_correct"].astype(bool)
        ]
        num_correct = len(correct)
        num_incorrect = len(incorrect)

        average_confidence_correct = correct["estimate"].mean()
        average_confidence_incorrect = incorrect["estimate"].mean()

        with open(outpath / f"{model}_{subset}_num_correct_{suffix}.txt", "w") as f:
            f.write(f"{num_correct}" + "\endinput")

        with open(outpath / f"{model}_{subset}_num_incorrect_{suffix}.txt", "w") as f:
            f.write(f"{num_incorrect}" + "\endinput")

        with open(
            outpath / f"{model}_{subset}_average_confidence_correct_{suffix}.txt", "w"
        ) as f:
            rounded = np.round(average_confidence_correct, 2)
            f.write(f"{rounded}" + "\endinput")

        with open(
            outpath / f"{model}_{subset}_average_confidence_incorrect_{suffix}.txt", "w"
        ) as f:
            rounded = np.round(average_confidence_incorrect, 2)
            f.write(f"{rounded}" + "\endinput")


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as f:
        performance_dict = pickle.load(f)

    # overall
    merged_dicts = join_confidence_and_performance(performance_dict["overall"])
    make_plot_of_confidence_vs_performance(merged_dicts, suffix="overall")
    # subset analysis
    for subset in subsets:
        make_subset_analysis(merged_dicts, subset, suffix="overall")

    # human aligned
    merged_dicts = join_confidence_and_performance(performance_dict["human_aligned"])
    make_plot_of_confidence_vs_performance(merged_dicts, suffix="human_aligned")
    # subset analysis
    for subset in subsets:
        make_subset_analysis(merged_dicts, subset, suffix="human_aligned")

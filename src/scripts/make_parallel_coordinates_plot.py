import matplotlib.pyplot as plt

from paths import scripts, figures, data

plt.style.use(scripts / "lamalab.mplstyle")

import pandas as pd

import numpy as np

from plotutils import range_frame, parallel_coordinates_plot, model_color_map
from utils import (
    ONE_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_WIDTH_INCH,
)
import pickle


relevant_models = [
    "gpt4",
    "claude2",
    "claude3",
    "llama70b",
    "gemini_pro",
]


clean_model_names = {
    "gpt4": "GPT-4",
    "claude2": "Claude 2",
    "claude3": "Claude 3",
    "llama70b": "Llama 70B",
    "gemini_pro": "Gemini Pro",
}


def prepare_data_for_parallel_coordinates(model_score_dict):
    # the input contains a dictionary with the model name as key
    # the value is a dataframe with the performance per question
    # we also have a column "topic" which is the topic of the question
    # this will be the different axes in the parallel coordinates plot
    # the color is determined by the model

    all_aligned = []
    for model, frame in model_score_dict.items():
        if model not in relevant_models:
            continue
        frame["model"] = model
        all_aligned.append(frame)

    all_aligned = pd.concat(all_aligned)

    parallel_coordinates_data = all_aligned.pivot_table(
        index="model", columns="topic", values="all_correct_", aggfunc=np.mean
    )

    return parallel_coordinates_data


def plot_parallel_coordinates(parallel_coordinates_data, suffix=""):
    fig, ax = plt.subplots(
        1, 1, figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    print(parallel_coordinates_data)
    parallel_coordinates_data_raw = [
        parallel_coordinates_data[col].fillna(0).values
        for col in parallel_coordinates_data.columns
    ]
    parallel_coordinates_plot(
        None,
        len(list(parallel_coordinates_data.index)),
        parallel_coordinates_data_raw,
        category=np.arange(
            len(list(parallel_coordinates_data.index))
        ),  # list(parallel_coordinates_data.index),
        colors=[model_color_map[model] for model in parallel_coordinates_data.index],
        ynames=list(parallel_coordinates_data.columns),
        ax=ax,
        category_names=[
            clean_model_names[model] for model in parallel_coordinates_data.index
        ],
    )

    fig.savefig(figures / f"parallel_coordinates_{suffix}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        model_scores = pickle.load(handle)["overall"]

    parallel_coordinates_data = prepare_data_for_parallel_coordinates(model_scores)

    plot_parallel_coordinates(parallel_coordinates_data, suffix="overall")

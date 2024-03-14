from glob import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plotutils import range_frame
from paths import output, scripts

plt.style.use(scripts / "lamalab.mplstyle")


model_file_name_to_label = {
    "claude2": "Claude 2",
    "claude2-react": "Claude 2 + ReAct",
    "claude3": "Claude 3",
    "galatica_120b": "Galactica 120B",
    "gemin-pro-zero-T": "Gemini Pro (T=0)",
    "gemini-pro": "Gemini Pro",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4-zero-T": "GPT-4 (T=0)",
    "gpt-35-turbo-react": "GPT-3.5 Turbo + ReAct",
    "llama-2-70b-chat": "Llama 70b",
    "mixtral-8x7b-instruct": "Mixtral 8x7b",
    "pplx-7b-chat": "Perplexity 7B chat",
    "pplx-7b-online": "Perplexity 7B online",
    "random_baseline": "Random baseline",
}


human_dir = output / "human_scores"
model_subset_dir = output / "human_subset_model_scores"
model_dir = output / "overall_model_scores"


def collect_human_scores():
    human_jsons = [
        c for c in glob(os.path.join(human_dir, "cl*.json")) if not "claude" in c
    ]

    scores = []

    for json_file in human_jsons:

        with open(json_file, "r") as handle:
            d = json.load(handle)

        if len(d["model_scores"]) > 100:
            scores.append(d["fraction_correct"])


def collect_model_scores(reportdir):
    model_reports = {}

    for model in model_file_name_to_label.keys():
        try:
            file = os.path.join(reportdir, model + ".json")

            with open(file, "r") as handle:
                d = json.load(handle)

                model_reports[model] = d["fraction_correct"]
        except Exception:
            pass

    model_scores = [(model_file_name_to_label[k], v) for k, v in model_reports.items()]

    return model_scores


def plot_performance(model_scores, outname, human_scores=None):
    model_scores = sorted(model_scores, key=lambda x: x[1])
    fig, ax = plt.subplots()
    ax.hlines(
        np.arange(len(model_scores)),
        0,
        [s[1] for s in model_scores],
        label="Model scores",
        color="#007acc",
        alpha=0.2,
        linewidth=5,
    )
    ax.plot(
        [s[1] for s in model_scores],
        np.arange(len(model_scores)),
        "o",
        markersize=5,
        color="#007acc",
        alpha=0.6,
    )

    scores = [s[1] for s in model_scores]

    range_frame(ax, np.array([0, max(scores)]), np.arange(len(model_scores)), pad=0.1)
    ax.set_yticks(np.arange(len(model_scores)))
    ax.set_yticklabels([s[0] for s in model_scores])

    if human_scores is not None:
        ax.vlines(
            np.mean(human_scores),
            -1,
            len(model_scores),
            label="Human mean",
            color="#00B945",
            alpha=0.6,
        )
        ax.vlines(
            np.max(human_scores),
            -1,
            len(model_scores),
            label="Human max",
            color="#FF9500",
            alpha=0.6,
        )
        ax.vlines(
            np.min(human_scores),
            -1,
            len(model_scores),
            label="Human min",
            color="#845B97",
            alpha=0.6,
        )

        ax.text(
            np.mean(human_scores),
            len(model_scores) - 0.5,
            "mean human",
            rotation=45,
            c="#00B945",
        )

        ax.text(
            np.max(human_scores),
            len(model_scores) - 0.5,
            "best human",
            rotation=45,
            c="#FF9500",
        )

        ax.text(
            np.min(human_scores),
            len(model_scores) - 0.5,
            "worst human",
            rotation=45,
            c="#845B97",
        )

    ax.set_xlabel("fraction of completely correct answers")
    # fig.tight_layout()
    fig.savefig(outname, bbox_inches="tight")
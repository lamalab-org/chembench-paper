import matplotlib.pyplot as plt
from plotutils import radar_factory, model_color_map
import pickle
import pandas as pd
from paths import figures, data, scripts
from definitions import MODELS_TO_PLOT

plt.style.use(scripts / "lamalab.mplstyle")


def load_model_scores():
    with open(data / "model_score_dicts.pkl", "rb") as handle:
        return pickle.load(handle)


def load_human_scores():
    with open(data / "humans_as_models_scores_combined.pkl", "rb") as handle:
        return pickle.load(handle)


def prepare_data(model_scores, score_type, human_scores=None):
    data = []
    for model, df in model_scores[score_type].items():
        if model not in MODELS_TO_PLOT:
            continue
        mean_scores = df.groupby("topic")["all_correct_"].mean()
        data.append((mean_scores, model, model_color_map.get(model, "gray")))

    if human_scores is not None:
        human_mean_scores = human_scores["topic_mean"]["all_correct_"]
        data.append(
            (
                human_mean_scores,
                "Human (Average)",
                model_color_map.get("human", "purple"),
            )
        )

    return sorted(data, key=lambda x: x[0].sum(), reverse=True)


def create_radar_plot(sorted_data, suffix):
    theta = radar_factory(len(sorted_data[0][0]), frame="polygon")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection="radar"))

    for data, label, color in sorted_data:
        ax.fill(theta, data, alpha=0.2, label=label, color=color)
        ax.plot(theta, data, color=color)

    ax.set_varlabels(sorted_data[0][0].index)
    ax.tick_params(pad=30)
    plt.xticks(rotation=45, ha="center")

    ax.set_yticklabels([])
    ax.set_yticks([])

    leg = ax.legend(loc=(-0.1, 1.2), ncols=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    fig.tight_layout()
    fig.savefig(
        figures / f"all_questions_models_completely_correct_radar_{suffix}.pdf",
        bbox_inches="tight",
    )


def main():
    model_scores = load_model_scores()
    human_scores = load_human_scores()

    # Overall performance plot (without human scores)
    overall_data = prepare_data(model_scores, "overall")
    create_radar_plot(overall_data, "overall")

    # Combined plot with human scores
    combined_data = prepare_data(model_scores, "human_aligned_combined", human_scores)
    create_radar_plot(combined_data, "human")


if __name__ == "__main__":
    main()

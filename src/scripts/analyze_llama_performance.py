import matplotlib.pyplot as plt
from paths import scripts, figures, data
from utils import ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH
import pickle


plt.style.use(scripts / "lamalab.mplstyle")


def plot_performance(model_score_dict):
    # make a dummy plots
    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))

    sevenb = model_score_dict['llama7b']["all_correct_"].mean()
    thirteenb = model_score_dict['llama13b']['all_correct_'].mean()
    seventyb = model_score_dict['llama70b']['all_correct_'].mean()

    ax.plot(
        [7, 13, 70],
        [sevenb, thirteenb, seventyb],
        "o",
        markersize=5,
        linestyle="-",
    )

    ax.set_xlabel("number of parameters in billions")
    ax.set_ylabel("fraction ")

    fig.savefig(figures / "llama_performance.pdf", bbox_inches="tight")


if __name__ == '__main__':
    with open(data / 'model_score_dicts.pkl', 'rb') as f:
        model_score_dicts = pickle.load(f)

    plot_performance(model_score_dicts['overall'])

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from paths import data, scripts, figures
from utils import ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, ONE_COL_WIDTH_INCH

plt.style.use(scripts / "lamalab.mplstyle")


def load_embeddings_and_labels():
    # read npy file
    embeddings = np.load(data / "embeddings.npy")
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    df = pd.read_pickle(data / "questions.pkl")
    labels = df["topic"].values
    return embeddings, labels


def plot_pca_map():
    embeddings, labels = load_embeddings_and_labels()
    labels = [label.split("(")[0] for label in labels]
    # PCA
    embeddings_pca = PCA(n_components=2).fit_transform(embeddings)
    pca_1, pca_2 = embeddings_pca[:, 0], embeddings_pca[:, 1]

    # shuffle
    np.random.seed(0)
    shuffled_indices = np.random.permutation(len(pca_1))
    pca_1 = pca_1[shuffled_indices]
    pca_2 = pca_2[shuffled_indices]
    labels = np.array(labels)[shuffled_indices]
    f = plt.figure(figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    # place legend outside of the plot

    plt.axis("off")
    sns.scatterplot(x=pca_1, y=pca_2, hue=labels, s=10, alpha=0.7)
    f.savefig(figures / "question_diversity.pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot_pca_map()

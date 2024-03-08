import numpy as np
import matplotlib.pyplot as plt
plt.style.use("lamalab.mplstyle")
from sklearn.decomposition import PCA
import seaborn as sns

def load_embeddings_and_labels():
    # read npy file
    embeddings = np.load("gpt_save/embeddings.npy")
    labels = np.load("gpt_save/labels.npy")
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
    f = plt.figure(figsize=(7.5, 7.5))
    # place legend outside of the plot

    plt.axis('off')
    sns.scatterplot(x=pca_1, y=pca_2, hue=labels, s=100, alpha=0.5)
    f.savefig("question_diversity.pdf", bbox_inches='tight')

if __name__ == "__main__":
  plot_pca_map()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_tsne(embeddings, label):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = tsne.fit_transform(np.array(embeddings))
    plt.scatter(reduced[:, 0], reduced[:, 1], label=label)
    for i, _ in enumerate(embeddings):
        plt.annotate(f"S{i+1}", (reduced[i, 0], reduced[i, 1]))

def plot_pca(embeddings_list, sentences, labels):

    plt.figure(figsize=(10, 7))

    # Create short labels S1, S2, ...
    point_labels = [f"S{i+1}" for i in range(len(sentences))]

    for emb, label in zip(embeddings_list, labels):
        reduced = PCA(n_components=2).fit_transform(emb)
        plt.scatter(reduced[:, 0], reduced[:, 1], label=label)

        # Add short labels instead of full sentences
        for i, txt in enumerate(point_labels):
            plt.text(reduced[i, 0], reduced[i, 1], txt, fontsize=8)

    plt.legend()
    plt.title("PCA Projection of Sentence Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

def plot_conf_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title); plt.show()

def plot_f1_scores(scores: dict):
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    plt.ylim(0, 1.05); plt.ylabel("F1 Score")
    plt.title("F1 Score Comparison"); plt.show()
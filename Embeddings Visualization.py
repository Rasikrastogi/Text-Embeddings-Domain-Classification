#%%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils.plotting import plot_tsne, plot_pca

# Load models
sbert = SentenceTransformer("all-MiniLM-L6-v2")
mpnet = SentenceTransformer("all-mpnet-base-v2")
use = SentenceTransformer("distiluse-base-multilingual-cased-v2")  # stable PyTorch USE

# Sentences
sentences = [
    "The cat is sleeping on the sofa.",
    "A dog is lying on the couch.",
    "Quantum physics is a fascinating subject.",
    "He submitted the assignment late.",
    "The iPhone 15 has a new camera system.",
    "COVID-19 impacted global economies.",
    "She baked a delicious chocolate cake.",
    "Vaccines help build immunity against viruses."
]

# Assign short IDs
ids = [f"S{i+1}" for i in range(len(sentences))]

# Encode
emb_sbert = sbert.encode(sentences)
emb_mpnet = mpnet.encode(sentences)
emb_use = use.encode(sentences)

# Cosine similarity matrices
sim_sbert = cosine_similarity(emb_sbert)
sim_mpnet = cosine_similarity(emb_mpnet)
sim_use = cosine_similarity(emb_use)

# Convert to DataFrames with IDs
df_sbert = pd.DataFrame(sim_sbert, index=ids, columns=ids).round(2)
df_mpnet = pd.DataFrame(sim_mpnet, index=ids, columns=ids).round(2)
df_use = pd.DataFrame(sim_use, index=ids, columns=ids).round(2)

# Print tables
print("Cosine Similarity - SBERT\n", df_sbert, "\n")
print("Cosine Similarity - MPNet\n", df_mpnet, "\n")
print("Cosine Similarity - USE\n", df_use, "\n")

# Plot heatmaps (and save separately)
sns.set(font_scale=0.8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(df_sbert, annot=True, cmap="coolwarm", cbar=False, ax=axes[0])
axes[0].set_title("Cosine Similarity - SBERT")

sns.heatmap(df_mpnet, annot=True, cmap="coolwarm", cbar=False, ax=axes[1])
axes[1].set_title("Cosine Similarity - MPNet")

sns.heatmap(df_use, annot=True, cmap="coolwarm", cbar=False, ax=axes[2])
axes[2].set_title("Cosine Similarity - USE")

plt.tight_layout()
plt.savefig("cosine_similarity_all.png")  # one combined image
plt.show()

# Save individual images
plt.figure(figsize=(5, 4))
sns.heatmap(df_sbert, annot=True, cmap="coolwarm", cbar=False)
plt.title("Cosine Similarity - SBERT")
plt.tight_layout()
plt.savefig("cosine_similarity_sbert.png")
plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(df_mpnet, annot=True, cmap="coolwarm", cbar=False)
plt.title("Cosine Similarity - MPNet")
plt.tight_layout()
plt.savefig("cosine_similarity_mpnet.png")
plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(df_use, annot=True, cmap="coolwarm", cbar=False)
plt.title("Cosine Similarity - USE")
plt.tight_layout()
plt.savefig("cosine_similarity_use.png")
plt.close()

# t-SNE & PCA plots
plot_tsne(emb_sbert, "SBERT")
plot_tsne(emb_mpnet, "MPNet")
plot_tsne(emb_use, "USE")

plot_pca([emb_sbert, emb_mpnet, emb_use], sentences, ["SBERT", "MPNet", "USE"])

# %%

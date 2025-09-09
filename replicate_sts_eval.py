#%%
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# ---------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------
def normalize_labels(dataset, labels):
    labels = np.array(labels, dtype=float)
    if dataset == "STSBenchmark":
        return labels / 5.0
    elif dataset == "SICK-R":
        return (labels - 1.0) / 4.0
    elif dataset == "BIOSSES":
        return labels / 4.0
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# ---------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------
def evaluate_sts(model, dataset, n_samples=200):
    print(f"\n▶ Evaluating {model} on {dataset} ...")

    if dataset == "STSBenchmark":
        ds = load_dataset("glue", "stsb", split=f"validation[:{n_samples}]")
        sents1, sents2, labels = ds["sentence1"], ds["sentence2"], ds["label"]

    elif dataset == "SICK-R":
        ds = load_dataset("sick", split=f"test[:{n_samples}]", trust_remote_code=True)
        sents1, sents2, labels = ds["sentence_A"], ds["sentence_B"], ds["relatedness_score"]

    elif dataset == "BIOSSES":
        ds = load_dataset("biosses", split=f"train[:{n_samples}]")
        sents1, sents2, labels = ds["sentence1"], ds["sentence2"], ds["score"]

    else:
        raise ValueError(f"Dataset {dataset} not supported")

    labels = normalize_labels(dataset, labels)

    model = SentenceTransformer(model)

    emb1 = model.encode(sents1, convert_to_tensor=True, show_progress_bar=True)
    emb2 = model.encode(sents2, convert_to_tensor=True, show_progress_bar=True)

    pred_scores = util.cos_sim(emb1, emb2).cpu().numpy().diagonal()

    pearson_corr, _ = pearsonr(pred_scores, labels)
    spearman_corr, _ = spearmanr(pred_scores, labels)

    print(f"Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
    return pearson_corr, spearman_corr

# ---------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(project="domain-classification", name="STS-Replication")

    models = {
        "SBERT-MiniLM": "all-MiniLM-L6-v2",
        "MPNet": "all-mpnet-base-v2",
        "USE": "distiluse-base-multilingual-cased-v2"
    }
    datasets = ["STSBenchmark", "SICK-R", "BIOSSES"]

    results = []
    for model_name, model_path in models.items():
        for dataset in datasets:
            p, s = evaluate_sts(model_path, dataset)
            results.append({
                "Model": model_name,
                "Dataset": dataset,
                "Pearson": p,
                "Spearman": s
            })

            # Log numbers to wandb
            wandb.log({
                "Model": model_name,
                "Dataset": dataset,
                "Pearson": p,
                "Spearman": s
            })

    df = pd.DataFrame(results)
    print("\nFinal Results:\n", df)

    # ---------------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------------
    sns.set(style="whitegrid", font_scale=1.1)

    # Pearson plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Dataset", y="Pearson", hue="Model", palette="Set2", ci=None, ax=ax)
    ax.set_title("STS Replication (Pearson Correlation)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Task")
    ax.legend(title="Model")
    plt.tight_layout()
    wandb.log({"Pearson Comparison": wandb.Image(fig)})
    plt.show()

    # Spearman plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Dataset", y="Spearman", hue="Model", palette="Set2", ci=None, ax=ax)
    ax.set_title("STS Replication (Spearman Correlation)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Task")
    ax.legend(title="Model")
    plt.tight_layout()
    wandb.log({"Spearman Comparison": wandb.Image(fig)})
    plt.show()



# %%

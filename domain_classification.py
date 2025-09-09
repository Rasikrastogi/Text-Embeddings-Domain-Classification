#%% 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import wandb

# Local utils
from utils.data_utils import load_balanced_3domains, get_sentence_pairs_and_labels
from utils.embedding_utils import embed_pairs_variant   

# --------------------------------------------------------------------
# 1. Init wandb
# --------------------------------------------------------------------
wandb.init(project="domain-classification", name="LogReg-Variants")

# --------------------------------------------------------------------
# 2. Load balanced dataset
# --------------------------------------------------------------------
df = load_balanced_3domains()
X_pairs, y = get_sentence_pairs_and_labels(df)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------------------------------------------------
# 3. Load embedding models
# --------------------------------------------------------------------
sbert = SentenceTransformer("all-MiniLM-L6-v2")
mpnet = SentenceTransformer("all-mpnet-base-v2")
use = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# --------------------------------------------------------------------
# 4. Run Logistic Regression across models + variants
# --------------------------------------------------------------------
variants = ["diff", "concat3", "concatcos", "concat", "cos"]
results = []

for name, model, is_use in [
    ("SBERT", sbert, False),
    ("MPNet", mpnet, False),
    ("USE", use, False),
]:
    for var in variants:
        print(f"\n▶ Running {name} with variant={var} ...")
        X = embed_pairs_variant(X_pairs, model, variant=var, is_use=is_use)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Collect metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")

        results.append({
            "Model": name,
            "Variant": var,
            "Dims": X.shape[1],
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        # Log metrics to wandb
        wandb.log({
            "Model": name,
            "Variant": var,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

# --------------------------------------------------------------------
# 5. Build DataFrame
# --------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n Results DataFrame")
print(df_results.to_string(index=False))

# --------------------------------------------------------------------
# 6. Create grouped bar plots for metrics
# --------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1"]

order = (
    df_results.groupby("Variant")["Dims"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

hue_order = ["SBERT", "MPNet", "USE"]

for metric in metrics:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=df_results,
        x="Variant", y=metric, hue="Model",
        ci=None, order=order, hue_order=hue_order, ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Performance by Variant — {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Feature Variant")
    ax.legend(title="Embedding Model")
    plt.tight_layout()
    wandb.log({f"{metric} Comparison": wandb.Image(fig)})
    plt.show()



# %%

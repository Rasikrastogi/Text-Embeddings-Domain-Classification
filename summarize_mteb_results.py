#%%
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

def collect_results(base_dir="./", models=None):
    if models is None:
        models = ["sbert", "sbert-mini", "use"]

    results = []

    for model in models:
        folder = os.path.join(base_dir, f"results_{model}")
        if not os.path.exists(folder):
            continue

        # Walk through all subfolders
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".json") and "model_meta" not in file:
                    with open(os.path.join(root, file), "r") as f:
                        data = json.load(f)

                    task = data.get("task_name", file.replace(".json", ""))
                    pearson, spearman = None, None

                    # Extract correlation scores correctly
                    if "scores" in data and "test" in data["scores"]:
                        test_scores = data["scores"]["test"][0]
                        pearson = test_scores.get("pearson")
                        spearman = test_scores.get("spearman")

                    results.append({
                        "Model": model.upper(),
                        "Task": task,
                        "Pearson": pearson,
                        "Spearman": spearman
                    })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # --- Init wandb ---
    wandb.init(project="domain-classification", name="MTEB-Summary")

    df = collect_results()

    if df.empty:
        print("⚠️ No results found. Did you run run_mteb.py first?")
    else:
        print("✅ Results collected")
        print(df)

        # Pivot for visualization
        df_plot = df.melt(
            id_vars=["Model", "Task"],
            value_vars=["Pearson", "Spearman"],
            var_name="Metric", value_name="Score"
        )

       # Separate Pearson plot
    df_pearson = df_plot[df_plot["Metric"] == "Pearson"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_pearson, x="Task", y="Score", hue="Model", errorbar=None, ax=ax)
    ax.set_title("MTEB Results (Pearson Correlation)")
    ax.set_ylim(0.7, 1)
    ax.legend(title="Model")
    plt.xticks(rotation=30)
    plt.grid(axis="y")
    plt.tight_layout()
    wandb.log({"Pearson Comparison": wandb.Image(fig)})
    plt.show()

    # Separate Spearman plot
    df_spearman = df_plot[df_plot["Metric"] == "Spearman"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_spearman, x="Task", y="Score", hue="Model", errorbar=None, ax=ax)
    ax.set_title("MTEB Results (Spearman Correlation)")
    ax.set_ylim(0.7, 1)
    ax.legend(title="Model")
    plt.xticks(rotation=30)
    plt.grid(axis="y")
    plt.tight_layout()
    wandb.log({"Spearman Comparison": wandb.Image(fig)})
    plt.show()

# %%

import pandas as pd
from datasets import load_dataset

def load_balanced_3domains(
    sts_size=100,
    sick_size=100,
    biosses_size=100,
    seed=42
):
    """
    Load 3 domain datasets (STS-B, SICK-R, BIOSSES),
    sample balanced subsets, and return a DataFrame with
    sentence1, sentence2, label (domain).
    """
    data = []

    # --- News: STS-Benchmark ---
    if sts_size > 0:
        sts = load_dataset("glue", "stsb", split=f"train[:{sts_size}]")
        for ex in sts:
            data.append({
                "sentence1": ex["sentence1"],
                "sentence2": ex["sentence2"],
                "label": "news"
            })

    # --- NLP: SICK-R ---
    if sick_size > 0:
        sick = load_dataset("sick", split=f"train[:{sick_size}]", trust_remote_code=True)
        for ex in sick:
            data.append({
                "sentence1": ex["sentence_A"],
                "sentence2": ex["sentence_B"],
                "label": "nlp"
            })

    # --- Biomedical: BIOSSES ---
    if biosses_size > 0:
        biosses = load_dataset("biosses", split=f"train[:{biosses_size}]")
        for ex in biosses:
            data.append({
                "sentence1": ex["sentence1"],
                "sentence2": ex["sentence2"],
                "label": "biomedical"
            })

    # --- Build DataFrame ---
    df = pd.DataFrame(data)

    if not df.empty:
        # Balance: sample equal numbers from each domain
        df = df.groupby("label").sample(
            n=df["label"].value_counts().min(),
            random_state=seed
        ).reset_index(drop=True)

    return df


def get_sentence_pairs_and_labels(df):
    """
    Extract (s1, s2) pairs and labels from DataFrame
    """
    X_pairs = list(zip(df["sentence1"], df["sentence2"]))
    y = df["label"].tolist()
    return X_pairs, y

#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer
from utils.data_utils import load_balanced_3domains, get_sentence_pairs_and_labels
from utils.embedding_utils import embed_pairs_variant   

# --------------------------------------------------------------------
# 1. Load balanced dataset
# --------------------------------------------------------------------
df = load_balanced_3domains()
X_pairs, y = get_sentence_pairs_and_labels(df)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------------------------------------------------
# 2. Models + Variants
# --------------------------------------------------------------------
models = [
    ("SBERT", SentenceTransformer("all-MiniLM-L6-v2"), False),
    ("MPNet", SentenceTransformer("all-mpnet-base-v2"), False),
    ("USE", SentenceTransformer("distiluse-base-multilingual-cased-v2"), False),
]
variants = ["diff", "concat3", "concatcos", "concat", "cos"]

# --------------------------------------------------------------------
# 3. Custom sentences
# --------------------------------------------------------------------
custom_sentences = [
    "The government announced new economic reforms to stabilize the market.",  # news-like
    "Neural networks are widely used in natural language processing tasks.",  # nlp-like
    "The latest research shows significant progress in cancer treatment.",   # biomedical-like
]

# --------------------------------------------------------------------
# 4. Train + Demo
# --------------------------------------------------------------------
for name, model, is_use in models:
    for var in variants:
        print(f"\n▶ Training {name} (variant={var}) for demo ...")
        X = embed_pairs_variant(X_pairs, model, variant=var, is_use=is_use)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)

        X_custom_pairs = embed_pairs_variant(
            [(s, s) for s in custom_sentences], model, variant=var, is_use=is_use
        )
        probs = clf.predict_proba(X_custom_pairs)

        print(f"\n=== {name} ({var}) ===")
        for sent, p in zip(custom_sentences, probs):
            print(f"\nSentence: {sent}")
            for label, score in zip(le.classes_, p):
                print(f"  {label}: {score*100:.2f}%")

# %%

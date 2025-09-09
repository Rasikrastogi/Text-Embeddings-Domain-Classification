#%%
# run_mteb.py

from mteb import MTEB
from sentence_transformers import SentenceTransformer

# --- Define models (original, simple, error-free) ---
models = {
    "sbert": SentenceTransformer("all-mpnet-base-v2"),       # strong English model
    "sbert-mini": SentenceTransformer("all-MiniLM-L6-v2"),   # lightweight SBERT
    "use": SentenceTransformer("distiluse-base-multilingual-cased-v2"),  # acts as USE (English also works)
}

# --- Define tasks (3 domains) ---
tasks = [
    "STSBenchmark",
    "SICK-R",
    "BIOSSES"
]

# --- Run evaluation ---
for task in tasks:
    print(f"\n===== Running task: {task} =====")
    for model_name, model in models.items():
        print(f"\n--- Model: {model_name} ---")
        evaluation = MTEB(tasks=[task])
        evaluation.run(model, output_folder=f"./results_{model_name}")


    # %%

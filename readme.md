#  Data Science Seminar – Text Embeddings & Domain Classification

This repository contains the implementation and results of my **Data Science Seminar** project at the University of Passau.  
The focus of the seminar was on **evaluating text embedding models across multiple domains** and analyzing their performance in similarity tasks and domain classification.

---

##  Project Overview
The main goal was to compare **sentence embedding models** on datasets from different domains (news, biomedical, technical, and social media).  
We evaluated how embeddings capture semantic similarity and how well they generalize across domains.

**Key objectives:**
- Benchmark multiple embedding models.
- Experiment with feature engineering strategies.
- Evaluate using both similarity scores and classification tasks.
- Analyze trade-offs between model complexity, dimensions, and accuracy.

---

##  Project Structure

```
Data Science Project/
│
├── results_sbert/              # Evaluation results for SBERT (all-MiniLM-L6-v2)
├── results_sbert-mini/         # Evaluation results for SBERT-mini variant
├── results_use/                # Evaluation results for Universal Sentence Encoder
│
├── utils/                      # Utility functions
│   ├── data_utils.py           # Dataset loading, preprocessing & sampling
│   ├── embedding_utils.py      # Functions for handling embeddings & similarity
│   ├── plotting.py             # Plotting utilities for results visualization
│   └── __init__.py             # Marks utils as a package
│
├── demo_sentences_real.py      # Quick demo with example sentence embeddings
├── domain_classification.py    # Domain classification experiments
├── Embeddings Visualization.py # Scripts for plotting embeddings (t-SNE/PCA)
├── replicate_sts_eval.py       # Reproduces STS-style evaluation
├── run_mteb.py                 # Runs MTEB benchmark tasks
├── summarize_mteb_results.py   # Aggregates & summarizes benchmark results
│
├── wandb/                      # Weights & Biases experiment tracking logs
│
└── README.md                   # Project documentation
```

---

##  Execution Flow (Step-by-Step)

To reproduce experiments, follow this sequence:

1. **Data Preparation**
   - `utils/data_utils.py` → Loads and preprocesses datasets.  
   - Ensures class balance and text cleaning.

2. **Embedding Utilities**
   - `utils/embedding_utils.py` → Functions for cosine similarity, vector difference, concatenation, etc.

3. **Core Experiments**
   - `replicate_sts_eval.py` → Run STS-style evaluation (Pearson, Spearman).  
   - `domain_classification.py` → Train logistic regression on domain labels.  
   - `run_mteb.py` → Run official MTEB benchmark tasks (multiple datasets).

4. **Visualization**
   - `Embeddings Visualization.py` → t-SNE & PCA plots of embeddings.  
   - `utils/plotting.py` → Helper functions for correlation heatmaps, bar charts, etc.

5. **Summarize Results**
   - `summarize_mteb_results.py` → Collect results across models & datasets.  
   - Outputs combined metrics for reporting.

6. **Demo**
   - `demo_sentences_real.py` → Small script to quickly test sentence similarity.  
   - Use this for demonstrations without full experiments.

7. **Results**
   - `results_sbert/`, `results_sbert-mini/`, `results_use/` → Contain stored outputs (metrics, plots, logs).  
   - Each folder corresponds to one embedding model.

8. **Experiment Tracking (Optional)**
   - `wandb/` → Logs for Weights & Biases if enabled.

 **Recommended run order**:  
`replicate_sts_eval.py` → `domain_classification.py` → `Embeddings Visualization.py` → `summarize_mteb_results.py`

---

##  Datasets

We used **five domain-specific datasets**, each representing unique challenges:

- **STS Benchmark (STSb)** → *News & general domain*  
- **SICK-R** → *NLP & inference*  
- **BIOSSES** → *Biomedical research articles*  

> For class balance, 100 sentence pairs were sampled from each dataset.

---

##  Embedding Models Evaluated

- **SBERT – all-MiniLM-L6-v2**  
- **MPNet – all-mpnet-base-v2**  
- **Universal Sentence Encoder (USE)**  

---

##  Feature Engineering Variants

We tested multiple approaches to convert embeddings into features:

- **Cosine Similarity (cos)** – similarity between sentence pairs  
- **Vector Difference (diff)** – absolute difference between embeddings  
- **Concatenation (concat)** – [embedding1 ⊕ embedding2]  
- **Concat3 (concat3)** – [embedding1 ⊕ embedding2 ⊕ |embedding1 - embedding2|]  

---

##  Evaluation

Metrics used:
- **Pearson correlation**
- **Spearman correlation**
- **F1 score (for classification tasks)**

We also compared performance **vs. embedding dimensionality**, discussing if higher dimensions justify the computational cost.

---

##  Results Summary

- **SBERT** performed best in terms of efficiency vs. accuracy balance.  
- **MPNet** achieved higher correlations but with higher computation cost.  
- **USE** was efficient for cross-domain tasks but underperformed in biomedical domain.  

**Observation:**  
Increasing dimensions beyond a certain threshold yielded only ~1–2% improvement, raising the question of cost-effectiveness.

---

##  Insights

- Domain-specific datasets (like BIOSSES) remain challenging due to rare vocabulary.  
- Concatenation-based features (concat3) often improved classification accuracy.  
- Results can be linked back to model architectures: transformer depth, pretraining corpus, and dimensionality strongly influence domain generalization.

---

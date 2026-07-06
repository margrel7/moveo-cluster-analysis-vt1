"""
Use Gemma4 via Ollama to generate short German cluster labels
based on the most representative reports in each cluster.

Each cluster is shown the labels already generated for previous clusters
so the LLM is forced to differentiate rather than repeat similar terms.

Outputs a CSV with suggested labels for human review and editing.

Run with:
    PYTHONPATH=src uv run python src/utils/generate_cluster_labels.py
"""

import numpy as np
import pandas as pd
import ollama
from sklearn.metrics.pairwise import cosine_distances

import os
from datetime import datetime
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_CSV    = "outputs/workshop_clusters.csv"
EMBEDDINGS_NPY = f"outputs/strategy2_fused_embeddings_50d_{RUN_ID}.npy"
OUTPUT_CSV     = "outputs/cluster_labels.csv"

OLLAMA_MODEL   = "gemma3:4b"
N_EXAMPLES     = 8   # reports sent to LLM per cluster

PROMPT_TEMPLATE = """Du analysierst Bürgerberichte über den öffentlichen Verkehr in Winterthur.

Diese Gruppen wurden bereits benannt (vermeide ähnliche oder zu allgemeine Labels):
{existing_labels}

Hier sind {n} Berichte aus einer NEUEN Gruppe:

{reports}

Aufgabe: Gib dieser Gruppe ein Label, das sie klar von den anderen Gruppen unterscheidet.
Fokussiere auf das spezifische Thema — nicht auf allgemeine Begriffe wie "Erfahrungen", "Feedback" oder "Probleme".
Das Label soll sofort klar machen, WAS konkret das Thema ist (z.B. Ort, Infrastruktur, Situation).

Antworte NUR in diesem Format (keine weiteren Erklärungen):
DE: <deutsches label, 3-5 Wörter, spezifisch>
EN: <english label, 3-5 words, specific>
DESC: <ein deutscher Satz, max. 15 Wörter, was macht diese Gruppe einzigartig?>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_representative_texts(
    cluster_df: pd.DataFrame,
    cluster_embs: np.ndarray,
    centroid: np.ndarray,
    n: int,
) -> list[str]:
    """Get the n most representative report texts for a cluster."""
    distances = cosine_distances(cluster_embs, centroid.reshape(1, -1)).flatten()
    order = np.argsort(distances)
    sorted_df = cluster_df.iloc[order].reset_index(drop=True)

    texts = []
    for _, row in sorted_df.iterrows():
        if len(texts) >= n:
            break
        text = str(row.get("raw_text", "") or "").strip()
        if text and text != "nan" and len(text) > 10:
            category = "✅" if row.get("category") == "works_well" else "⚠️"
            texts.append(f"{category} {text[:150]}")

    return texts


def parse_response(response: str) -> dict:
    """Parse the LLM response into label components."""
    result = {"label_de": "", "label_en": "", "description_de": ""}
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("DE:"):
            result["label_de"] = line[3:].strip()
        elif line.startswith("EN:"):
            result["label_en"] = line[3:].strip()
        elif line.startswith("DESC:"):
            result["description_de"] = line[5:].strip()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_labels(
    results_csv: str,
    embeddings_npy: str,
    output_csv: str,
):
    print("Loading data...")
    df         = pd.read_csv(results_csv)
    embeddings = np.load(embeddings_npy)

    # Merge raw text
    llava = pd.read_csv("outputs/preprocessed_llava.csv")[["photo_id", "raw_text"]]
    df = df.merge(llava, on="photo_id", how="left")

    print(f"Loaded {len(df)} reports.")
    print(f"Model: {OLLAMA_MODEL}\n")

    rows            = []
    existing_labels = []  # accumulates as clusters are processed

    for cluster_id in sorted(df["hdbscan_label_assigned"].unique()):
        mask         = df["hdbscan_label_assigned"] == cluster_id
        cluster_df   = df[mask].reset_index(drop=True)
        cluster_embs = embeddings[mask.values]
        centroid     = cluster_embs.mean(axis=0)

        n_good = (cluster_df["category"] == "works_well").sum()
        n_prob = (cluster_df["category"] == "problem").sum()

        print(f"Cluster {cluster_id} (n={len(cluster_df)}, {n_good} good, {n_prob} problems)...")

        texts = get_representative_texts(cluster_df, cluster_embs, centroid, N_EXAMPLES)

        if not texts:
            print(f"  [SKIP] No texts available.")
            fallback = {
                "cluster_id":     cluster_id,
                "n_total":        len(cluster_df),
                "n_works_well":   n_good,
                "n_problem":      n_prob,
                "label_de":       f"Cluster {cluster_id}",
                "label_en":       f"Cluster {cluster_id}",
                "description_de": "",
                "llm_raw":        "",
            }
            rows.append(fallback)
            existing_labels.append(fallback)
            continue

        # Build context string from already-labelled clusters
        if existing_labels:
            existing_str = "\n".join([
                f"- Cluster {r['cluster_id']}: {r['label_de']} / {r['label_en']}"
                for r in existing_labels
            ])
        else:
            existing_str = "(noch keine — dies ist die erste Gruppe)"

        reports_text = "\n".join([f"- {t}" for t in texts])
        prompt = PROMPT_TEMPLATE.format(
            n=len(texts),
            reports=reports_text,
            existing_labels=existing_str,
        )

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw    = response["message"]["content"].strip()
            parsed = parse_response(raw)

            print(f"  DE: {parsed['label_de']}")
            print(f"  EN: {parsed['label_en']}")
            print(f"  DESC: {parsed['description_de']}")

            result_row = {
                "cluster_id":     cluster_id,
                "n_total":        len(cluster_df),
                "n_works_well":   n_good,
                "n_problem":      n_prob,
                "label_de":       parsed["label_de"],
                "label_en":       parsed["label_en"],
                "description_de": parsed["description_de"],
                "llm_raw":        raw,
            }
            rows.append(result_row)
            existing_labels.append(result_row)  # feed into next cluster's context

        except Exception as e:
            print(f"  [ERROR] {e}")
            fallback = {
                "cluster_id":     cluster_id,
                "n_total":        len(cluster_df),
                "n_works_well":   n_good,
                "n_problem":      n_prob,
                "label_de":       f"Cluster {cluster_id}",
                "label_en":       f"Cluster {cluster_id}",
                "description_de": "",
                "llm_raw":        "",
            }
            rows.append(fallback)
            existing_labels.append(fallback)

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)

    print(f"\n=== Label Summary ===")
    print(result[["cluster_id", "n_total", "label_de", "label_en"]].to_string(index=False))
    print(f"\nSaved to {output_csv}")
    print("Open the CSV, review and edit the labels, then update CLUSTER_LABELS in generate_cluster_cards.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_labels(
        results_csv    = RESULTS_CSV,
        embeddings_npy = EMBEDDINGS_NPY,
        output_csv     = OUTPUT_CSV,
    )
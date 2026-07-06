import numpy as np
import pandas as pd
from clustering_utils import run_kmeans, run_hdbscan, save_results, print_summary
from src.config import S3_EMBEDDINGS_NPY, S3_EMBEDDINGS_CSV, \
                   S3_CLUSTERING_CSV, RUN_ID, print_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_NAME = f"Strategy 3 — LLM Enrichment (Gemma4 + paraphrase-multilingual) ({RUN_ID})"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_clustering():
    print_config("strategy3_clustering.py")

    print(f"Loading embeddings from {S3_EMBEDDINGS_NPY}...")
    embeddings = np.load(S3_EMBEDDINGS_NPY)
    print(f"Embedding matrix shape: {embeddings.shape}")

    index_map = pd.read_csv(S3_EMBEDDINGS_CSV)
    if "array_index" in index_map.columns:
        index_map = index_map.rename(columns={"array_index": "index"})

    # K-means on full 768-dim normalised space
    # (768-dim is manageable — no UMAP reduction needed before K-means)
    kmeans_labels, best_k, kmeans_silhouette = run_kmeans(
        embeddings, normalise=True
    )

    # HDBSCAN — no precomputed 2D, reduces internally via UMAP
    hdbscan_labels, n_clusters, n_noise = run_hdbscan(
        embeddings, precomputed_2d=None
    )

    results = save_results(
        index_map, index_map, kmeans_labels, hdbscan_labels, best_k, S3_CLUSTERING_CSV
    )
    print_summary(STRATEGY_NAME, results, best_k, kmeans_silhouette, n_clusters, n_noise)


if __name__ == "__main__":
    run_clustering()
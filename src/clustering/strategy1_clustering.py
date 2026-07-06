import numpy as np
import pandas as pd
from clustering_utils import run_kmeans, run_hdbscan, save_results, print_summary
from src.config import S1_EMBEDDINGS_NPY, S1_EMBEDDINGS_CSV, S1_INDEX_MAP_CSV, \
                   S1_CLUSTERING_CSV, RUN_ID, print_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_NAME = f"Strategy 1 — CLIP Joint Embedding ({RUN_ID})"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_clustering():
    print_config("strategy1_clustering.py")

    print(f"Loading embeddings from {S1_EMBEDDINGS_NPY}...")
    embeddings = np.load(S1_EMBEDDINGS_NPY)
    print(f"Embedding matrix shape: {embeddings.shape}")

    index_map = pd.read_csv(S1_INDEX_MAP_CSV)
    metadata  = pd.read_csv(S1_EMBEDDINGS_CSV)

    kmeans_labels, best_k, kmeans_silhouette = run_kmeans(
        embeddings, normalise=True
    )
    hdbscan_labels, n_clusters, n_noise = run_hdbscan(
        embeddings, precomputed_2d=None
    )

    results = save_results(
        index_map, metadata, kmeans_labels, hdbscan_labels, best_k, S1_CLUSTERING_CSV
    )
    print_summary(STRATEGY_NAME, results, best_k, kmeans_silhouette, n_clusters, n_noise)


if __name__ == "__main__":
    run_clustering()
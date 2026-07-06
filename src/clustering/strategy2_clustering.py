import numpy as np
import pandas as pd
from clustering_utils import run_kmeans, run_hdbscan, save_results, print_summary
from src.config import S2_FUSED_50D_NPY, S2_FUSED_2D_NPY, S2_FUSED_CSV, \
                   S2_CLUSTERING_CSV, RUN_ID, print_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_NAME = f"Strategy 2 — Separate Embeddings (MobileNetV3 + multilingual) ({RUN_ID})"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_clustering():
    print_config("strategy2_clustering.py")

    print(f"Loading 50D embeddings from {S2_FUSED_50D_NPY}...")
    embeddings_50d = np.load(S2_FUSED_50D_NPY)
    print(f"50D matrix shape: {embeddings_50d.shape}")

    print(f"Loading 2D embeddings from {S2_FUSED_2D_NPY}...")
    embeddings_2d = np.load(S2_FUSED_2D_NPY)
    print(f"2D matrix shape:  {embeddings_2d.shape}")

    index_map = pd.read_csv(S2_FUSED_CSV)
    if "array_index" in index_map.columns:
        index_map = index_map.rename(columns={"array_index": "index"})

    # K-means on 50D — already UMAP-reduced, no re-normalisation needed
    kmeans_labels, best_k, kmeans_silhouette = run_kmeans(
        embeddings_50d, normalise=False
    )

    # HDBSCAN on pre-computed 2D — skip internal UMAP
    hdbscan_labels, n_clusters, n_noise = run_hdbscan(
        embeddings_50d, precomputed_2d=embeddings_2d
    )

    results = save_results(
        index_map, index_map, kmeans_labels, hdbscan_labels, best_k, S2_CLUSTERING_CSV
    )
    print_summary(STRATEGY_NAME, results, best_k, kmeans_silhouette, n_clusters, n_noise)


if __name__ == "__main__":
    run_clustering()
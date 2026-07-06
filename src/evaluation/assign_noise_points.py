"""
Assign HDBSCAN noise points (label -1) to their nearest cluster
using cosine distance in the 50D embedding space.

This produces a 'workshop-ready' clustering where every report is assigned
to exactly one cluster — suitable for the citizen workshop where every
report needs to belong to a discussion group.

The original hdbscan_label column is preserved as hdbscan_label_original
so the assignment is fully transparent and reversible.

Run with:
    PYTHONPATH=src uv run python src/utils/assign_noise_points.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

import os
from datetime import datetime
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M"))

# ---------------------------------------------------------------------------
# Configuration — point at the run you want to use for the workshop
# ---------------------------------------------------------------------------

RESULTS_CSV    = f"outputs/strategy2_clustering_results_{RUN_ID}.csv"
EMBEDDINGS_NPY = f"outputs/strategy2_fused_embeddings_50d_{RUN_ID}.npy"
OUTPUT_CSV     = "outputs/workshop_clusters.csv"


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_noise_to_nearest_cluster(
    results_csv: str,
    embeddings_npy: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    For each noise point (hdbscan_label == -1), find the cluster whose
    centroid is nearest in the 50D embedding space (cosine distance) and
    assign it there.

    Steps:
        1. Load embeddings and clustering results
        2. Compute cluster centroids from assigned points only
        3. For each noise point, compute cosine distance to all centroids
        4. Assign to nearest centroid
        5. Save result with both original and assigned labels

    Args:
        results_csv:    Path to strategy2_clustering_results_{RUN_ID}.csv
        embeddings_npy: Path to strategy2_fused_embeddings_50d_{RUN_ID}.npy
        output_csv:     Path to save workshop-ready cluster assignments

    Returns:
        DataFrame with hdbscan_label_assigned column added.
    """
    print(f"Loading results from {results_csv}...")
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} reports.")

    print(f"Loading embeddings from {embeddings_npy}...")
    embeddings = np.load(embeddings_npy)
    print(f"Embedding matrix shape: {embeddings.shape}\n")

    # Verify row alignment
    assert len(df) == len(embeddings), (
        f"Row count mismatch: results {len(df)} vs embeddings {len(embeddings)}"
    )

    # -- Preserve original labels -------------------------------------------
    df["hdbscan_label_original"] = df["hdbscan_label"].copy()

    n_noise    = (df["hdbscan_label"] == -1).sum()
    n_assigned = (df["hdbscan_label"] != -1).sum()
    clusters   = sorted(df[df["hdbscan_label"] != -1]["hdbscan_label"].unique())

    print(f"Noise points to assign : {n_noise} ({100 * n_noise / len(df):.1f}%)")
    print(f"Assigned points        : {n_assigned}")
    print(f"Clusters               : {clusters}\n")

    # -- Compute cluster centroids ------------------------------------------
    centroids = {}
    for cluster_id in clusters:
        mask = df["hdbscan_label"] == cluster_id
        cluster_embeddings = embeddings[mask.values]
        centroids[cluster_id] = cluster_embeddings.mean(axis=0)

    centroid_matrix = np.array([centroids[c] for c in clusters])  # (n_clusters, 50)

    # -- Assign noise points to nearest centroid ----------------------------
    noise_mask    = df["hdbscan_label"] == -1
    noise_indices = np.where(noise_mask.values)[0]
    noise_embs    = embeddings[noise_indices]  # (n_noise, 50)

    # Cosine distance between each noise point and each centroid
    distances = cosine_distances(noise_embs, centroid_matrix)  # (n_noise, n_clusters)
    nearest   = np.argmin(distances, axis=1)                   # index into clusters list

    # Apply assignments
    df["hdbscan_label_assigned"] = df["hdbscan_label"].copy()
    for i, row_idx in enumerate(noise_indices):
        assigned_cluster = clusters[nearest[i]]
        df.at[row_idx, "hdbscan_label_assigned"] = assigned_cluster
        dist = distances[i, nearest[i]]
        print(f"  Noise point {df.at[row_idx, 'photo_id'][:8]}... "
              f"→ Cluster {assigned_cluster} (distance={dist:.4f})")

    # -- Summary ------------------------------------------------------------
    print(f"\n=== Assignment Summary ===")
    print(f"  Original noise points : {n_noise}")
    print(f"  Remaining noise       : {(df['hdbscan_label_assigned'] == -1).sum()}")
    print(f"\n  Final cluster sizes (hdbscan_label_assigned):")
    for label, count in sorted(df["hdbscan_label_assigned"].value_counts().items()):
        original = (df["hdbscan_label_original"] == label).sum()
        added    = count - original
        added_str = f" (+{added} noise)" if added > 0 else ""
        print(f"    Cluster {label}: {count} reports{added_str}")

    # -- Save ---------------------------------------------------------------
    df.to_csv(output_csv, index=False)
    print(f"\nWorkshop clusters saved to {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assign_noise_to_nearest_cluster(
        results_csv    = RESULTS_CSV,
        embeddings_npy = EMBEDDINGS_NPY,
        output_csv     = OUTPUT_CSV,
    )
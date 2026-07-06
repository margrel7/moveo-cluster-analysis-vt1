import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_CLUSTER_METHOD

# ---------------------------------------------------------------------------
# Configuration defaults (can be overridden per strategy)
# ---------------------------------------------------------------------------

K_MIN = 2
K_MAX = 12
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 20

UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# UMAP reduction 
# ---------------------------------------------------------------------------

def reduce_with_umap(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D using UMAP.
    Used before HDBSCAN when 2D embeddings are not pre-computed.
    """
    print("  Reducing to 2D with UMAP...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
        metric=metric,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

def run_kmeans(
    embeddings: np.ndarray,
    normalise: bool = True,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> tuple[np.ndarray, int, float]:
    """
    Run K-means for k in [k_min, k_max] and select best k by silhouette score.

    Args:
        embeddings: Embedding matrix to cluster on.
        normalise:  Whether to L2-normalise before clustering. Set False if
                    embeddings are already normalised (e.g. Strategy 2 50D).
        k_min:      Minimum k to try.
        k_max:      Maximum k to try.

    Returns:
        (labels, best_k, best_silhouette_score)
    """
    print(f"\n--- K-means (k={k_min}..{k_max}) ---")

    emb = normalize(embeddings, norm="l2") if normalise else embeddings

    best_k = None
    best_labels = None
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(emb)
        score = silhouette_score(emb, labels, metric="cosine")
        print(f"  k={k:2d}  silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"\n  ✓ Best k={best_k}  silhouette={best_score:.4f}")
    return best_labels, best_k, best_score


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(
    embeddings: np.ndarray,
    precomputed_2d: np.ndarray | None = None,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, int, int]:
    print(f"\n--- HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}) ---")

    reduced = precomputed_2d if precomputed_2d is not None else reduce_with_umap(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=HDBSCAN_CLUSTER_METHOD,
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"  ✓ Clusters found : {n_clusters}")
    print(f"  ✓ Noise points   : {n_noise} ({100 * n_noise / len(labels):.1f}%)")

    return labels, n_clusters, n_noise


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    index_map: pd.DataFrame,
    metadata: pd.DataFrame,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    best_k: int,
    output_csv: str,
) -> pd.DataFrame:
    results = index_map[["photo_id"]].copy()
    results["kmeans_label"] = kmeans_labels
    results["hdbscan_label"] = hdbscan_labels
    results["kmeans_k"] = best_k
    results = results.merge(metadata, on="photo_id", how="left")
    results.to_csv(output_csv, index=False)
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    strategy_name: str,
    results: pd.DataFrame,
    best_k: int,
    kmeans_silhouette: float,
    n_clusters: int,
    n_noise: int,
):
    print(f"\n=== {strategy_name} — Clustering Summary ===")
    print(f"  Total reports        : {len(results)}")
    print(f"\n  K-means")
    print(f"    Best k             : {best_k}")
    print(f"    Silhouette score   : {kmeans_silhouette:.4f}")
    print(f"\n  HDBSCAN")
    print(f"    Clusters found     : {n_clusters}")
    print(f"    Noise points       : {n_noise} ({100 * n_noise / len(results):.1f}%)")

    print(f"\n  K-means cluster sizes:")
    for label, count in sorted(results["kmeans_label"].value_counts().items()):
        print(f"    Cluster {label}: {count} reports")

    print(f"\n  HDBSCAN cluster sizes:")
    for label, count in sorted(results["hdbscan_label"].value_counts().items()):
        name = "Noise" if label == -1 else f"Cluster {label}"
        print(f"    {name}: {count} reports")

    print(f"\n  Results saved to: {results}")
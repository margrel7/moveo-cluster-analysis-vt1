import os
import numpy as np
import pandas as pd
import umap

# ── Configuration ────────────────────────────────────────────────────────────
N_COMPONENTS = 50       # Target dimensionality
N_NEIGHBORS = 15        # UMAP: number of neighbours to consider
MIN_DIST = 0.1          # UMAP: minimum distance between points in reduced space
RANDOM_STATE = 42       # For reproducibility

STRATEGIES = {
    "clip_joint": {
        "embeddings": "outputs/embeddings_clip_joint.npy",
        "index": "outputs/embedding_index_clip_joint.csv",
        "output_embeddings": "outputs/umap_clip_joint.npy",
        "output_index": "outputs/umap_index_clip_joint.csv",
    },
    "vector_level": {
        "embeddings": "outputs/embeddings_vector_level.npy",
        "index": "outputs/embedding_index_vector_level.csv",
        "output_embeddings": "outputs/umap_vector_level.npy",
        "output_index": "outputs/umap_index_vector_level.csv",
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = N_COMPONENTS,
    n_neighbors: int = N_NEIGHBORS,
    min_dist: float = MIN_DIST,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def run_umap_reduction(strategy_name: str, config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    print(f"─── Strategy: {strategy_name} ───")

    # Load embeddings and index
    embeddings = np.load(config["embeddings"])
    index_df = pd.read_csv(config["index"])

    print(f"Input shape:  {embeddings.shape}")
    print(f"Applying UMAP: {embeddings.shape[1]}d → {N_COMPONENTS}d")
    print(f"  n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}, metric=cosine")

    # Apply UMAP
    reduced = reduce_embeddings(embeddings)

    print(f"Output shape: {reduced.shape}")

    # Save
    np.save(config["output_embeddings"], reduced)
    index_df.to_csv(config["output_index"], index=False)

    print(f"Saved to {config['output_embeddings']}")
    print()

    return reduced, index_df


def run_all_umap():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("UMAP Dimensionality Reduction")
    print(f"Target dimensions: {N_COMPONENTS}")
    print("=" * 60)
    print()

    results = {}
    for strategy_name, config in STRATEGIES.items():
        reduced, index_df = run_umap_reduction(strategy_name, config)
        results[strategy_name] = {
            "embeddings": reduced,
            "index": index_df,
        }

    print("=" * 60)
    print("UMAP Reduction Complete")
    print("=" * 60)
    for strategy_name, config in STRATEGIES.items():
        print(f"{strategy_name}:")
        print(f"  Embeddings: {config['output_embeddings']}")
        print(f"  Index:      {config['output_index']}")

    return results


if __name__ == "__main__":
    run_all_umap()
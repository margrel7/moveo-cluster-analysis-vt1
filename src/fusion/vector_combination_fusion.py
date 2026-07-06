import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import umap

from src.config import (
    S2_IMAGE_EMBEDDINGS_NPY,
    S2_IMAGE_EMBEDDINGS_CSV,
    S2_TEXT_EMBEDDINGS_NPY,
    S2_FUSED_NPY,
    S2_FUSED_50D_NPY,
    S2_FUSED_2D_NPY,
    S2_FUSED_CSV,
    IMAGE_WEIGHT,
    TEXT_WEIGHT,
    RUN_ID,
    print_config,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UMAP_N_NEIGHBORS  = 15
UMAP_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse_embeddings(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
) -> np.ndarray:
    """
    Weighted late fusion of image and text embeddings.

    Pipeline:
        1. L2-normalise each modality independently
        2. Scale each by its weight (IMAGE_WEIGHT, TEXT_WEIGHT from config)
        3. Concatenate into a single (N, 960+768) = (N, 1728) vector
        4. L2-normalise the concatenated vector row-wise

    Weights are set via IMAGE_WEIGHT / TEXT_WEIGHT environment variables.
    Default: equal weighting (0.5/0.5).
    Recommended for this dataset: 0.3/0.7 — text carries richer semantic
    signal than blurry phone photos.

    Args:
        image_emb: (N, 960) L2-normalised image embedding matrix
        text_emb:  (N, 768) L2-normalised text embedding matrix

    Returns:
        (N, 1728) L2-normalised fused embedding matrix
    """
    assert image_emb.shape[0] == text_emb.shape[0], (
        f"Row count mismatch: image {image_emb.shape[0]} vs text {text_emb.shape[0]}. "
        "Ensure both embedding scripts ran on the same input file and RUN_ID."
    )

    img_norm = normalize(image_emb, norm="l2") * IMAGE_WEIGHT
    txt_norm = normalize(text_emb,  norm="l2") * TEXT_WEIGHT

    print(f"  Fusion weights : image={IMAGE_WEIGHT}, text={TEXT_WEIGHT}")

    concatenated = np.concatenate([img_norm, txt_norm], axis=1)
    fused = normalize(concatenated, norm="l2")

    return fused


def reduce_umap(embeddings: np.ndarray, n_components: int, metric: str = "cosine") -> np.ndarray:
    """
    Apply UMAP dimensionality reduction.

    Used in two contexts:
        n_components=50 → for K-means (preserves more variance than 2D)
        n_components=2  → for HDBSCAN (density estimation in 2D)
    """
    print(f"  Reducing {embeddings.shape} → {n_components}D with UMAP...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        metric=metric,
        random_state=UMAP_RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_fusion(
    image_npy: str,
    text_npy: str,
    image_csv: str,
    output_npy: str,
    output_npy_50d: str,
    output_npy_2d: str,
    output_csv: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load image and text embeddings, fuse them, and save outputs.

    Saves:
        output_npy      — (N, 1728) fused embedding matrix (raw, for reference)
        output_npy_50d  — (N, 50) UMAP-reduced (for K-means)
        output_npy_2d   — (N, 2) UMAP-reduced (for HDBSCAN)
        output_csv      — metadata index map
    """
    print("Loading image embeddings...")
    image_emb = np.load(image_npy)
    print(f"  Image matrix : {image_emb.shape}")

    print("Loading text embeddings...")
    text_emb = np.load(text_npy)
    print(f"  Text matrix  : {text_emb.shape}")

    # -- Fuse ---------------------------------------------------------------
    print("\nFusing embeddings (L2-normalise → concatenate → L2-normalise)...")
    fused = fuse_embeddings(image_emb, text_emb)
    print(f"  Fused matrix : {fused.shape}")

    # -- UMAP reductions ----------------------------------------------------
    print("\nGenerating UMAP reductions...")

    reduced_50d = reduce_umap(fused, n_components=50)
    np.save(output_npy_50d, reduced_50d.astype(np.float32))
    print(f"  50D saved to : {output_npy_50d}")

    reduced_2d = reduce_umap(fused, n_components=2)
    np.save(output_npy_2d, reduced_2d.astype(np.float32))
    print(f"  2D  saved to : {output_npy_2d}")

    # -- Save raw fused matrix and metadata ---------------------------------
    np.save(output_npy, fused.astype(np.float32))

    meta_df = pd.read_csv(image_csv)
    meta_df.to_csv(output_csv, index=False)

    print(f"\n=== Strategy 2 — Fusion Summary ({RUN_ID}) ===")
    print(f"  Image dims     : {image_emb.shape[1]}")
    print(f"  Text dims      : {text_emb.shape[1]}")
    print(f"  Fused dims     : {fused.shape[1]}")
    print(f"  UMAP 50D shape : {reduced_50d.shape}")
    print(f"  UMAP 2D shape  : {reduced_2d.shape}")
    print(f"  Raw fused      : {output_npy}")
    print(f"  Metadata       : {output_csv}")

    return fused, meta_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_config("strategy2_fusion.py")
    run_fusion(
        image_npy      = S2_IMAGE_EMBEDDINGS_NPY,
        text_npy       = S2_TEXT_EMBEDDINGS_NPY,
        image_csv      = S2_IMAGE_EMBEDDINGS_CSV,
        output_npy     = S2_FUSED_NPY,
        output_npy_50d = S2_FUSED_50D_NPY,
        output_npy_2d  = S2_FUSED_2D_NPY,
        output_csv     = S2_FUSED_CSV,
    )
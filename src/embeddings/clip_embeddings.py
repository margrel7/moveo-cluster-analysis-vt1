"""
Strategy 1b — Multilingual CLIP Joint Embedding (xlm-roberta-base-ViT-B-32)
Drop-in replacement for Strategy 1 with a multilingual text encoder.
Run with: RUN_ID=strategy1b uv run python src/embeddings/strategy1b_clip_embeddings.py
"""

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from src.config import (
    PREPROCESSED_LLAVA_CSV,
    S1_EMBEDDINGS_CSV,
    S1_EMBEDDINGS_NPY,
    S1_INDEX_MAP_CSV,
    RUN_ID,
    print_config,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Multilingual CLIP — XLM-RoBERTa text encoder + original ViT-B/32 image encoder
# Supports 100 languages including German and English natively.
# Switch comments below to revert to original English-only CLIP:
#MODEL_NAME = "xlm-roberta-base-ViT-B-32"
#PRETRAINED = "laion5b_s13b_b90k"
#MODEL_NAME = "ViT-B-32"     # original English-only CLIP
#PRETRAINED = "openai"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    """
    Load CLIP via open_clip and move to the best available device.
    On Apple Silicon, MPS gives a meaningful speedup over CPU.
    """
    print(f"Loading {MODEL_NAME} ({PRETRAINED}) via open_clip...")
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise a 1D numpy vector. Returns a zero vector if norm is 0."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def embed_image(
    filepath: str,
    model,
    preprocess,
    device: torch.device,
) -> np.ndarray | None:
    """
    Generate a CLIP image embedding from a local file.
    Returns an L2-normalised 512-dim vector, or None on failure.
    """
    try:
        image = preprocess(Image.open(filepath).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return l2_normalise(features.squeeze().cpu().numpy())
    except Exception as e:
        print(f"  [ERROR] Image embedding failed for {filepath}: {e}")
        return None


def embed_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
) -> np.ndarray | None:
    """
    Generate a CLIP text embedding from a string.
    Returns an L2-normalised 512-dim vector, or None on failure.
    """
    try:
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
        return l2_normalise(features.squeeze().cpu().numpy())
    except Exception as e:
        print(f"  [ERROR] Text embedding failed: {e}")
        return None


def combine_embeddings(
    image_emb: np.ndarray | None,
    text_emb: np.ndarray | None,
) -> tuple[np.ndarray | None, str]:
    """
    Combine image and text embeddings into a single joint embedding.
    Both available → average + L2-normalise.
    Fallbacks retained for robustness — should not trigger after LLaVA imputation.
    """
    if image_emb is not None and text_emb is not None:
        return l2_normalise((image_emb + text_emb) / 2.0), "image+text"
    elif image_emb is not None:
        return image_emb, "image_only"
    elif text_emb is not None:
        return text_emb, "text_only"
    else:
        return None, "none"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_clip_embeddings(input_path: str, output_csv: str, output_npy: str) -> pd.DataFrame:
    """
    Generate CLIP joint embeddings for all reports.

    For each row:
        1. Embed the image via open_clip encode_image()
        2. Embed the preprocessed_text via open_clip encode_text()
        3. Average the two L2-normalised vectors and re-normalise

    Outputs:
        output_csv  — metadata + modality_used per row
        output_npy  — (N, 512) float32 embedding matrix
        index map   — derived from output_npy path, array row → photo_id
    """
    model, preprocess, tokenizer, device = load_model()

    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.\n")

    embeddings = []
    metadata   = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        photo_id = row["photo_id"]
        print(f"[{idx}/{len(df)}] {photo_id[:8]}...")

        image_emb = None
        if pd.notna(row.get("local_image_path")):
            image_emb = embed_image(row["local_image_path"], model, preprocess, device)

        text_emb = None
        if pd.notna(row.get("preprocessed_text")) and str(row["preprocessed_text"]).strip():
            text_emb = embed_text(row["preprocessed_text"], model, tokenizer, device)

        combined, modality = combine_embeddings(image_emb, text_emb)

        if combined is None:
            print(f"  [SKIP] No image or text available — excluded from clustering.")

        metadata.append({
            "photo_id":          photo_id,
            "username":          row.get("username"),
            "category":          row.get("category"),
            "lat":               row.get("lat"),
            "long":              row.get("long"),
            "is_llava_imputed":  row.get("is_llava_imputed", False),
            "modality_used":     modality,
            "embedding_available": combined is not None,
        })

        if combined is not None:
            embeddings.append({"photo_id": photo_id, "embedding": combined})

    # -- Build outputs ------------------------------------------------------
    meta_df    = pd.DataFrame(metadata)
    emb_df     = pd.DataFrame([e["photo_id"] for e in embeddings], columns=["photo_id"])
    emb_matrix = np.array([e["embedding"] for e in embeddings], dtype=np.float32)

    np.save(output_npy, emb_matrix)
    emb_df.to_csv(S1_INDEX_MAP_CSV, index=True, index_label="array_index")
    meta_df.to_csv(output_csv, index=False)

    print(f"\n=== Strategy 1 — {MODEL_NAME} ({RUN_ID}) ===")
    print(f"  Total reports      : {len(df)}")
    for mod, count in meta_df["modality_used"].value_counts().items():
        print(f"  {mod:<20}: {count}")
    n_excluded = (meta_df["modality_used"] == "none").sum()
    if n_excluded:
        print(f"  [WARN] {n_excluded} row(s) excluded.")
    print(f"\n  Embedding matrix shape : {emb_matrix.shape}")
    print(f"  Saved to               : {output_npy}")
    print(f"  Index map saved to     : {S1_INDEX_MAP_CSV}")
    print(f"  Metadata saved to      : {output_csv}")

    return meta_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_config("strategy1b_clip_embeddings.py")
    run_clip_embeddings(
        input_path=PREPROCESSED_LLAVA_CSV,
        output_csv=S1_EMBEDDINGS_CSV,
        output_npy=S1_EMBEDDINGS_NPY,
    )
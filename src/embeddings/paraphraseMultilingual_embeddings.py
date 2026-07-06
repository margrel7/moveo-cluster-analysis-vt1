import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    PREPROCESSED_LLAVA_CSV,
    S2_TEXT_EMBEDDINGS_NPY,
    S2_TEXT_EMBEDDINGS_CSV,
    S3_ENRICHED_CSV,
    S3_EMBEDDINGS_NPY,
    S3_EMBEDDINGS_CSV,
    RUN_ID,
    print_config,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

STRATEGY_CONFIGS = {
    "strategy2": {
        "input_path":  PREPROCESSED_LLAVA_CSV,
        "text_column": "preprocessed_text",
        "output_npy":  S2_TEXT_EMBEDDINGS_NPY,
        "output_csv":  S2_TEXT_EMBEDDINGS_CSV,
        "label":       f"Strategy 2 — paraphrase-multilingual (raw user text) ({RUN_ID})",
    },
    "strategy3": {
        "input_path":  S3_ENRICHED_CSV,
        "text_column": "enriched_text",
        "output_npy":  S3_EMBEDDINGS_NPY,
        "output_csv":  S3_EMBEDDINGS_CSV,
        "label":       f"Strategy 3 — paraphrase-multilingual (Gemma4 enriched text) ({RUN_ID})",
    },
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_text_embeddings(
    input_path: str,
    text_column: str,
    output_npy: str,
    output_csv: str,
    label: str,
) -> pd.DataFrame:
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.")
    print(f"Text column: '{text_column}'\n")

    embeddings = []
    metadata   = []
    n_failed   = 0

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        photo_id = row["photo_id"]
        text     = row.get(text_column)

        print(f"[{idx}/{len(df)}] {photo_id[:8]}...")

        if pd.isna(text) or not str(text).strip():
            print(f"  [SKIP] No text in column '{text_column}'.")
            n_failed += 1
            continue

        try:
            vec = model.encode(str(text), normalize_embeddings=True)
            embeddings.append({"photo_id": photo_id, "embedding": vec})
            metadata.append({
                "photo_id":          photo_id,
                "username":          row.get("username"),
                "category":          row.get("category"),
                "lat":               row.get("lat"),
                "long":              row.get("long"),
                "is_llava_imputed":  row.get("is_llava_imputed", False),
                text_column:         text,
            })
        except Exception as e:
            print(f"  [ERROR] Embedding failed: {e}")
            n_failed += 1

    emb_matrix = np.array([e["embedding"] for e in embeddings], dtype=np.float32)
    meta_df    = pd.DataFrame(metadata)

    np.save(output_npy, emb_matrix)
    meta_df.to_csv(output_csv, index=True, index_label="array_index")

    print(f"\n=== {label} ===")
    print(f"  Embedded       : {len(embeddings)}")
    print(f"  Failed/skipped : {n_failed}")
    print(f"  Matrix shape   : {emb_matrix.shape}")
    print(f"  Saved to       : {output_npy}")

    return meta_df


# ---------------------------------------------------------------------------
# Entry point — pass 'strategy2' or 'strategy3' as argument
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "strategy2"

    if strategy not in STRATEGY_CONFIGS:
        print(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_CONFIGS.keys())}")
        sys.exit(1)

    print_config(f"text_embeddings.py ({strategy})")
    cfg = STRATEGY_CONFIGS[strategy]
    run_text_embeddings(
        input_path=cfg["input_path"],
        text_column=cfg["text_column"],
        output_npy=cfg["output_npy"],
        output_csv=cfg["output_csv"],
        label=cfg["label"],
    )
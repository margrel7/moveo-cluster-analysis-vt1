import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Run ID — set via environment variable or defaults to timestamp
# ---------------------------------------------------------------------------

RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M"))

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUTS_DIR = "outputs"


# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------

def output_path(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    return os.path.join(OUTPUTS_DIR, f"{base}_{RUN_ID}{ext}")


# ---------------------------------------------------------------------------
# Pre-built paths for each pipeline stage
# Imported directly by scripts — no path string duplication anywhere.
# ---------------------------------------------------------------------------

PREPROCESSED_LLAVA_CSV = os.environ.get("PREPROCESSED_LLAVA_CSV", "outputs/preprocessed_llava.csv")
PREPROCESSED_IMAGES_CSV = "outputs/preprocessed_images.csv"
RAW_DATA_CSV = os.environ.get("RAW_DATA_CSV", "data/raw/raw_data.csv")

# Strategy 1 / 1b
S1_EMBEDDINGS_NPY        = output_path("strategy1_embeddings.npy")
S1_EMBEDDINGS_CSV        = output_path("strategy1_embeddings.csv")
S1_INDEX_MAP_CSV         = output_path("strategy1_embeddings_index_map.npy").replace(".npy", ".csv")
S1_CLUSTERING_CSV        = output_path("strategy1_clustering_results.csv")

# Strategy 2
S2_IMAGE_EMBEDDINGS_NPY  = output_path("strategy2_image_embeddings.npy")
S2_IMAGE_EMBEDDINGS_CSV  = output_path("strategy2_image_embeddings.csv")
S2_TEXT_EMBEDDINGS_NPY   = output_path("strategy2_text_embeddings.npy")
S2_TEXT_EMBEDDINGS_CSV   = output_path("strategy2_text_embeddings.csv")
S2_FUSED_NPY             = output_path("strategy2_fused_embeddings.npy")
S2_FUSED_50D_NPY         = output_path("strategy2_fused_embeddings_50d.npy")
S2_FUSED_2D_NPY          = output_path("strategy2_fused_embeddings_2d.npy")
S2_FUSED_CSV             = output_path("strategy2_fused_embeddings.csv")
S2_CLUSTERING_CSV        = output_path("strategy2_clustering_results.csv")

# Strategy 3
S3_ENRICHED_CSV          = output_path("strategy3_enriched.csv")
S3_ENRICHED_BACKUP_CSV   = output_path("strategy3_enriched_raw_backup.csv")
S3_EMBEDDINGS_NPY        = output_path("strategy3_embeddings.npy")
S3_EMBEDDINGS_CSV        = output_path("strategy3_embeddings.csv")
S3_CLUSTERING_CSV        = output_path("strategy3_clustering_results.csv")


# ---------------------------------------------------------------------------
# Strategy 2 — fusion weights
# ---------------------------------------------------------------------------

IMAGE_WEIGHT = float(os.environ.get("IMAGE_WEIGHT", "0.5"))
TEXT_WEIGHT  = float(os.environ.get("TEXT_WEIGHT",  "0.5"))

# ---------------------------------------------------------------------------
# HDBSCAN tuning parameters
# ---------------------------------------------------------------------------

HDBSCAN_MIN_CLUSTER_SIZE = int(os.environ.get("HDBSCAN_MIN_CLUSTER_SIZE", "10"))
HDBSCAN_MIN_SAMPLES      = int(os.environ.get("HDBSCAN_MIN_SAMPLES", "5"))
HDBSCAN_CLUSTER_METHOD   = os.environ.get("HDBSCAN_CLUSTER_METHOD", "eom")  # "eom" or "leaf"


# ---------------------------------------------------------------------------
# Pretty-print current config (called at the start of each script)
# ---------------------------------------------------------------------------

def print_config(script_name: str):
    print(f"{'─' * 50}")
    print(f"  Script       : {script_name}")
    print(f"  Run ID       : {RUN_ID}")
    print(f"  Outputs      : {OUTPUTS_DIR}/")
    print(f"  Image weight : {IMAGE_WEIGHT}")
    print(f"  Text weight  : {TEXT_WEIGHT}")
    print(f"  HDBSCAN      : min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}, method={HDBSCAN_CLUSTER_METHOD}")
    print(f"{'─' * 50}\n")
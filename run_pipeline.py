#!/usr/bin/env python3
"""
Full multimodal clustering pipeline using the Strategy 2b configuration.

This script chains all pipeline stages from preprocessing through to clustering
and evaluation. It was developed and optimised for the MOVEO Winterthur dataset
(176 citizen reports, bilingual DE/EN, 2026). The default parameters reflect the
best-performing configuration for that dataset. They are not guaranteed to be
optimal for other datasets and should be treated as a starting point.

Stages:
    1. Text preprocessing
    2. Image download + quality screening
    3. LLaVA imputation (for text-missing reports)
    4. MobileNetV3 image embeddings
    5. Multilingual text embeddings
    6. Weighted vector-level fusion (default: image 0.3 / text 0.7)
    7. K-means clustering (k=2..12, silhouette selection)
    8. HDBSCAN clustering (on UMAP 2D)
    9. Evaluation (silhouette, AMI, ARI, topic distribution)

Parameters that likely need adjustment for a new dataset:
    - IMAGE_WEIGHT / TEXT_WEIGHT: depends on relative quality of images vs text
    - HDBSCAN_MIN_CLUSTER_SIZE: rule of thumb sqrt(n), here sqrt(176) ≈ 13
    - RAW_DATA_CSV: must match your data schema (see expected columns below)
    - LLaVA prompt in preprocessing/llava_imputation.py: references Winterthur
      and Swiss public transport — adapt to your domain

Expected input CSV columns:
    photo_id, username, category, topic, description,
    how_resolved_strengthened, lat, long, url_original, date

Usage:
    # Run with defaults (S2b parameters)
    PYTHONPATH=src python run_pipeline.py

    # Override parameters via environment variables
    RUN_ID=my_run IMAGE_WEIGHT=0.4 TEXT_WEIGHT=0.6 python run_pipeline.py

    # Skip preprocessing if already done
    SKIP_PREPROCESSING=1 RUN_ID=my_run python run_pipeline.py

    # Skip embeddings if already computed
    SKIP_EMBEDDINGS=1 RUN_ID=my_run python run_pipeline.py
"""

import os
import sys
import subprocess
from datetime import datetime
import sys

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M"))

# Input data
RAW_DATA_CSV = os.environ.get("RAW_DATA_CSV", "data/raw/raw_data.csv")

# Fusion weights — optimised for MOVEO Winterthur dataset
# Text carries more semantic signal than images for this dataset.
# Adjust based on the relative quality of images vs text in your data.
IMAGE_WEIGHT = os.environ.get("IMAGE_WEIGHT", "0.3")
TEXT_WEIGHT  = os.environ.get("TEXT_WEIGHT",  "0.7")

# HDBSCAN parameters — optimised for n=176
# min_cluster_size: increase proportionally for larger datasets (sqrt(n) as guide)
# leaf vs eom: leaf gives finer clusters, eom is more conservative
HDBSCAN_MIN_CLUSTER_SIZE = os.environ.get("HDBSCAN_MIN_CLUSTER_SIZE", "7")
HDBSCAN_MIN_SAMPLES      = os.environ.get("HDBSCAN_MIN_SAMPLES",      "3")
HDBSCAN_CLUSTER_METHOD   = os.environ.get("HDBSCAN_CLUSTER_METHOD",   "leaf")

# Skip flags — set to "1" to skip stages already completed
SKIP_PREPROCESSING = os.environ.get("SKIP_PREPROCESSING", "0") == "1"
SKIP_EMBEDDINGS    = os.environ.get("SKIP_EMBEDDINGS",    "0") == "1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, stage: str):
    print(f"\n{'─' * 60}")
    print(f"  STAGE: {stage}")
    print(f"  CMD:   {cmd}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, shell=True, env={**os.environ,
        "PYTHONPATH": "src",
        "RUN_ID": RUN_ID,
        "RAW_DATA_CSV": RAW_DATA_CSV,
        "IMAGE_WEIGHT": IMAGE_WEIGHT,
        "TEXT_WEIGHT": TEXT_WEIGHT,
        "HDBSCAN_MIN_CLUSTER_SIZE": HDBSCAN_MIN_CLUSTER_SIZE,
        "HDBSCAN_MIN_SAMPLES": HDBSCAN_MIN_SAMPLES,
        "HDBSCAN_CLUSTER_METHOD": HDBSCAN_CLUSTER_METHOD,
    })
    if result.returncode != 0:
        print(f"\n[FAILED] Stage '{stage}' exited with code {result.returncode}.")
        print("Fix the error above and re-run with SKIP_PREPROCESSING=1 and/or SKIP_EMBEDDINGS=1 to resume.")
        sys.exit(result.returncode)


def header():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         MOVEO Multimodal Clustering Pipeline                 ║
║         Strategy 2b — Weighted Fusion + Tuned HDBSCAN        ║
╚══════════════════════════════════════════════════════════════╝

  RUN_ID               : {RUN_ID}
  RAW_DATA_CSV         : {RAW_DATA_CSV}
  IMAGE_WEIGHT         : {IMAGE_WEIGHT}
  TEXT_WEIGHT          : {TEXT_WEIGHT}
  HDBSCAN parameters   : min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}, method={HDBSCAN_CLUSTER_METHOD}
  SKIP_PREPROCESSING   : {SKIP_PREPROCESSING}
  SKIP_EMBEDDINGS      : {SKIP_EMBEDDINGS}

  Note: Default parameters were optimised for the MOVEO Winterthur
  dataset (n=176, bilingual DE/EN). Adjust for other datasets.
""")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    header()

    # ── Stage 1-3: Preprocessing ──────────────────────────────────────────
    if not SKIP_PREPROCESSING:
        run(f"{PYTHON} src/preprocessing/text_preprocessing.py",
            "1/6 — Text preprocessing")

        run(f"{PYTHON} src/preprocessing/image_preprocessing.py",
            "2/6 — Image download + quality screening")

        run(f"{PYTHON} src/preprocessing/llava_imputation.py",
            "3/6 — LLaVA imputation for text-missing reports")
    else:
        print("\n[SKIP] Preprocessing skipped (SKIP_PREPROCESSING=1)")
        print("       Using existing outputs/preprocessed_llava.csv\n")

    # ── Stage 4-5: Embeddings ─────────────────────────────────────────────
    if not SKIP_EMBEDDINGS:
        run(f"{PYTHON} src/embeddings/mobileNetV3_embeddings.py",
            "4/6 — MobileNetV3 image embeddings")

        run(f"{PYTHON} src/embeddings/paraphraseMultilingual_embeddings.py strategy2",
            "5/6 — Multilingual text embeddings")
    else:
        print("\n[SKIP] Embeddings skipped (SKIP_EMBEDDINGS=1)")
        print(f"       Using existing embedding files for RUN_ID={RUN_ID}\n")

    # ── Stage 6: Fusion ───────────────────────────────────────────────────
    run(f"{PYTHON} src/fusion/vector_combination_fusion.py",
        "6/9 — Weighted vector-level fusion + UMAP reduction")

    # ── Stage 7-8: Clustering ─────────────────────────────────────────────
    run(f"{PYTHON} src/clustering/strategy2_clustering.py",
        "7/9 — K-means + HDBSCAN clustering")

    # ── Stage 9: Evaluation ───────────────────────────────────────────────
    run(f"{PYTHON} src/evaluation/strategy2_evaluation.py",
        "8/9 — Evaluation (silhouette, AMI, ARI, topic distribution)")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Pipeline complete                                           ║
╚══════════════════════════════════════════════════════════════╝

  Results saved to: outputs/*_{RUN_ID}.*

  Next steps:
    1. Review clustering results in outputs/strategy2_clustering_results_{RUN_ID}.csv
    2. Run workshop generation script:
       python3 run_pipeline.py --run-id {RUN_ID}
""")


if __name__ == "__main__":
    main()
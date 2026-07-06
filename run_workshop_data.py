#!/usr/bin/env python3
"""
Generate all workshop outputs from a completed clustering run.

Chains three steps:
    1. Assign HDBSCAN noise points to nearest cluster (cosine distance)
    2. Generate printable cluster cards (HTML, A4 portrait)
    3. Generate interactive map (Folium, with sentiment filter)

Requires a completed clustering run (run_pipeline.py or equivalent).
The RUN_ID must match the clustering run whose results you want to use.

Usage:
    # Use default RUN_ID from environment
    RUN_ID=s2_v2_weighted PYTHONPATH=src python run_workshop_data.py

    # Or pass explicitly
    python run_workshop_data.py --run-id s2_v2_weighted

    # Skip noise assignment if already done
    SKIP_NOISE_ASSIGNMENT=1 RUN_ID=s2_v2_weighted python run_workshop_data.py

Outputs:
    outputs/workshop_clusters.csv                  — all reports with cluster labels
    outputs/workshop_cluster_cards.html            — printable A4 cluster cards (portrait)
    outputs/workshop_map.html                      — interactive Folium map

Print settings for cluster cards:
    Open workshop_cluster_cards.html in Chrome or Safari
    Print → A4 portrait, no margins, scale to fit page
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate workshop outputs from a clustering run."
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M")),
        help="RUN_ID of the clustering run to use (default: $RUN_ID env var)"
    )
    parser.add_argument(
        "--skip-noise-assignment",
        action="store_true",
        default=os.environ.get("SKIP_NOISE_ASSIGNMENT", "0") == "1",
        help="Skip noise assignment if workshop_clusters.csv already exists"
    )
    return parser.parse_args()


def run(cmd: str, stage: str, env: dict):
    print(f"\n{'─' * 60}")
    print(f"  STAGE: {stage}")
    print(f"  CMD:   {cmd}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"\n[FAILED] Stage '{stage}' exited with code {result.returncode}.")
        sys.exit(result.returncode)


def check_inputs(run_id: str):
    required = [
        f"outputs/strategy2_clustering_results_{run_id}.csv",
        f"outputs/strategy2_fused_embeddings_50d_{run_id}.npy",
        "outputs/preprocessed_llava.csv",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n[ERROR] Required input files not found:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nMake sure you have run the clustering pipeline with RUN_ID={run_id}")
        print("and that the output files exist in the outputs/ directory.")
        sys.exit(1)


def header(run_id: str, skip_noise: bool):
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         MOVEO Workshop Generation                            ║
╚══════════════════════════════════════════════════════════════╝

  RUN_ID                 : {run_id}
  SKIP_NOISE_ASSIGNMENT  : {skip_noise}

  Outputs:
    outputs/workshop_clusters.csv
    outputs/workshop_cluster_cards.html
    outputs/workshop_map.html
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_id = args.run_id
    skip_noise = args.skip_noise_assignment

    header(run_id, skip_noise)
    check_inputs(run_id)

    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "RUN_ID": run_id,
    }

    # ── Step 1: Noise assignment ───────────────────────────────────────────
    if not skip_noise:
        run(f"{PYTHON} src/evaluation/assign_noise_points.py",
            "1/3 — Assign noise points to nearest cluster", env)
    else:
        if not os.path.exists("outputs/workshop_clusters.csv"):
            print("[ERROR] SKIP_NOISE_ASSIGNMENT=1 but outputs/workshop_clusters.csv not found.")
            print("        Run without --skip-noise-assignment first.")
            sys.exit(1)
        print("\n[SKIP] Noise assignment skipped — using existing workshop_clusters.csv\n")

    # ── Step 2: Cluster cards ─────────────────────────────────────────────
    run(f"{PYTHON} src/cluster_cards.py",
        "2/3 — Generate printable cluster cards (HTML, portrait)", env)

    # ── Step 3: Interactive map ───────────────────────────────────────────
    run(f"{PYTHON} src/generate_map_en.py",
        "3/3 — Generate interactive map (Folium)", env)

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Workshop outputs ready                                      ║
╚══════════════════════════════════════════════════════════════╝

  outputs/workshop_clusters.csv
    → All 176 reports with assigned cluster labels
    → Use for further analysis or export

  outputs/workshop_cluster_cards.html
    → Simple portrait cards (photo + quote per example)
    → Print → A4 portrait, no margins

  outputs/workshop_map.html
    → Open in any browser
    → Interactive map with sentiment filter and cluster toggles
    → Fully self-contained (images embedded as base64)

  Note: Labels are loaded from outputs/cluster_labels.csv.
  Run run_cluster_labeling.py first if that file does not exist.
""")


if __name__ == "__main__":
    main()
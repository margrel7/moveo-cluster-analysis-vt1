#!/usr/bin/env python3
"""
Generate LLM-suggested cluster labels, then pause for human review.

This script is intentionally a two-step process:
    Step 1: Run generate_cluster_labels.py → outputs/cluster_labels.csv
    Step 2: PAUSE — human reviews and edits the CSV
    Step 3: Human confirms → pipeline continues

The pause exists because automated LLM labels are a starting point, not
a final output. Labels that are too generic, repetitive, or inaccurate
will appear in the workshop cards and map. Human review is mandatory.

Usage:
    RUN_ID=s2b_verify PYTHONPATH=src .venv/bin/python3 run_cluster_labeling.py

    # Skip label generation if cluster_labels.csv already exists
    SKIP_GENERATION=1 .venv/bin/python3 run_cluster_labeling.py

Requires:
    - outputs/workshop_clusters.csv (from run_workshop_data.py step 1)
    - outputs/strategy2_fused_embeddings_50d_{RUN_ID}.npy
    - Ollama running locally with gemma3:4b pulled

Output:
    outputs/cluster_labels.csv — review and edit this before continuing
"""

import os
import sys
import subprocess
from datetime import datetime

PYTHON = sys.executable

RUN_ID            = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M"))
SKIP_GENERATION   = os.environ.get("SKIP_GENERATION", "0") == "1"
LABELS_CSV        = "outputs/cluster_labels.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, stage: str):
    print(f"\n{'─' * 60}")
    print(f"  STAGE: {stage}")
    print(f"  CMD:   {cmd}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, shell=True, env={
        **os.environ,
        "PYTHONPATH": "src",
        "RUN_ID": RUN_ID,
    })
    if result.returncode != 0:
        print(f"\n[FAILED] Stage '{stage}' exited with code {result.returncode}.")
        sys.exit(result.returncode)


def print_csv_preview():
    """Print the generated labels so the user can see them without opening the file."""
    try:
        import pandas as pd
        df = pd.read_csv(LABELS_CSV)
        print("\nGenerated labels:")
        print(f"{'─' * 60}")
        for _, row in df.iterrows():
            print(f"  Cluster {int(row['cluster_id']):>2}  |  {str(row['label_de']):<35}  |  {str(row['label_en'])}")
        print(f"{'─' * 60}")
    except Exception as e:
        print(f"[WARN] Could not preview CSV: {e}")


def wait_for_review():
    """Pause and wait for the user to confirm they have reviewed the labels."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HUMAN REVIEW REQUIRED                                       ║
╚══════════════════════════════════════════════════════════════╝

  Open and edit the label file:
    {LABELS_CSV}

  Columns to edit:
    label_de  — German label (3-5 words, specific)
    label_en  — English label (3-5 words, specific)

  Common issues to fix:
    - Labels containing "Winterthur" → remove, it is redundant
    - Generic labels like "Erfahrungen" or "Feedback" → replace
    - Two clusters with similar labels → differentiate them
    - Labels longer than 5 words → shorten

  The llm_raw column shows the full LLM output for reference.
  Do NOT edit cluster_id, n_total, n_works_well, n_problem columns.
""")

    while True:
        answer = input("  Have you reviewed and edited the labels? (yes/skip/quit): ").strip().lower()
        if answer in ("yes", "y"):
            print("\n  Labels confirmed. Continuing...\n")
            break
        elif answer in ("skip", "s"):
            print("\n  [WARN] Skipping review — labels will be used as generated.\n")
            break
        elif answer in ("quit", "q"):
            print("\n  Exiting. Re-run after editing the CSV.\n")
            sys.exit(0)
        else:
            print("  Please type 'yes', 'skip', or 'quit'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         MOVEO Cluster Labeling                               ║
╚══════════════════════════════════════════════════════════════╝

  RUN_ID           : {RUN_ID}
  SKIP_GENERATION  : {SKIP_GENERATION}
  Output           : {LABELS_CSV}
""")

    # ── Step 1: Generate labels ───────────────────────────────────────────
    if not SKIP_GENERATION:
        print("  Requires Ollama running locally with gemma3:4b.")
        print("  This takes approximately 2-3 minutes for 9 clusters.\n")
        run(f"{PYTHON} src/cluster_labeling.py",
            "1/1 — Generate cluster labels via Gemma3:4b")
    else:
        if not os.path.exists(LABELS_CSV):
            print(f"[ERROR] SKIP_GENERATION=1 but {LABELS_CSV} not found.")
            print("        Run without SKIP_GENERATION=1 first.")
            sys.exit(1)
        print(f"\n[SKIP] Label generation skipped — using existing {LABELS_CSV}\n")

    # ── Step 2: Preview and review ────────────────────────────────────────
    print_csv_preview()
    wait_for_review()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Labels ready                                                ║
╚══════════════════════════════════════════════════════════════╝

  {LABELS_CSV} will be used automatically by:
    - generate_cluster_cards.py
    - generate_cluster_cards_with_map.py
    - generate_workshop_map.py

  Next step:
    RUN_ID={RUN_ID} .venv/bin/python3 run_workshop_data.py
""")


if __name__ == "__main__":
    main()
from evaluation_utils import run_evaluation
from src.config import S1_CLUSTERING_CSV, S1_EMBEDDINGS_NPY, RAW_DATA_CSV, RUN_ID, print_config

if __name__ == "__main__":
    print_config("strategy1_evaluation.py")
    run_evaluation(
        strategy_name  = f"Strategy 1 — CLIP Joint Embedding ({RUN_ID})",
        results_csv    = S1_CLUSTERING_CSV,
        embeddings_npy = S1_EMBEDDINGS_NPY,
        raw_data_csv   = RAW_DATA_CSV,
    )
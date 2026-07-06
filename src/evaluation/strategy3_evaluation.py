from evaluation_utils import run_evaluation
from src.config import S3_CLUSTERING_CSV, S3_EMBEDDINGS_NPY, RAW_DATA_CSV, RUN_ID, print_config

if __name__ == "__main__":
    print_config("strategy3_evaluation.py")
    run_evaluation(
        strategy_name  = f"Strategy 3 — LLM Enrichment (Gemma4 + paraphrase-multilingual) ({RUN_ID})",
        results_csv    = S3_CLUSTERING_CSV,
        embeddings_npy = S3_EMBEDDINGS_NPY,
        raw_data_csv   = RAW_DATA_CSV,
    )
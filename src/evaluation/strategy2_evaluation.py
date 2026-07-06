from evaluation_utils import run_evaluation
from src.config import S2_CLUSTERING_CSV, S2_FUSED_50D_NPY, RAW_DATA_CSV, RUN_ID, print_config

if __name__ == "__main__":
    print_config("strategy2_evaluation.py")
    run_evaluation(
        strategy_name  = f"Strategy 2 — MobileNetV3 + paraphrase-multilingual ({RUN_ID})",
        results_csv    = S2_CLUSTERING_CSV,
        embeddings_npy = S2_FUSED_50D_NPY,
        raw_data_csv   = RAW_DATA_CSV,
    )
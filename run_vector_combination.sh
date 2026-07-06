#!/bin/bash
export PYTHONPATH=src
export RUN_ID=2_v4_02_08
export IMAGE_WEIGHT=0.2
export TEXT_WEIGHT=0.8
export HDBSCAN_MIN_CLUSTER_SIZE=7
export HDBSCAN_MIN_SAMPLES=3
export HDBSCAN_CLUSTER_METHOD=leaf

echo "Strategy 2 full pipeline (RUN_ID=$RUN_ID, img=$IMAGE_WEIGHT, txt=$TEXT_WEIGHT)"

uv run python src/embeddings/mobileNetV3_embeddings.py && \
uv run python src/embeddings/paraphraseMultilingual_embeddings.py strategy2 && \
uv run python src/fusion/vector_combination_fusion.py && \
uv run python src/clustering/strategy2_clustering.py && \
uv run python src/evaluation/strategy2_evaluation.py

echo "Done."
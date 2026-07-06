#!/bin/bash
export PYTHONPATH=src
export RUN_ID=s1d_vitl14
export PREPROCESSED_LLAVA_CSV=outputs/preprocessed_llava_translated.csv
export RAW_DATA_CSV=data/raw/raw_data.csv

echo "Running Strategy 1d (RUN_ID=$RUN_ID)"

uv run python src/embeddings/clip_embeddings.py && \
uv run python src/clustering/strategy1_clustering.py && \
uv run python src/evaluation/strategy1_evaluation.py

echo "Done."
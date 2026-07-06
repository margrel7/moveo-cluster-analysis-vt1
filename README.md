# MOVEO Multimodal Clustering Pipeline

Clusters multimodal citizen reports (image + text) from the MOVEO Winterthur initiative
into thematic categories and generates workshop-ready outputs.

Developed as part of a Master's thesis (VT1) at the Zurich University of Applied Sciences (ZHAW), 2026.

---

## Requirements

- Python 3.12+
- Ollama running locally (`ollama serve`) with the following models pulled:
  - `llava` — for imputing text-missing reports during preprocessing
  - `gemma3:4b` — for generating cluster labels
- Input CSV matching the expected schema (see below)

Install dependencies:
```bash
uv sync
```

---

## Input data

Place your raw data CSV at `data/raw/raw_data.csv`.

Required columns:
```
photo_id, username, category, topic, description,
how_resolved_strengthened, lat, long, url_original, date
```

`category` must be either `works_well` or `problem`.

**Note:** The preprocessing scripts were developed for the MOVEO Winterthur dataset
(n=176, bilingual German/English). The following will likely need adaptation for
other datasets:
- LLaVA imputation prompt in `src/preprocessing/llava_imputation.py` references
  Winterthur public transport
- Swiss bounding box coordinate validation in `src/preprocessing/text_preprocessing.py`
- Fusion weights and HDBSCAN parameters (see Step 1 below)

---

## Step 1 — Run the clustering pipeline

```bash
RUN_ID=my_run .venv/bin/python3 run_pipeline.py
```

This chains all pipeline stages:
1. Text preprocessing (concatenation, normalisation)
2. Image download and quality screening
3. LLaVA imputation for text-missing reports
4. MobileNetV3 image embeddings (960-dim)
5. Multilingual text embeddings (768-dim)
6. Weighted vector-level fusion → UMAP 50D + UMAP 2D
7. K-means clustering (k=2..12, silhouette selection)
8. HDBSCAN clustering
9. Evaluation (silhouette, AMI, ARI, topic distribution)

Default parameters (optimised for MOVEO Winterthur n=176):

| Parameter | Default | Notes |
|---|---|---|
| IMAGE_WEIGHT | 0.3 | Increase if image quality is high |
| TEXT_WEIGHT | 0.7 | Decrease if text is sparse or noisy |
| HDBSCAN_MIN_CLUSTER_SIZE | 7 | Rule of thumb: sqrt(n) |
| HDBSCAN_MIN_SAMPLES | 3 | Lower = less noise, coarser clusters |
| HDBSCAN_CLUSTER_METHOD | leaf | `eom` gives fewer, larger clusters |

Override any parameter via environment variable:
```bash
IMAGE_WEIGHT=0.4 TEXT_WEIGHT=0.6 RUN_ID=my_run .venv/bin/python3 run_pipeline.py
```

Skip stages already completed:
```bash
# Skip preprocessing only
SKIP_PREPROCESSING=1 RUN_ID=my_run .venv/bin/python3 run_pipeline.py

# Skip preprocessing and embeddings
SKIP_PREPROCESSING=1 SKIP_EMBEDDINGS=1 RUN_ID=my_run .venv/bin/python3 run_pipeline.py
```

Outputs are saved to `outputs/*_{RUN_ID}.*`.

---

## Step 2 — Generate and review cluster labels

Requires Ollama running with `gemma3:4b`.

```bash
RUN_ID=my_run .venv/bin/python3 run_cluster_labeling.py
```

This script:
1. Sends the 8 most representative reports per cluster to Gemma3:4b
2. Generates bilingual labels (German + English) with context accumulation
3. Prints a preview of all labels
4. **Pauses and waits for your review**

Open `outputs/cluster_labels.csv` and edit the `label_de` and `label_en` columns.

Common issues to fix:
- Labels containing the city name — redundant, remove it
- Generic labels like "Erfahrungen" or "Feedback" — replace with specific themes
- Two clusters with similar labels — differentiate them
- Labels longer than 5 words — shorten

The `llm_raw` column shows the full LLM output for reference.
Do not edit `cluster_id`, `n_total`, `n_works_well`, or `n_problem` columns.

Type `yes` when done to confirm.

**Do not skip this step.** If `cluster_labels.csv` does not exist when the workshop
scripts run, cluster IDs will be used as labels in all outputs.

---

## Step 3 — Generate workshop outputs

```bash
RUN_ID=my_run .venv/bin/python3 run_workshop_data.py
```

Produces four outputs:

| File | Description |
|---|---|
| `outputs/workshop_clusters.csv` | All reports with assigned cluster labels (noise resolved) |
| `outputs/workshop_cluster_cards.html` | Printable A4 portrait cluster cards |
| `outputs/workshop_cluster_cards_with_map.html` | Cards with static map page per cluster |
| `outputs/workshop_map.html` | Interactive map with sentiment filter and cluster toggles |

**Print settings for cluster cards:**
Open in Chrome or Safari → Print → A4 portrait, no margins, scale to fit page.

**For spatial context during workshops:**
Use `workshop_map.html` in a browser — it is interactive, filterable by sentiment,
and supports per-cluster layer toggles. It is more useful than the static map pages
in most workshop contexts.

---

## Known limitations

**Results are not fully reproducible across environments.**
UMAP uses approximate nearest neighbor search with non-deterministic tie-breaking.
The fixed `random_state=42` helps within the same environment but does not guarantee
identical results across different machines or library versions. The thesis results
were produced on a specific machine in April 2026 and represent the canonical output.
Re-running the pipeline from scratch will produce structurally similar but not
identical clusters.

**Parameters are dataset-specific.**
The default weights (0.3/0.7) and HDBSCAN parameters were optimised for 176 bilingual
German/English citizen reports. They are a reasonable starting point but should be
tuned for datasets with different sizes, languages, or image quality.

**Cluster labels require human review.**
LLM-generated labels are a starting point. Generic, redundant, or inaccurate labels
will appear in all workshop outputs if not corrected.

---

## Project structure

```
run_pipeline.py              — full clustering pipeline (preprocessing → evaluation)
run_cluster_labeling.py      — generate and review cluster labels
run_workshop_data.py         — generate workshop outputs

src/
  config.py                  — centralised configuration and RUN_ID path management
  preprocessing/             — text normalisation, image download, LLaVA imputation
  embeddings/                — MobileNetV3 image embeddings, multilingual text embeddings
  fusion/                    — weighted vector-level combination and UMAP reduction
  clustering/                — K-means and HDBSCAN
  evaluation/                — silhouette, AMI, ARI, topic distribution
  utils/                     — noise assignment, cluster cards, cluster labeling, map

outputs/                     — all generated files
data/
  raw/                       — input CSV
  images/                    — downloaded report images
```


import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

nltk.download("stopwords", quiet=True)

_STOP_DE = set(stopwords.words("german"))
_STOP_EN = set(stopwords.words("english"))
STOP_WORDS = _STOP_DE | _STOP_EN


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------

def evaluate_external(
    labels: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float]:
    mask = labels != -1
    ami = adjusted_mutual_info_score(ground_truth[mask], labels[mask])
    ari = adjusted_rand_score(ground_truth[mask], labels[mask])
    print(f"  AMI  : {ami:.4f}")
    print(f"  ARI  : {ari:.4f}")
    return ami, ari


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def evaluate_internal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    exclude_noise: bool = True,
) -> float:
    if exclude_noise:
        mask = labels != -1
        emb = embeddings[mask]
        lbl = labels[mask]
    else:
        emb, lbl = embeddings, labels

    score = silhouette_score(emb, lbl, metric="cosine")
    label = "(cosine, excl. noise)" if exclude_noise else "(cosine)"
    print(f"  Silhouette {label}: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Topic keyword helpers
# ---------------------------------------------------------------------------

def extract_topic_keywords(topic: str) -> list[str]:
    if pd.isna(topic) or not str(topic).strip():
        return []
    topic = topic.lower()
    topic = re.sub(r"[^a-z0-9äöüß\s]", " ", topic)
    return [t for t in topic.split() if len(t) >= 3 and t not in STOP_WORDS]


def topic_consistency_score(topics: list[list[str]]) -> float:
    all_tokens = [t for keywords in topics for t in keywords]
    if not all_tokens:
        return 0.0
    counts = Counter(all_tokens)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return round(1.0 - norm_entropy, 4)


# ---------------------------------------------------------------------------
# Topic distribution printer
# ---------------------------------------------------------------------------

def analyse_topic_distribution(
    df: pd.DataFrame,
    label_col: str,
    top_n: int = 8,
):
    print(f"\n  {'─' * 60}")
    for cluster_id in sorted(df[label_col].unique()):
        cluster_df = df[df[label_col] == cluster_id]
        name = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"

        cat = cluster_df["category"].value_counts().to_dict()
        works_well = cat.get("works_well", 0)
        problem    = cat.get("problem", 0)

        topic_keywords = cluster_df["topic"].apply(extract_topic_keywords).tolist()
        all_tokens     = [t for kws in topic_keywords for t in kws]
        top_keywords   = [w for w, _ in Counter(all_tokens).most_common(top_n)]
        consistency    = topic_consistency_score(topic_keywords)
        suggested      = " / ".join(top_keywords[:2]) if top_keywords else "unlabelled"

        print(f"\n  {name}  (n={len(cluster_df)})")
        print(f"    works_well : {works_well}  |  problem : {problem}")
        print(f"    Top topics : {', '.join(top_keywords)}")
        print(f"    Consistency: {consistency}")
        print(f"    → Suggested label: '{suggested}'")

    print(f"\n  {'─' * 60}")


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    strategy_name: str,
    results_csv: str,
    embeddings_npy: str,
    raw_data_csv: str,
):
    print("Loading results...")
    results = pd.read_csv(results_csv)
    raw     = pd.read_csv(raw_data_csv)[["photo_id", "topic"]]
    df      = results.merge(raw, on="photo_id", how="left")

    gt           = (df["category"] == "works_well").astype(int).values
    embeddings   = np.load(embeddings_npy)
    emb_norm     = normalize(embeddings, norm="l2")
    km_labels    = df["kmeans_label"].values
    hdb_labels   = df["hdbscan_label"].values

    # ── K-means ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("K-MEANS EVALUATION")
    print("=" * 60)

    print(f"\n  External validation (vs works_well/problem):")
    km_ami, km_ari = evaluate_external(km_labels, gt)
    km_sil = evaluate_internal(emb_norm, km_labels, exclude_noise=False)

    print(f"\n  Category × Cluster crosstab:")
    print(pd.crosstab(df["kmeans_label"], df["category"]).to_string())

    print(f"\n  Topic distribution per cluster:")
    analyse_topic_distribution(df, "kmeans_label")

    # ── HDBSCAN ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HDBSCAN EVALUATION")
    print("=" * 60)

    n_noise = (hdb_labels == -1).sum()
    print(f"\n  Noise points: {n_noise} ({100 * n_noise / len(df):.1f}%)")

    print(f"\n  External validation (vs works_well/problem):")
    hdb_ami, hdb_ari = evaluate_external(hdb_labels, gt)
    hdb_sil = evaluate_internal(emb_norm, hdb_labels, exclude_noise=True)

    print(f"\n  Category × Cluster crosstab:")
    print(pd.crosstab(df["hdbscan_label"], df["category"]).to_string())

    print(f"\n  Topic distribution per cluster:")
    analyse_topic_distribution(df, "hdbscan_label")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{strategy_name.upper()} — SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame({
        "Metric":    ["AMI", "ARI", "Silhouette"],
        "K-means":   [round(km_ami,  4), round(km_ari,  4), round(km_sil,  4)],
        "HDBSCAN":   [round(hdb_ami, 4), round(hdb_ari, 4), round(hdb_sil, 4)],
    })
    print(summary.to_string(index=False))

    return {
        "kmeans":  {"ami": km_ami,  "ari": km_ari,  "silhouette": km_sil},
        "hdbscan": {"ami": hdb_ami, "ari": hdb_ari, "silhouette": hdb_sil},
    }
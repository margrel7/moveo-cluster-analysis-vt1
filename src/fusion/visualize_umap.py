import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ────────────────────────────────────────────────────────────
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42

STRATEGIES = {
    "CLIP Joint Embedding": {
        "embeddings": "outputs/embeddings_clip_joint.npy",
        "index": "outputs/embedding_index_clip_joint.csv",
    },
    "Vector-Level Combination": {
        "embeddings": "outputs/embeddings_vector_level.npy",
        "index": "outputs/embedding_index_vector_level.csv",
    },
}

OUTPUT_DIR = "outputs/visualizations"
# ─────────────────────────────────────────────────────────────────────────────


def compute_2d_umap(embeddings: np.ndarray) -> np.ndarray:
    """Compute 2D UMAP projection for visualization."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        random_state=RANDOM_STATE,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def plot_umap(
    reduced_2d: np.ndarray,
    index_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    strategy_name: str,
    output_path: str,
):
    merged = index_df.merge(
        raw_df[["photo_id", "category", "user_id"]],
        on="photo_id",
        how="left",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"UMAP 2D Projection — {strategy_name}", fontsize=14, fontweight="bold")

    # ── Plot 1: coloured by category ─────────────────────────────────────────
    ax1 = axes[0]
    color_map = {"works_well": "#2ecc71", "problem": "#e74c3c"}
    colors_cat = merged["category"].map(color_map).fillna("#aaaaaa")

    ax1.scatter(
        reduced_2d[:, 0], reduced_2d[:, 1],
        c=colors_cat,
        s=20,
        alpha=0.7,
        linewidths=0,
    )
    ax1.set_title("Coloured by participant category")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    n_works = (merged["category"] == "works_well").sum()
    n_problem = (merged["category"] == "problem").sum()
    legend_cat = [
        mpatches.Patch(color="#2ecc71", label=f"Works well (n={n_works})"),
        mpatches.Patch(color="#e74c3c", label=f"Problem (n={n_problem})"),
    ]
    ax1.legend(handles=legend_cat, fontsize=9)

    # ── Plot 2: coloured by user_id ───────────────────────────────────────────
    ax2 = axes[1]
    unique_users = merged["user_id"].unique()
    cmap = plt.cm.get_cmap("tab20", len(unique_users))
    user_color_map = {uid: cmap(i) for i, uid in enumerate(unique_users)}
    colors_user = merged["user_id"].map(user_color_map)

    ax2.scatter(
        reduced_2d[:, 0], reduced_2d[:, 1],
        c=list(colors_user),
        s=20,
        alpha=0.7,
        linewidths=0,
    )
    ax2.set_title(f"Coloured by participant (n={len(unique_users)} users)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_visualization():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_df = pd.read_csv("data/raw/raw_data.csv")

    print("=" * 60)
    print("UMAP 2D Visualization")
    print("=" * 60)
    print()

    for strategy_name, config in STRATEGIES.items():
        print(f"Processing: {strategy_name}...")

        embeddings = np.load(config["embeddings"])
        index_df = pd.read_csv(config["index"])

        print(f"  Computing 2D UMAP for {embeddings.shape[0]} reports...")
        reduced_2d = compute_2d_umap(embeddings)

        filename = strategy_name.lower().replace(" ", "_").replace("-", "_")
        output_path = os.path.join(OUTPUT_DIR, f"umap_2d_{filename}.png")

        plot_umap(reduced_2d, index_df, raw_df, strategy_name, output_path)
        print()

    print("Done. Visualizations saved to outputs/visualizations/")


if __name__ == "__main__":
    run_visualization()
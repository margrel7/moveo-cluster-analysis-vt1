import base64
import os
import folium
import pandas as pd
from folium.plugins import MarkerCluster

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_CSV = "outputs/workshop_clusters.csv"
OUTPUT_HTML = "outputs/workshop_map.html"

CLUSTER_LABELS = {
    0: "Veloinfrastruktur",
    1: "Unterführungen & Beleuchtung",
    2: "Warteräume & Aufenthaltskomfort",
    3: "Zugpünktlichkeit & Anschlüsse",
    4: "Grünflächen",
    5: "Altstadt & Bahnhofsumgebung",
    6: "Bahnhofsgestaltung & Zugang",
    7: "Busbetrieb",
    8: "Fussgänger & Kreuzungen",
}

# Distinct colours for 9 clusters
CLUSTER_COLORS = {
    0: "#2E86AB",  # blue       — Veloinfrastruktur
    1: "#E84855",  # red        — Unterführungen
    2: "#3BB273",  # green      — Warteräume
    3: "#F4A259",  # orange     — Zugpünktlichkeit
    4: "#6A994E",  # dark green — Grünflächen
    5: "#7B2D8B",  # purple     — Altstadt
    6: "#1D3461",  # dark blue  — Bahnhofsgestaltung
    7: "#F7B731",  # yellow     — Busbetrieb
    8: "#C1292E",  # dark red   — Fussgänger
}

# Winterthur city centre
MAP_CENTER = [47.4997, 8.7241]
MAP_ZOOM   = 14


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_base64(filepath: str) -> str | None:
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_popup(row: pd.Series, cluster_label: str, color: str) -> folium.Popup:
    """Build an HTML popup with photo, text and cluster label."""
    # Image
    img_html = ""
    path = row.get("local_image_path", "")
    if pd.notna(path) and os.path.exists(str(path)):
        b64 = image_to_base64(str(path))
        if b64:
            img_html = f'<img src="data:image/jpeg;base64,{b64}" style="width:240px;height:160px;object-fit:cover;border-radius:6px;margin-bottom:8px;display:block;">'

    # Text
    text = str(row.get("raw_text", "") or "").strip()
    if not text or text == "nan":
        text = str(row.get("topic", "") or "—").strip()
    if len(text) > 200:
        text = text[:197] + "..."

    category    = row.get("category", "")
    cat_label   = "✅ Works well" if category == "works_well" else "⚠️ Problem"
    cat_color   = "#1A6B3A" if category == "works_well" else "#C00000"
    cat_bg      = "#E2EFDA" if category == "works_well" else "#FCE4D6"

    popup_html = f"""
    <div style="width:260px; font-family:'Helvetica Neue',Arial,sans-serif;">
        {img_html}
        <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;color:{color};margin-bottom:4px;">
            {cluster_label}
        </div>
        <div style="font-size:13px;line-height:1.5;color:#222;
                    font-style:italic;margin-bottom:8px;">
            "{text}"
        </div>
        <div style="display:inline-block;padding:3px 10px;border-radius:12px;
                    background:{cat_bg};color:{cat_color};
                    font-size:11px;font-weight:600;">
            {cat_label}
        </div>
    </div>"""

    return folium.Popup(popup_html, max_width=280)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_map(results_csv: str, output_html: str):
    print("Loading data...")
    df = pd.read_csv(results_csv)

    # Merge image paths and text
    llava  = pd.read_csv("outputs/preprocessed_llava.csv")[
        ["photo_id", "raw_text", "local_image_path"]
    ]
    topics = pd.read_csv("data/raw/raw_data.csv")[["photo_id", "topic"]]
    df = df.merge(llava,   on="photo_id", how="left")
    df = df.merge(topics,  on="photo_id", how="left")

    # Exclude reports without coordinates
    before = len(df)
    df = df[df["lat"].notna() & df["long"].notna()].reset_index(drop=True)
    print(f"  {before - len(df)} reports excluded (no coordinates)")
    print(f"  {len(df)} reports on map\n")

    # ---------------------------------------------------------------------------
    # Build map
    # ---------------------------------------------------------------------------
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="CartoDB positron",
    )

    # One feature group per cluster so they can be toggled independently
    for cluster_id in sorted(df["hdbscan_label_assigned"].unique()):
        cluster_df  = df[df["hdbscan_label_assigned"] == cluster_id]
        label       = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        color       = CLUSTER_COLORS.get(cluster_id, "#888888")
        n           = len(cluster_df)
        n_good      = (cluster_df["category"] == "works_well").sum()
        n_prob      = (cluster_df["category"] == "problem").sum()

        group = folium.FeatureGroup(
            name=f'<span style="color:{color}">⬤</span> {label} ({n})',
            show=True,
        )

        for _, row in cluster_df.iterrows():
            popup  = make_popup(row, label, color)
            tooltip = f"{label} — {'✅' if row['category'] == 'works_well' else '⚠️'}"

            folium.CircleMarker(
                location=[row["lat"], row["long"]],
                radius=8,
                color="white",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=popup,
                tooltip=tooltip,
            ).add_to(group)

        group.add_to(m)
        print(f"  Cluster {cluster_id} ({label}): {n} markers ({n_good} good, {n_prob} problems)")

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:14px 18px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.15);
                font-family:'Helvetica Neue',Arial,sans-serif;">
        <div style="font-size:12px;font-weight:700;margin-bottom:10px;
                    letter-spacing:0.08em;text-transform:uppercase;color:#444;">
            MOVEO Clusters
        </div>"""

    for cluster_id, label in CLUSTER_LABELS.items():
        color = CLUSTER_COLORS.get(cluster_id, "#888")
        count = len(df[df["hdbscan_label_assigned"] == cluster_id])
        legend_html += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <div style="width:12px;height:12px;border-radius:50%;
                        background:{color};flex-shrink:0;"></div>
            <div style="font-size:12px;color:#333;">{label}
                <span style="color:#999;font-size:11px;">({count})</span>
            </div>
        </div>"""

    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    m.save(output_html)
    print(f"\nMap saved to {output_html}")
    print("Open in browser to explore.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_map(
        results_csv = RESULTS_CSV,
        output_html = OUTPUT_HTML,
    )
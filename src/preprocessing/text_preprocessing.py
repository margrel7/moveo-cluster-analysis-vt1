import re
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_FIELDS = ["topic", "description", "how_resolved_strengthened"]

PASSTHROUGH_COLS = [
    "photo_id",
    "user_id",
    "lat",
    "long",
    "url_original",
    "category",       # kept separate — used only for post-clustering validation
    "username",
    "date",
]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def concatenate_text_fields(row: pd.Series) -> str | None:
    parts = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    return " | ".join(parts) if parts else None


def normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9äöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_text_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.")

    # -- Step 1: Concatenate text fields ------------------------------------
    df["raw_text"] = df.apply(concatenate_text_fields, axis=1)

    n_with_text = df["raw_text"].notna().sum()
    n_without_text = df["raw_text"].isna().sum()
    print(f"\nText availability:")
    print(f"  With usable text : {n_with_text}")
    print(f"  No text (image-only, LLaVA candidates): {n_without_text}")

    # -- Step 2: Flag missing text and imputation placeholder ---------------
    df["is_text_missing"] = df["raw_text"].isna()
    df["is_llava_imputed"] = False   # updated by the LLaVA imputation script

    # -- Step 3: Normalise text (only where text exists) --------------------
    df["preprocessed_text"] = df["raw_text"].apply(
        lambda t: normalise_text(t) if pd.notna(t) else None
    )

    # -- Step 4: Flag bad coordinates (outside Switzerland bounding box) ----
    # Winterthur / CH rough bounds: lat 45.8–47.8, lon 5.9–10.5
    if "lat" in df.columns and "long" in df.columns:
        invalid_coords = (
            df["lat"].notna()
            & df["long"].notna()
            & ~(
                df["lat"].between(45.8, 47.8)
                & df["long"].between(5.9, 10.5)
            )
        )
        df.loc[invalid_coords, ["lat", "long"]] = None
        n_bad = invalid_coords.sum()
        if n_bad:
            print(f"\n  Nullified {n_bad} out-of-Switzerland coordinate(s) (e.g. Copenhagen GPS error).")

    # -- Step 5: Save output ------------------------------------------------
    keep_cols = [c for c in PASSTHROUGH_COLS if c in df.columns] + [
        "raw_text",
        "preprocessed_text",
        "is_text_missing",
        "is_llava_imputed",
    ]
    result = df[keep_cols].copy()
    result.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved to {output_path}")
    print(f"Output columns: {list(result.columns)}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_text_preprocessing(
        input_path="data/raw/raw_data_all_en.csv",
        output_path="outputs/preprocessed_text_all_en.csv",
    )
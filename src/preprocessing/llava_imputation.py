import base64
import re

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llava"          # swap for "llava:13b" if you have the VRAM
OUTPUT_CSV = "outputs/preprocessed_llava.csv"

CAPTION_PROMPT = (
    "You are a regular public transport user in Winterthur, Switzerland "
    "submitting a short report about something you noticed on your daily commute. "
    "Look at this photo and respond in exactly this format:\n"
    "Topic: [3-5 words naming what you see]\n"
    "Description: [one sentence describing the situation]\n"
    "How to resolve or strengthen: [one short sentence]\n"
    "Keep the total response under 50 words. "
    "Be direct and practical, like a commuter — not a researcher."
)


# ---------------------------------------------------------------------------
# Text normalisation (mirrors text_preprocessing.py)
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9äöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# LLaVA captioning
# ---------------------------------------------------------------------------

def encode_image_base64(filepath: str) -> str:
    """Read a local image file and return its base64-encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(filepath: str) -> str | None:
    """
    Send an image to LLaVA via Ollama and return a normalised caption.
    Returns None if the model call fails.
    """
    try:
        image_b64 = encode_image_base64(filepath)
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": CAPTION_PROMPT,
                    "images": [image_b64],
                }
            ],
        )
        raw_caption = response["message"]["content"].strip()
        return normalise_text(raw_caption)
    except Exception as e:
        print(f"  [ERROR] LLaVA call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_llava_imputation(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.")

    # Add raw caption column for transparency / debugging
    if "llava_raw_caption" not in df.columns:
        df["llava_raw_caption"] = None

    # Identify candidates: text missing AND image available
    candidates = df[
        df["is_text_missing"] & df["local_image_path"].notna()
    ]
    no_image = df[
        df["is_text_missing"] & df["local_image_path"].isna()
    ]

    print(f"\nText-missing rows          : {df['is_text_missing'].sum()}")
    print(f"  → With image (LLaVA)     : {len(candidates)}")
    print(f"  → Without image (skipped): {len(no_image)}")

    if len(no_image) > 0:
        print(f"  [WARN] {len(no_image)} row(s) have no text AND no image — "
              f"they cannot be imputed and will remain empty.")

    if len(candidates) == 0:
        print("\nNo rows to impute. Saving unchanged.")
        df.to_csv(output_path, index=False)
        return df

    print(f"\nRunning LLaVA captioning on {len(candidates)} image(s)...\n")

    n_success = 0
    n_failed = 0

    for idx, (row_idx, row) in enumerate(candidates.iterrows(), start=1):
        photo_id = row["photo_id"]
        filepath = row["local_image_path"]

        print(f"[{idx}/{len(candidates)}] {photo_id[:8]}...")

        caption = generate_caption(filepath)

        if caption:
            df.at[row_idx, "preprocessed_text"] = caption
            df.at[row_idx, "llava_raw_caption"] = caption
            df.at[row_idx, "is_llava_imputed"] = True
            print(f"  ✓ Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}")
            n_success += 1
        else:
            print(f"  ✗ Failed — row will remain text-empty.")
            n_failed += 1

    # -- Summary ------------------------------------------------------------
    print("\n=== LLaVA Imputation Summary ===")
    print(f"  Candidates   : {len(candidates)}")
    print(f"  Imputed OK   : {n_success}")
    print(f"  Failed       : {n_failed}")
    print(f"  Still empty  : {df['preprocessed_text'].isna().sum()} row(s)")

    df.to_csv(output_path, index=False)
    print(f"\nImputed data saved to {output_path}")
    print(f"Output columns: {list(df.columns)}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_llava_imputation(
        input_path="outputs/preprocessed_images.csv",
        output_path=OUTPUT_CSV,
    )
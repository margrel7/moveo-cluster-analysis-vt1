import base64
import re
import time

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "gemma3:4b"
OUTPUT_CSV   = "outputs/strategy3_enriched.csv"

ENRICHMENT_PROMPT_TEMPLATE = """Example input:
Photo: bus stop with shelter
User report: the heating is broken and it gets very cold waiting here

Example output:
Covered bus stop shelter with seating area. The heating system is non-functional, leaving passengers exposed to cold temperatures during winter waiting times. Infrastructure maintenance issue affecting passenger comfort.

Now do the same for this report:
Photo: [attached]
User report: {user_text}

Output:"""

ENRICHMENT_PROMPT_IMAGE_ONLY = """You are documenting a public transport citizen report.

Image: [attached]

Write 2-3 sentences describing what you see.
Start with the location type. Describe the key infrastructure and any visible issue or positive feature."""


# ---------------------------------------------------------------------------
# Preamble stripping — defined before enrich_report which calls it
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    patterns = [
        r"^here'?s?\s+a\s+description[^:]*:\s*",
        r"^based on the (image and )?(user )?text[^:]*:\s*",
        r"^description:\s*",
        r"^report:\s*",
        r"^location:\s*",
        r"^the location is\s+",
        # Only strip the verb phrase — not everything after it
        r"^the (image|photo) (depicts|shows|features)\s+",
        r"^this report (concerns|details|describes|highlights|shows)\s+",
        r"^this (image|photo) (shows|depicts|features)\s+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# Text normalisation (mirrors preprocessing pipeline)
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9äöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Gemma4 enrichment
# ---------------------------------------------------------------------------

def encode_image_base64(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def enrich_report(
    filepath: str,
    user_text: str | None,
) -> tuple[str | None, str | None]:
    try:
        image_b64 = encode_image_base64(filepath)

        if user_text and str(user_text).strip():
            prompt = ENRICHMENT_PROMPT_TEMPLATE.format(user_text=str(user_text).strip())
        else:
            prompt = ENRICHMENT_PROMPT_IMAGE_ONLY

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }],
        )
        raw = response["message"]["content"].strip()
        raw_backup = raw                        # preserve before any cleaning
        cleaned = strip_preamble(raw)
        normalised = normalise_text(cleaned)
        return normalised, raw_backup

    except Exception as e:
        print(f"  [ERROR] Gemma4 call failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_llm_enrichment(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.")
    print(f"Model: {OLLAMA_MODEL}\n")

    enriched_texts = []
    enriched_raws  = []
    enrichment_oks = []

    n_success = 0
    n_failed  = 0

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        photo_id  = row["photo_id"]
        filepath  = row.get("local_image_path")
        user_text = row.get("preprocessed_text")

        print(f"[{idx}/{len(df)}] {photo_id[:8]}...")

        if pd.isna(filepath):
            print(f"  [SKIP] No local image — cannot enrich without visual input.")
            enriched_texts.append(None)
            enriched_raws.append(None)
            enrichment_oks.append(False)
            n_failed += 1
            continue

        normalised, raw_backup = enrich_report(filepath, user_text)

        if normalised:
            enriched_texts.append(normalised)
            enriched_raws.append(raw_backup)    # always save original output
            enrichment_oks.append(True)
            print(f"  ✓ {normalised[:80]}{'...' if len(normalised) > 80 else ''}")
            n_success += 1
        else:
            fallback = str(user_text) if pd.notna(user_text) else None
            enriched_texts.append(fallback)
            enriched_raws.append(None)
            enrichment_oks.append(False)
            print(f"  ✗ Failed — falling back to original text.")
            n_failed += 1

        time.sleep(0.1)

    df["enriched_text"] = enriched_texts
    df["enriched_raw"]  = enriched_raws
    df["enrichment_ok"] = enrichment_oks

    # -- Save raw backup FIRST — protects enriched_raw before any cleanup ---
    backup_path = output_path.replace(".csv", "_raw_backup.csv")
    df.to_csv(backup_path, index=False)
    print(f"\nRaw backup saved to    {backup_path}")

    # -- Save main output ---------------------------------------------------
    df.to_csv(output_path, index=False)

    print(f"\n=== Strategy 3 — LLM Enrichment Summary ===")
    print(f"  Total reports  : {len(df)}")
    print(f"  Enriched OK    : {n_success}")
    print(f"  Failed/skipped : {n_failed}")
    print(f"  Still empty    : {df['enriched_text'].isna().sum()}")
    print(f"\nEnriched data saved to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_llm_enrichment(
        input_path="outputs/preprocessed_llava.csv",
        output_path=OUTPUT_CSV,
    )
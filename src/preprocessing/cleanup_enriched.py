import re
import pandas as pd

INPUT_CSV  = "outputs/strategy3_enriched.csv"
OUTPUT_CSV = "outputs/strategy3_enriched.csv" 


PREAMBLE_PATTERNS = [
    r"^here'?s?\s+a\s+description[^:]*:\s*",
    r"^based on the (image and )?(user )?text[^:]*:\s*",
    r"^the (image|photo) (depicts|shows)[^,]*,?\s*",
    r"^description:\s*",
    r"^this report (concerns|details|describes|highlights|shows)[^,]*,?\s*",
    r"^this (image|photo) (shows|depicts|features)[^,]*,?\s*",
    r"^the location is\s+",
    r"^report:\s*",
]

# Location names removed because all 176 reports are from the same city —
# 'winterthur' and 'switzerland' appear uniformly across all rows and add
# no discriminative signal for clustering. Documented in methodology section.
NOISE_PATTERNS = [
    r"\bin winterthur\b",
    r"\bin switzerland\b",
    r"\bwinterthur\b",
    r"\bswitzerland\b",
    r"\bschweiz\b",
]


def strip_preamble(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pattern in PREAMBLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_noise(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")

    before = df["enriched_text"].copy()
    df["enriched_text"] = df["enriched_text"].apply(strip_preamble).apply(strip_noise)

    changed = (before != df["enriched_text"]).sum()
    print(f"Cleaned {changed} rows (preamble + location noise).")

    # Show a few examples
    mask = before != df["enriched_text"]
    for _, row in df[mask].head(5).iterrows():
        print(f"\n  Before: {str(before[row.name])[:80]}")
        print(f"  After : {str(row['enriched_text'])[:80]}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved cleaned data to {OUTPUT_CSV}")
import pandas as pd
import re

# Load files
orig  = pd.read_csv("outputs/preprocessed_llava.csv")
trans = pd.read_csv("data/raw/raw_data_all_en.csv")

# Extract photo_id from filename (remove .jpeg)
trans["photo_id"] = trans["filename"].str.replace("..jpeg", "", regex=False).str.strip()

# Normalise translated description
def normalise(text):
    if not isinstance(text, str): return None
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

trans["preprocessed_text_en"] = trans["description"].apply(normalise)

# Merge translated text onto original pipeline output
merged = orig.copy()
merged = merged.merge(trans[["photo_id", "preprocessed_text_en"]], on="photo_id", how="left")

# Replace preprocessed_text with translated version (keep LLaVA captions for 10 imputed rows)
mask_not_imputed = merged["is_llava_imputed"] == False
merged.loc[mask_not_imputed, "preprocessed_text"] = merged.loc[mask_not_imputed, "preprocessed_text_en"]
merged = merged.drop(columns=["preprocessed_text_en"])

merged.to_csv("outputs/preprocessed_llava_translated.csv", index=False)
print(f"Saved {len(merged)} rows")
print(f"Missing text: {merged['preprocessed_text'].isna().sum()}")
print(merged[["photo_id", "preprocessed_text"]].head(3).to_string())
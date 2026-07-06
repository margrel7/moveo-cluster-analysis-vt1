import os
import cv2
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_DIR = "data/images"
OUTPUT_CSV = "outputs/preprocessed_images.csv"

BLUR_THRESHOLD = 50.0       # Laplacian variance below this → blur warning
DARKNESS_THRESHOLD = 20.0   # Mean brightness below this  → darkness warning

# ---------------------------------------------------------------------------
# HTTP session with retry logic
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


SESSION = _build_session()

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_image(url: str, photo_id: str) -> str | None:
    filepath = os.path.join(IMAGE_DIR, f"{photo_id}.jpeg")

    if os.path.exists(filepath):
        return filepath

    try:
        response = SESSION.get(url, timeout=10)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    except Exception as e:
        print(f"  [ERROR] Failed to download {photo_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Quality screening
# ---------------------------------------------------------------------------

def compute_laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_mean_brightness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def screen_image_quality(filepath: str) -> dict:
    image = cv2.imread(filepath)

    if image is None:
        return {
            "image_ok": False,
            "quality_warning": "could not read image",
            "blur_score": None,
            "brightness_score": None,
        }

    brightness = compute_mean_brightness(image)
    blur = compute_laplacian_variance(image)
    warnings = []

    if brightness < DARKNESS_THRESHOLD:
        warnings.append("too dark")
    if blur < BLUR_THRESHOLD:
        warnings.append("too blurry")

    return {
        "image_ok": True,
        "quality_warning": ", ".join(warnings) if warnings else "none",
        "blur_score": round(blur, 2),
        "brightness_score": round(brightness, 2),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_image_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.\n")

    results = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        photo_id = row["photo_id"]
        url = row.get("url_original")

        print(f"[{idx}/{len(df)}] {photo_id[:8]}...")

        # Guard: missing URL
        if pd.isna(url) or not str(url).strip():
            print(f"  [SKIP] No URL for {photo_id[:8]}")
            results.append({
                "photo_id": photo_id,
                "local_image_path": None,
                "download_ok": False,
                "image_ok": False,
                "quality_warning": "no url",
                "blur_score": None,
                "brightness_score": None,
            })
            continue

        # Step 1: Download
        filepath = download_image(url, photo_id)
        if filepath is None:
            results.append({
                "photo_id": photo_id,
                "local_image_path": None,
                "download_ok": False,
                "image_ok": False,
                "quality_warning": "download failed",
                "blur_score": None,
                "brightness_score": None,
            })
            continue

        # Step 2: Quality screening (advisory only — no hard exclusion)
        quality = screen_image_quality(filepath)
        if quality["quality_warning"] not in ("none", None):
            print(f"  [WARNING] {quality['quality_warning']}")

        results.append({
            "photo_id": photo_id,
            "local_image_path": filepath,
            "download_ok": True,
            **quality,
        })

    # -- Merge image results back onto the preprocessed dataframe -----------
    image_df = pd.DataFrame(results)
    result = df.merge(image_df, on="photo_id", how="left")

    # -- Summary ------------------------------------------------------------
    n_downloaded = image_df["download_ok"].sum()
    n_failed = (~image_df["download_ok"]).sum()
    n_warnings = (
        image_df["quality_warning"]
        .apply(lambda w: w not in ("none", None) and pd.notna(w))
        .sum()
    )

    print("\n=== Image Preprocessing Summary ===")
    print(f"  Total reports    : {len(image_df)}")
    print(f"  Downloaded OK    : {n_downloaded}")
    print(f"  Failed downloads : {n_failed}")
    print(f"  Quality warnings : {n_warnings}  (advisory — not excluded)")

    if n_warnings > 0:
        warned = image_df[
            image_df["quality_warning"].apply(
                lambda w: w not in ("none", None) and pd.notna(w)
            )
        ][["photo_id", "quality_warning", "blur_score", "brightness_score"]]
        print("\n  Warned images:")
        print(warned.to_string(index=False))

    result.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Output columns: {list(result.columns)}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_image_preprocessing(
        input_path="outputs/preprocessed_text.csv",
        output_path=OUTPUT_CSV,
    )
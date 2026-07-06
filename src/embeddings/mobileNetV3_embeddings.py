import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from src.config import (
    PREPROCESSED_LLAVA_CSV,
    S2_IMAGE_EMBEDDINGS_NPY,
    S2_IMAGE_EMBEDDINGS_CSV,
    RUN_ID,
    print_config,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 960

# ImageNet normalisation — required for MobileNetV3 pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    print("Loading MobileNetV3 Large (ImageNet pretrained)...")
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )

    # Remove classifier — keep only feature extractor + adaptive avg pool
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return model, transform, device


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def l2_normalise(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def embed_image(
    filepath: str,
    model,
    transform,
    device: torch.device,
) -> np.ndarray | None:
    try:
        image = transform(Image.open(filepath).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image)
        vec = features.squeeze().cpu().numpy()
        # MobileNetV3 classifier removal may leave a 2D tensor on some versions
        if vec.ndim > 1:
            vec = vec.flatten()
        return l2_normalise(vec)
    except Exception as e:
        print(f"  [ERROR] Image embedding failed for {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_image_embeddings(input_path: str, output_csv: str, output_npy: str) -> pd.DataFrame:
    model, transform, device = load_model()

    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reports.\n")

    embeddings = []
    metadata   = []
    n_failed   = 0

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        photo_id = row["photo_id"]
        print(f"[{idx}/{len(df)}] {photo_id[:8]}...")

        filepath = row.get("local_image_path")
        if pd.isna(filepath):
            print(f"  [SKIP] No local image path.")
            n_failed += 1
            continue

        emb = embed_image(filepath, model, transform, device)
        if emb is None:
            n_failed += 1
            continue

        embeddings.append({"photo_id": photo_id, "embedding": emb})
        metadata.append({
            "photo_id":         photo_id,
            "username":         row.get("username"),
            "category":         row.get("category"),
            "lat":              row.get("lat"),
            "long":             row.get("long"),
            "is_llava_imputed": row.get("is_llava_imputed", False),
        })

    emb_matrix = np.array([e["embedding"] for e in embeddings], dtype=np.float32)
    meta_df    = pd.DataFrame(metadata)

    np.save(output_npy, emb_matrix)
    meta_df.to_csv(output_csv, index=True, index_label="array_index")

    print(f"\n=== Strategy 2 — Image Embeddings (MobileNetV3) ({RUN_ID}) ===")
    print(f"  Embedded       : {len(embeddings)}")
    print(f"  Failed/skipped : {n_failed}")
    print(f"  Matrix shape   : {emb_matrix.shape}")
    print(f"  Saved to       : {output_npy}")

    return meta_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_config("strategy2_image_embeddings.py")
    run_image_embeddings(
        input_path=PREPROCESSED_LLAVA_CSV,
        output_csv=S2_IMAGE_EMBEDDINGS_CSV,
        output_npy=S2_IMAGE_EMBEDDINGS_NPY,
    )
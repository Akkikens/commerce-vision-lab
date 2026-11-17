"""Paths and defaults. Everything heavy is cached outside the repo."""

from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("CVLAB_CACHE", Path.home() / ".cache" / "cvlab"))
ARTIFACT_DIR = Path(os.environ.get("CVLAB_ARTIFACTS", "artifacts"))
RESULTS_DIR = Path("results")

# Dataset: Kaggle "Fashion Product Images (Small)" (MIT on Kaggle) re-hosted on the
# Hugging Face Hub at 384x512. We read only the parquet row groups we need.
DATASET_REPO = "benitomartin/fashion-product-images-small-384x512"
DATASET_SHARD = "data/train-00000-of-00005.parquet"

# Generic CLIP baseline and a fashion-tuned SigLIP. Both load through open_clip.
MODELS = {
    "clip-vit-b32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "fashion-siglip": ("hf-hub:Marqo/marqo-fashionSigLIP", None),
}
DEFAULT_MODEL = "clip-vit-b32"

SEED = 13

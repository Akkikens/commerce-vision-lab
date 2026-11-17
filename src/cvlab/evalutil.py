"""Helpers shared by the eval scripts: cached embeddings and result files."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, RESULTS_DIR, SEED


def environment() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "platform": platform.platform(terse=True)}
    try:
        import subprocess

        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def write_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"name": name, "seed": SEED, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "environment": environment(), **payload}
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def cached_image_embeddings(model_key: str, products, embedder=None) -> tuple[np.ndarray, float]:
    """Embed catalog images once per model; returns (embeddings, seconds spent embedding)."""
    path = CACHE_DIR / f"img_emb_{model_key}_{len(products)}.npz"
    if path.exists():
        d = np.load(path)
        if list(d["ids"]) == [p.id for p in products]:
            return d["emb"], float(d["seconds"])
    if embedder is None:
        from .embeddings import load_embedder

        embedder = load_embedder(model_key)
    t0 = time.perf_counter()
    emb = embedder.embed_images([p.image() for p in products])
    secs = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=emb, ids=np.array([p.id for p in products]), seconds=secs)
    return emb, secs

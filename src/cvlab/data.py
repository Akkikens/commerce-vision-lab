"""Catalog loading: a small, stratified apparel subset of a public dataset.

Images are cached under ``CVLAB_CACHE`` (default ``~/.cache/cvlab``) and are never
written into the repository.
"""

from __future__ import annotations

import io
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import CACHE_DIR, DATASET_REPO, DATASET_SHARD, SEED

# Labels a pattern only when the product name states one explicitly. This is a
# weak (keyword) label, not ground truth; see README "Label notes".
PATTERN_RULES: dict[str, re.Pattern[str]] = {
    "striped": re.compile(r"\bstrip(e|ed|es)\b", re.I),
    "checked": re.compile(r"\bcheck(ed|s)?\b", re.I),
    "printed": re.compile(r"\bprint(ed)?\b", re.I),
    "solid": re.compile(r"\b(solid|plain)\b", re.I),
}

APPAREL_MASTER = {"Apparel", "Footwear"}


def pattern_from_name(name: str) -> str | None:
    hits = [label for label, rx in PATTERN_RULES.items() if rx.search(name or "")]
    return hits[0] if len(hits) == 1 else None


@dataclass
class Product:
    id: int
    name: str
    article_type: str
    colour: str
    gender: str
    usage: str
    pattern: str | None
    image_bytes: bytes = field(repr=False)

    def image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")

    def attributes(self) -> dict[str, str]:
        attrs = {
            "category": self.article_type,
            "colour": self.colour,
            "gender": self.gender,
            "usage": self.usage,
        }
        if self.pattern:
            attrs["pattern"] = self.pattern
        return attrs


def _subset_path(n_max: int) -> Path:
    return CACHE_DIR / f"catalog_subset_{n_max}.parquet"


def load_catalog(
    n_max: int = 3000,
    per_type_cap: int = 250,
    min_per_type: int = 40,
    row_groups: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[Product]:
    """Return a stratified apparel/footwear subset, downloading and caching on first use."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _subset_path(n_max)
    if not path.exists():
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(f"datasets/{DATASET_REPO}/{DATASET_SHARD}") as fh:
            pf = pq.ParquetFile(fh)
            cols = ["id", "gender", "masterCategory", "articleType", "baseColour",
                    "usage", "productDisplayName", "image"]
            table = pa.concat_tables([pf.read_row_group(i, columns=cols) for i in row_groups])
        rows = [r for r in table.to_pylist()
                if r["masterCategory"] in APPAREL_MASTER and r["baseColour"]
                and r["productDisplayName"] and r["image"] and r["image"]["bytes"]]
        counts = Counter(r["articleType"] for r in rows)
        keep_types = {t for t, c in counts.items() if c >= min_per_type}
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["articleType"] in keep_types:
                by_type[r["articleType"]].append(r)
        rng = random.Random(SEED)
        picked: list[dict] = []
        for t in sorted(by_type):
            group = by_type[t]
            rng.shuffle(group)
            picked.extend(group[:per_type_cap])
        rng.shuffle(picked)
        picked = picked[:n_max]
        out = pa.Table.from_pylist([
            {"id": r["id"], "name": r["productDisplayName"], "article_type": r["articleType"],
             "colour": r["baseColour"], "gender": r["gender"], "usage": r["usage"] or "",
             "image_bytes": r["image"]["bytes"]}
            for r in picked
        ])
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, path)

    table = pq.read_table(path)
    return [
        Product(id=r["id"], name=r["name"], article_type=r["article_type"], colour=r["colour"],
                gender=r["gender"], usage=r["usage"], pattern=pattern_from_name(r["name"]),
                image_bytes=r["image_bytes"])
        for r in table.to_pylist()
    ]


def split_by_id(products: list[Product], test_frac: float = 0.3) -> tuple[list[int], list[int]]:
    """Deterministic train/test split on product id (no product appears in both)."""
    ids = sorted({p.id for p in products})
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n_test = int(len(ids) * test_frac)
    test_ids = set(ids[:n_test])
    train_idx = [i for i, p in enumerate(products) if p.id not in test_ids]
    test_idx = [i for i, p in enumerate(products) if p.id in test_ids]
    return train_idx, test_idx

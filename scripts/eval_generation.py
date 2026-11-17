"""Generate on-model scene images and measure the quality gate.

Three variants per product:
  * composite        - cut-out pasted on an SD-Turbo scene (product pixels untouched)
  * harmonized@0.35  - composite + low-strength img2img over the whole image
  * control@0.8      - composite + high-strength img2img: a deliberately degraded output

Labels are by construction (composite = acceptable, control = must be caught); the
harmonized variant is the realistic, uncertain case and is only reported. Gate
thresholds for the preservation checks are calibrated on a calibration split of
products and then applied unchanged to a held-out test split.

Writes results/generation.json; images and the review queue go to artifacts/ (gitignored).
Run: python scripts/eval_generation.py
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import time
from dataclasses import asdict

import numpy as np

from cvlab.config import ARTIFACT_DIR, SEED
from cvlab.data import load_catalog
from cvlab.embeddings import load_embedder
from cvlab.evalutil import write_result
from cvlab.generation.backgrounds import SDTurboGenerator
from cvlab.generation.pipeline import SCENES, ProductScenePipeline
from cvlab.generation.quality import QualityGate, Thresholds
from cvlab.generation.review_queue import ReviewQueue
from cvlab.generation.segment import RembgSegmenter

VARIANTS = {"composite": 0.0, "harmonized@0.35": 0.35, "control@0.8": 0.8}
# metric -> True if higher is better
PRESERVATION = {"mask_iou": True, "product_similarity": True, "product_ssim": True,
                "colour_delta_e": False}
TYPES = ["Tops", "Tshirts", "Shirts", "Dresses", "Kurtas", "Track Pants", "Shorts",
         "Jeans", "Sweatshirts", "Jackets", "Trousers", "Tunics", "Sports Shoes", "Heels"]


def calibrate(good: list[float], bad: list[float], higher_better: bool) -> tuple[float, float]:
    """Threshold maximising balanced accuracy on the calibration split."""
    cands = sorted(set(good) | set(bad))
    best_t, best_ba = cands[0], -1.0
    for a, b in zip(cands, cands[1:] + [cands[-1]], strict=True):
        t = (a + b) / 2
        if higher_better:
            tpr, tnr = np.mean([g >= t for g in good]), np.mean([x < t for x in bad])
        else:
            tpr, tnr = np.mean([g <= t for g in good]), np.mean([x > t for x in bad])
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_t, best_ba = t, ba
    return float(best_t), float(best_ba)


def summarize(vals: list[float]) -> dict:
    return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=20)
    args = ap.parse_args()

    products = [p for p in load_catalog() if p.article_type in TYPES]
    rng = random.Random(SEED)
    rng.shuffle(products)
    # Round-robin over types so both splits are diverse.
    by_type: dict[str, list] = {}
    for p in products:
        by_type.setdefault(p.article_type, []).append(p)
    picked = []
    while len(picked) < args.n_cal + args.n_test:
        for t in TYPES:
            if by_type.get(t) and len(picked) < args.n_cal + args.n_test:
                picked.append(by_type[t].pop())
    cal, test = picked[: args.n_cal], picked[args.n_cal:]

    t0 = time.perf_counter()
    seg = RembgSegmenter("u2net")
    gen = SDTurboGenerator(steps=1)
    emb = load_embedder("clip-vit-b32")
    load_s = time.perf_counter() - t0
    gate = QualityGate(emb, seg, Thresholds())
    pipe = ProductScenePipeline(seg, gen, gate)
    scenes = list(SCENES)
    img_dir = ARTIFACT_DIR / "generation"
    img_dir.mkdir(parents=True, exist_ok=True)

    def run(split):
        rows = []
        for i, p in enumerate(split):
            desc = f"a {p.colour.lower()} {p.article_type.lower()} worn by a model"
            scene = scenes[i % len(scenes)]
            for variant, strength in VARIANTS.items():
                t = time.perf_counter()
                r = pipe.run(p.image(), desc, scene, seed=SEED + i, harmonize_strength=strength)
                secs = time.perf_counter() - t
                r.image.save(img_dir / f"{p.id}_{variant}.png")
                rows.append({"product_id": p.id, "article_type": p.article_type,
                             "scene": scene, "variant": variant, "seconds": round(secs, 3),
                             **{k: v for k, v in asdict(r.gate).items()
                                if k not in ("passed", "reasons")}})
            print(f"  {p.id} {p.article_type} done")
        return rows

    print("calibration split")
    cal_rows = run(cal)
    print("test split")
    test_rows = run(test)

    # Calibrate preservation thresholds: composite (good) vs control (bad).
    th = Thresholds()
    calib = {}
    for metric, hb in PRESERVATION.items():
        good = [r[metric] for r in cal_rows if r["variant"] == "composite"]
        bad = [r[metric] for r in cal_rows if r["variant"] == "control@0.8"]
        t, ba = calibrate(good, bad, hb)
        setattr(th, metric if metric != "colour_delta_e" else "colour_delta_e", round(t, 4))
        calib[metric] = {"threshold": round(t, 4), "calibration_balanced_accuracy": round(ba, 4)}

    def passes(r) -> tuple[bool, list[str]]:
        reasons = []
        if r["prompt_margin"] < th.prompt_margin:
            reasons.append("prompt_margin")
        for metric, hb in PRESERVATION.items():
            t = getattr(th, metric)
            if (hb and r[metric] < t) or (not hb and r[metric] > t):
                reasons.append(metric)
        return not reasons, reasons

    queue = ReviewQueue(ARTIFACT_DIR / "review_queue.jsonl")
    if queue.path.exists():
        queue.path.unlink()
    per_variant = {}
    for variant in VARIANTS:
        rows = [r for r in test_rows if r["variant"] == variant]
        verdicts = [passes(r) for r in rows]
        for r, (ok, reasons) in zip(rows, verdicts, strict=True):
            if not ok:
                queue.submit({"product_id": r["product_id"], "variant": variant,
                              "reasons": reasons})
        fail_counts: dict[str, int] = {}
        for _, reasons in verdicts:
            for x in reasons:
                fail_counts[x] = fail_counts.get(x, 0) + 1
        per_variant[variant] = {
            "n": len(rows),
            "pass_rate": round(float(np.mean([v[0] for v in verdicts])), 4),
            "failed_checks": fail_counts,
            "seconds_per_image": summarize([r["seconds"] for r in rows]),
            **{m: summarize([r[m] for r in rows])
               for m in ["prompt_alignment", "prompt_margin", *PRESERVATION]},
        }
        print(variant, per_variant[variant]["pass_rate"], fail_counts)

    out = {
        "models": {"segmenter": seg.name, "generator": gen.name, "gate_embedder": emb.name,
                   "canvas": list(pipe.canvas)},
        "model_load_seconds": round(load_s, 1),
        "splits": {"calibration_products": len(cal), "test_products": len(test),
                   "test_types": sorted({p.article_type for p in test})},
        "thresholds": {**asdict(th)},
        "calibration": calib,
        "test": per_variant,
        "review_queue_items": len(queue.pending()),
        "review_queue_rate_test": round(len(queue.pending()) / len(test_rows), 4),
        "label_note": "composite=acceptable and control@0.8=unacceptable by construction, "
                      "not by human judgement",
    }
    print("wrote", write_result("generation", out))


if __name__ == "__main__":
    main()

"""Print the README results tables from results/*.json (no numbers are typed by hand)."""

from __future__ import annotations

import json
from pathlib import Path

R = Path("results")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def load(name: str) -> dict | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def classification(d: dict) -> list[str]:
    models = list(d["models"])
    out = ["| Task (classes, test n) | Majority baseline | "
           + " | ".join(f"{m} zero-shot | {m} linear probe" for m in models) + " |",
           "|---|---|" + "---|---|" * len(models)]
    for task, t in d["tasks"].items():
        cells = []
        for m in models:
            for kind in ("zero_shot", "linear_probe"):
                r = t[m][kind]
                cells.append(f"{r['accuracy']:.3f} / {r['macro_f1']:.3f}")
        out.append(f"| {task} ({t['n_classes']}, {t['n_test']}) | "
                   f"{t['majority_baseline_acc']:.3f} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Cells are accuracy / macro-F1. Embedding cost: " + "; ".join(
        f"{m} {v['embed_ms_per_image']} ms/image ({v['dim']}-d)" for m, v in d["models"].items()))
    return out


def search(d: dict) -> list[str]:
    models = list(d["models"])
    rows = [
        ("Text: listing title -> that item (any item with the same title)", "text_title",
         ["recall@1", "recall@5", "recall@10"]),
        ("Text: shopper query \"<colour> <type>\"", "text_attribute", ["precision@10"]),
        ("Image: re-cropped/mirrored/brightened copy -> original", "image_near_duplicate",
         ["recall@1", "recall@5"]),
        ("Image: similar items, same type", "image_similar", ["precision@10_same_type"]),
        ("Image: similar items, same type + colour", "image_similar",
         ["precision@10_same_type_and_colour"]),
    ]
    out = ["| Query set | Metric | " + " | ".join(models) + " |",
           "|---|---|" + "---|" * len(models)]
    for label, key, metrics in rows:
        for m in metrics:
            out.append(f"| {label} | {m.split('_')[0]} | "
                       + " | ".join(f"{d['models'][x][key][m]:.3f}" for x in models) + " |")
    q = d["queries"]
    out.append("")
    out.append(f"Catalog {d['catalog_size']} items, index `{d['index']}` (exact inner product); "
               f"{q['title']} title queries, {q['attribute']} attribute queries, "
               f"{q['image']} image queries.")
    return out


def generation(d: dict) -> list[str]:
    t = d["test"]
    out = ["| Variant (test n) | Gate pass rate | Product SSIM | Colour ΔE | Mask IoU | "
           "Product CLIP sim | Prompt margin | Seconds/image |",
           "|---|---|---|---|---|---|---|---|"]
    for v, r in t.items():
        out.append(f"| {v} ({r['n']}) | {pct(r['pass_rate'])} | {r['product_ssim']['mean']:.3f} | "
                   f"{r['colour_delta_e']['mean']:.2f} | {r['mask_iou']['mean']:.3f} | "
                   f"{r['product_similarity']['mean']:.3f} | {r['prompt_margin']['mean']:.3f} | "
                   f"{r['seconds_per_image']['mean']:.2f} |")
    th = d["thresholds"]
    out.append("")
    out.append("Calibrated thresholds (calibration split): " + ", ".join(
        f"{k} {th[k]}" for k in ("mask_iou", "product_similarity", "product_ssim",
                                 "colour_delta_e")) + f"; prompt_margin {th['prompt_margin']} "
               "(a priori). Failed checks on test: " + "; ".join(
                   f"{v}: {r['failed_checks']}" for v, r in t.items()) + ".")
    return out


def copy(d: dict) -> list[str]:
    out = ["| System prompt | First draft: any violation | Most common | Accepted first draft | "
           "Accepted after 1 repair | Fell back to template | Final violations | s/product |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in d["runs"].items():
        fd, pl = r["first_draft"], r["pipeline"]
        top = ", ".join(f"{k} {pct(v)}" for k, v in list(fd["violation_rate_by_rule"].items())[:3])
        out.append(f"| {name} | {pct(fd['any_violation_rate'])} | {top or '-'} | "
                   f"{pct(pl['accepted_first_draft'])} | {pct(pl['accepted_after_repair'])} | "
                   f"{pct(pl['fallback_to_template'])} | {pct(pl['final_violation_rate'])} | "
                   f"{pl['seconds_per_product']['mean']:.2f} |")
    out.append("")
    out.append(f"Provider `{d['provider']}`, {d['n_products']} test-split products.")
    return out


if __name__ == "__main__":
    for title, name, fn in [("Classification", "classification", classification),
                            ("Search", "search", search), ("Generation", "generation", generation),
                            ("Copy", "copy_local", copy)]:
        d = load(name)
        if d:
            print(f"### {title}\n")
            print("\n".join(fn(d)))
            print()

"""Zero-shot vs linear-probe attribute classification on frozen CLIP/SigLIP embeddings.

Writes results/classification.json. Run: python scripts/eval_classification.py
"""

from __future__ import annotations

import argparse
from collections import Counter

from cvlab.classify import LinearProbe, ZeroShotClassifier, accuracy, macro_f1
from cvlab.config import CACHE_DIR, MODELS
from cvlab.data import load_catalog, split_by_id
from cvlab.embeddings import load_embedder
from cvlab.evalutil import cached_image_embeddings, write_result

TOP_COLOURS = 10


def task_labels(products):
    colours = [c for c, _ in Counter(p.colour for p in products if p.colour != "Multi")
               .most_common(TOP_COLOURS)]
    return {
        "category": lambda p: p.article_type,
        "colour": lambda p: p.colour if p.colour in colours else None,
        "pattern": lambda p: p.pattern,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    products = load_catalog(n_max=args.n)
    train_idx, test_idx = split_by_id(products)
    tasks = task_labels(products)
    out = {"dataset": {"n_products": len(products), "n_train": len(train_idx),
                       "n_test": len(test_idx), "split": "by product id, 70/30"},
           "tasks": {}, "models": {}}

    for model_key in args.models:
        embedder = load_embedder(model_key)
        emb, secs = cached_image_embeddings(model_key, products, embedder)
        out["models"][model_key] = {"open_clip": list(MODELS[model_key]), "dim": int(emb.shape[1]),
                                    "embed_seconds_total": round(secs, 2),
                                    "embed_ms_per_image": round(1000 * secs / len(products), 2)}
        for task, fn in tasks.items():
            tr = [i for i in train_idx if fn(products[i])]
            te = [i for i in test_idx if fn(products[i])]
            y_tr = [fn(products[i]) for i in tr]
            y_te = [fn(products[i]) for i in te]
            labels = sorted(set(y_tr) | set(y_te))
            majority = Counter(y_tr).most_common(1)[0][0]
            zs = ZeroShotClassifier.build(embedder, task, labels)
            zs_pred = zs.predict(emb[te])
            probe = LinearProbe.fit(emb[tr], y_tr)
            lp_pred = probe.predict(emb[te])
            probe.save(CACHE_DIR / f"probe_{model_key}_{task}.json")
            out["tasks"].setdefault(task, {"n_classes": len(labels), "n_train": len(tr),
                                           "n_test": len(te), "labels": labels,
                                           "majority_baseline_acc": round(accuracy(
                                               y_te, [majority] * len(y_te)), 4)})
            out["tasks"][task][model_key] = {
                "zero_shot": {"accuracy": round(accuracy(y_te, zs_pred), 4),
                              "macro_f1": round(macro_f1(y_te, zs_pred), 4)},
                "linear_probe": {"accuracy": round(accuracy(y_te, lp_pred), 4),
                                 "macro_f1": round(macro_f1(y_te, lp_pred), 4)},
            }
            print(f"{model_key:15s} {task:9s} n_test={len(te):4d} "
                  f"zs acc={accuracy(y_te, zs_pred):.3f} f1={macro_f1(y_te, zs_pred):.3f} | "
                  f"probe acc={accuracy(y_te, lp_pred):.3f} f1={macro_f1(y_te, lp_pred):.3f}")
        del embedder
    print("wrote", write_result("classification", out))


if __name__ == "__main__":
    main()

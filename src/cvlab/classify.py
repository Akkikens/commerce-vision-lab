"""Attribute classification on top of frozen embeddings.

* Zero-shot: cosine similarity between the image embedding and prompt embeddings.
* Linear probe: multinomial logistic regression on image embeddings. Training uses
  scikit-learn (``ml`` extra); inference is plain numpy so the service does not
  need sklearn.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder

PROMPTS = {
    "category": ["a product photo of {}", "a photo of {}, an apparel product"],
    "colour": ["a photo of a {} coloured garment", "a {} product"],
    "pattern": ["a photo of a {} garment", "clothing with a {} pattern"],
}

# Natural-language names for dataset labels that read badly in a prompt.
LABEL_TEXT = {
    "Tshirts": "t-shirts",
    "Kurtas": "a kurta",
    "Innerwear Vests": "an innerwear vest",
    "Bra": "a bra",
    "Briefs": "briefs underwear",
    "Sarees": "a saree",
    "checked": "checked plaid",
    "solid": "plain solid colour",
}


def label_text(label: str) -> str:
    return LABEL_TEXT.get(label, label.lower())


@dataclass
class ZeroShotClassifier:
    labels: list[str]
    text_emb: np.ndarray  # (n_labels, dim), normalised

    @classmethod
    def build(cls, embedder: Embedder, task: str, labels: Sequence[str]) -> ZeroShotClassifier:
        templates = PROMPTS[task]
        rows = []
        for lab in labels:
            e = embedder.embed_texts([t.format(label_text(lab)) for t in templates]).mean(0)
            rows.append(e / np.linalg.norm(e))
        return cls(list(labels), np.stack(rows).astype(np.float32))

    def scores(self, img_emb: np.ndarray) -> np.ndarray:
        return img_emb @ self.text_emb.T

    def predict(self, img_emb: np.ndarray) -> list[str]:
        return [self.labels[i] for i in self.scores(img_emb).argmax(1)]


@dataclass
class LinearProbe:
    labels: list[str]
    coef: np.ndarray  # (n_labels, dim)
    intercept: np.ndarray  # (n_labels,)

    @classmethod
    def fit(cls, X: np.ndarray, y: Sequence[str], C: float = 10.0) -> LinearProbe:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(X, list(y))
        labels = [str(c) for c in clf.classes_]
        coef, intercept = clf.coef_, clf.intercept_
        if len(labels) == 2:  # sklearn stores binary problems as one row
            coef = np.vstack([-coef[0], coef[0]])
            intercept = np.array([-intercept[0], intercept[0]])
        return cls(labels, coef.astype(np.float32), intercept.astype(np.float32))

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef.T + self.intercept
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[str]:
        return [self.labels[i] for i in self.proba(X).argmax(1)]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"labels": self.labels, "coef": self.coef.tolist(),
                                    "intercept": self.intercept.tolist()}))

    @classmethod
    def load(cls, path: Path) -> LinearProbe:
        d = json.loads(path.read_text())
        return cls(d["labels"], np.asarray(d["coef"], np.float32),
                   np.asarray(d["intercept"], np.float32))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true))
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))

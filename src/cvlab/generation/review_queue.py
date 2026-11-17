"""Append-only JSONL human-review queue for generations that fail the gate."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


class ReviewQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def submit(self, item: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        rec = {"review_id": rid, "status": "pending", "created_at": time.time(), **item}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rid

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        latest: dict[str, dict] = {}
        for line in self.path.read_text().splitlines():
            rec = json.loads(line)
            latest[rec["review_id"]] = {**latest.get(rec["review_id"], {}), **rec}
        return list(latest.values())

    def pending(self) -> list[dict]:
        return [r for r in self._all() if r["status"] == "pending"]

    def decide(self, review_id: str, approved: bool, reviewer: str) -> None:
        if review_id not in {r["review_id"] for r in self._all()}:
            raise KeyError(review_id)
        rec = {"review_id": review_id, "status": "approved" if approved else "rejected",
               "reviewer": reviewer, "decided_at": time.time()}
        with self._lock, self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

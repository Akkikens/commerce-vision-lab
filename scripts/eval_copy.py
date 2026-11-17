"""Measure guardrail violations in LLM product copy before/after repair and fallback.

Writes results/copy.json. Default provider is a local Qwen2.5-0.5B-Instruct (no API key
needed). Run: python scripts/eval_copy.py [--provider local|anthropic|openai|stub]
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import numpy as np

from cvlab.config import SEED
from cvlab.copywriting.guardrails import check_copy
from cvlab.copywriting.providers import (
    MINIMAL_PROMPT,
    SYSTEM_PROMPT,
    get_provider,
    parse_json,
    user_prompt,
)
from cvlab.copywriting.writer import CopyWriter
from cvlab.data import load_catalog, split_by_id
from cvlab.evalutil import write_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="local")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prompts", nargs="+", default=["guarded", "minimal"])
    args = ap.parse_args()

    products = load_catalog()
    _, test_idx = split_by_id(products)
    rng = random.Random(SEED)
    sample = [products[i] for i in rng.sample(test_idx, args.n)]

    provider = get_provider(args.provider)
    runs = {}
    for prompt_name in args.prompts:
        system = SYSTEM_PROMPT if prompt_name == "guarded" else MINIMAL_PROMPT
        runs[prompt_name] = evaluate(provider, CopyWriter(provider, system_prompt=system),
                                     system, sample)
        print(prompt_name, runs[prompt_name]["first_draft"], runs[prompt_name]["pipeline"])
    out = {"provider": provider.name, "n_products": len(sample), "runs": runs,
           "note": "First-draft and pipeline passes are separate greedy generations, so the "
                   "pipeline's first draft equals the measured first draft. The repair "
                   "prompt always carries the guarded rules' violations as feedback."}
    print("wrote", write_result(f"copy_{args.provider}", out))


def evaluate(provider, writer, system, sample) -> dict:
    first_rules: Counter[str] = Counter()
    parse_failures = 0
    sources: Counter[str] = Counter()
    lat, examples = [], []
    final_bad = 0
    first_any = 0
    for p in sample:
        attrs = p.attributes()
        # Raw first draft, measured separately so parse failures are visible.
        raw = provider.complete(system, user_prompt(attrs))
        try:
            d = parse_json(raw)
        except (ValueError, KeyError):
            parse_failures += 1
            d = {"alt_text": "", "description": ""}
        chk = check_copy(d["alt_text"], d["description"], attrs)
        first_any += not chk.ok
        for r in chk.rules():
            first_rules[r] += 1
        if not chk.ok and len(examples) < 6:
            examples.append({"attributes": attrs, "draft": d,
                             "violations": [f"{v.rule}: {v.detail}" for v in chk.violations]})
        t = time.perf_counter()
        res = writer.write(attrs)
        lat.append(time.perf_counter() - t)
        sources[res.source] += 1
        final_bad += bool(check_copy(res.alt_text, res.description, attrs).violations)

    n = len(sample)
    return {
        "first_draft": {
            "parse_failure_rate": round(parse_failures / n, 4),
            "any_violation_rate": round(first_any / n, 4),
            "violation_rate_by_rule": {k: round(v / n, 4) for k, v in first_rules.most_common()},
        },
        "pipeline": {
            "accepted_first_draft": round(sources["llm"] / n, 4),
            "accepted_after_repair": round(sources["llm_repaired"] / n, 4),
            "fallback_to_template": round(sources["fallback_template"] / n, 4),
            "final_violation_rate": round(final_bad / n, 4),
            "seconds_per_product": {"mean": round(float(np.mean(lat)), 3),
                                    "p95": round(float(np.percentile(lat, 95)), 3)},
        },
        "example_violations": examples,
    }


if __name__ == "__main__":
    main()

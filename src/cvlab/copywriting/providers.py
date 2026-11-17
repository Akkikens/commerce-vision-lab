"""LLM providers behind one interface.

* ``StubProvider``: deterministic template, used in tests and as the safe fallback.
* ``LocalHFProvider``: a small local instruct model via transformers (no API key).
* ``AnthropicProvider`` / ``OpenAIProvider``: hosted models; need API credentials.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .guardrails import BRAND

SYSTEM_PROMPT = f"""You write e-commerce copy for {BRAND}, an activewear and apparel brand.
You are given structured product attributes. Use ONLY those facts.
Rules:
- Do not mention any fabric or material unless it is listed in the attributes.
- Do not make performance, care, or sustainability claims (e.g. moisture-wicking,
  breathable, waterproof, sustainable, organic, machine washable).
- Do not mention other brands. Write the brand name exactly as "{BRAND}" if you use it.
- Use the colour exactly as given; do not mention other colours.
- alt_text: one plain sentence under 125 characters describing what the image shows,
  including the colour and the product type. Do not start with "Image of".
- description: 2-3 sentences.
Return only JSON: {{"alt_text": "...", "description": "..."}}"""

# Ablation: same output format, none of the content rules.
MINIMAL_PROMPT = """Write e-commerce alt text and a 2-3 sentence product description for the
product described by the attributes. Return only JSON: {"alt_text": "...", "description": "..."}"""


def user_prompt(attrs: dict[str, str], feedback: list[str] | None = None) -> str:
    msg = "Product attributes:\n" + json.dumps(attrs, indent=1)
    if feedback:
        msg += ("\n\nYour previous draft broke these rules; fix them:\n- "
                + "\n- ".join(feedback))
    return msg


def parse_json(text: str) -> dict[str, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    d = json.loads(m.group(0))
    return {"alt_text": str(d.get("alt_text", "")), "description": str(d.get("description", ""))}


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class StubProvider:
    """Template copy built only from attributes, so it cannot hallucinate."""

    name = "stub-template"

    def complete(self, system: str, user: str) -> str:
        attrs = json.loads(user.split("Product attributes:\n", 1)[1].split("\n\n")[0])
        colour = attrs.get("colour", "").lower()
        cat = attrs.get("category", "product").lower()
        gender = attrs.get("gender", "")
        pattern = attrs.get("pattern")
        noun = f"{pattern} {cat}" if pattern else cat
        alt = f"{gender} {colour} {noun}".strip().capitalize()
        usage = attrs.get("usage", "").lower()
        desc = (f"{colour} {noun} from {BRAND}".capitalize().replace(BRAND.lower(), BRAND)
                + (f", designed for {usage} wear" if usage else "") + ".")
        return json.dumps({"alt_text": alt, "description": desc})


class LocalHFProvider:
    """Small local instruct model (default Qwen2.5-0.5B-Instruct) with greedy decoding."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..embeddings import pick_device

        self.torch = torch
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self.model = self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.name = f"local:{model_id}"

    def complete(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        n_in = enc["input_ids"].shape[1]
        return self.tok.decode(out[0, n_in:], skip_special_tokens=True)


class AnthropicProvider:
    """Claude via the official SDK. Credentials resolve from the environment
    (ANTHROPIC_API_KEY, or an ``ant auth login`` profile)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            # Route refused requests to a fallback model server-side.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIProvider:
    """OpenAI Responses API. The model id is required (set CVLAB_OPENAI_MODEL)."""

    def __init__(self, model: str | None = None):
        import openai

        self.model = model or os.environ.get("CVLAB_OPENAI_MODEL")
        if not self.model:
            raise ValueError("set CVLAB_OPENAI_MODEL to an OpenAI model id")
        self.client = openai.OpenAI()
        self.name = f"openai:{self.model}"

    def complete(self, system: str, user: str) -> str:
        resp = self.client.responses.create(model=self.model, instructions=system, input=user)
        return resp.output_text


def get_provider(name: str) -> LLMProvider:
    if name == "stub":
        return StubProvider()
    if name == "local":
        return LocalHFProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"unknown provider {name!r}")

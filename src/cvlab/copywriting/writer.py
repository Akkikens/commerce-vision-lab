"""Generate -> check -> repair once -> fall back to the template if still unsafe."""

from __future__ import annotations

from dataclasses import dataclass, field

from .guardrails import check_copy
from .providers import SYSTEM_PROMPT, LLMProvider, StubProvider, parse_json, user_prompt


@dataclass
class CopyResult:
    alt_text: str
    description: str
    source: str  # "llm", "llm_repaired" or "fallback_template"
    first_draft_violations: list[str] = field(default_factory=list)
    final_violations: list[str] = field(default_factory=list)
    needs_review: bool = False


class CopyWriter:
    def __init__(self, provider: LLMProvider, max_repairs: int = 1,
                 system_prompt: str = SYSTEM_PROMPT):
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_repairs = max_repairs
        self.fallback = StubProvider()

    def _draft(self, provider: LLMProvider, attrs: dict, feedback=None) -> dict[str, str]:
        try:
            return parse_json(provider.complete(self.system_prompt, user_prompt(attrs, feedback)))
        except (ValueError, KeyError):
            return {"alt_text": "", "description": ""}

    def write(self, attrs: dict[str, str]) -> CopyResult:
        draft = self._draft(self.provider, attrs)
        check = check_copy(draft["alt_text"], draft["description"], attrs)
        first = [f"{v.rule}: {v.detail}" for v in check.violations]
        if check.ok:
            return CopyResult(**draft, source="llm")
        for _ in range(self.max_repairs):
            draft = self._draft(self.provider, attrs, feedback=first)
            check = check_copy(draft["alt_text"], draft["description"], attrs)
            if check.ok:
                return CopyResult(**draft, source="llm_repaired", first_draft_violations=first)
        safe = self._draft(self.fallback, attrs)
        final = check_copy(safe["alt_text"], safe["description"], attrs)
        return CopyResult(**safe, source="fallback_template", first_draft_violations=first,
                          final_violations=[v.rule for v in final.violations],
                          needs_review=True)

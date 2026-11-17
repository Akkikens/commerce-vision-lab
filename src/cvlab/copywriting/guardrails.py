"""Rule-based guardrails for generated product copy.

The model only sees structured attributes (category, colour, gender, usage, pattern).
Anything it asserts beyond those (a fabric, a performance claim, a different colour, a
competitor's name) is unsupported and is flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Fictional brand used throughout the project. Not a real company.
BRAND = "Northlight"
BRAND_VARIANTS_BANNED = ["north light", "northlite", "NORTHLIGHT"]
COMPETITOR_TERMS = ["nike", "adidas", "puma", "reebok", "lululemon", "under armour",
                    "gymshark", "uniqlo", "zara", "h&m", "levi's", "levis"]

MATERIALS = ["cotton", "organic cotton", "polyester", "nylon", "spandex", "elastane", "lycra",
             "leather", "suede", "silk", "wool", "merino", "cashmere", "linen", "denim",
             "rayon", "viscose", "modal", "bamboo", "fleece", "jersey", "chiffon", "satin",
             "velvet", "mesh", "rubber", "canvas", "georgette", "crepe", "khadi"]
# Materials implied by the category itself.
CATEGORY_MATERIALS = {"Jeans": {"denim"}}

UNSUPPORTED_CLAIMS = ["sustainable", "eco-friendly", "eco friendly", "organic", "recycled",
                      "waterproof", "water-resistant", "moisture-wicking", "sweat-wicking",
                      "breathable", "anti-odor", "anti-odour", "uv protection", "upf",
                      "hypoallergenic", "wrinkle-free", "machine washable", "4-way stretch",
                      "four-way stretch", "best-selling", "bestselling", "award-winning"]

COLOURS = ["black", "white", "blue", "navy", "brown", "grey", "gray", "red", "green", "pink",
           "purple", "beige", "yellow", "maroon", "orange", "olive", "cream", "teal",
           "turquoise", "gold", "silver", "khaki", "tan", "lavender", "peach", "magenta",
           "mustard", "charcoal", "off white", "multi"]
# Words that legitimately co-occur with a colour attribute.
COLOUR_SYNONYMS = {"grey": {"gray", "charcoal"}, "navy blue": {"navy", "blue"},
                   "off white": {"white", "cream"}, "multi": set(COLOURS),
                   "grey melange": {"grey", "gray"}}

ALT_TEXT_MAX = 125


@dataclass
class Violation:
    rule: str
    detail: str


@dataclass
class CheckResult:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def rules(self) -> list[str]:
        return sorted({v.rule for v in self.violations})


def _has(term: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, re.I) is not None


def check_copy(alt_text: str, description: str, attrs: dict[str, str]) -> CheckResult:
    res = CheckResult()
    text = f"{alt_text}\n{description}"
    allowed_materials = {m.lower() for m in attrs.get("materials", "").split(",") if m.strip()}
    allowed_materials |= CATEGORY_MATERIALS.get(attrs.get("category", ""), set())

    for m in MATERIALS:
        if _has(m, text) and m not in allowed_materials:
            res.violations.append(Violation("hallucinated_material", m))
    for c in UNSUPPORTED_CLAIMS:
        if _has(c, text):
            res.violations.append(Violation("unsupported_claim", c))
    for b in COMPETITOR_TERMS:
        if _has(b, text):
            res.violations.append(Violation("competitor_term", b))
    for v in BRAND_VARIANTS_BANNED:
        if re.search(re.escape(v), text) and v != BRAND:
            res.violations.append(Violation("brand_style", v))

    colour = attrs.get("colour", "").lower()
    ok_colours = set(colour.split()) | {colour} | COLOUR_SYNONYMS.get(colour, set())
    for c in COLOURS:
        if _has(c, text) and c not in ok_colours:
            res.violations.append(Violation("colour_mismatch", f"{c} (attribute: {colour})"))

    a = alt_text.strip()
    if len(a) > ALT_TEXT_MAX:
        res.violations.append(Violation("alt_text_length", f"{len(a)} > {ALT_TEXT_MAX}"))
    if re.match(r"(image|picture|photo) of\b", a, re.I):
        res.violations.append(Violation("alt_text_style", "starts with 'image of'"))
    if colour and not any(_has(c, a) for c in ok_colours if c):
        res.violations.append(Violation("alt_text_missing_colour", colour))
    if not a:
        res.violations.append(Violation("empty", "alt_text"))
    if not description.strip():
        res.violations.append(Violation("empty", "description"))
    return res

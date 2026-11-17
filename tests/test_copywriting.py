import json

from cvlab.copywriting.guardrails import ALT_TEXT_MAX, check_copy
from cvlab.copywriting.providers import StubProvider, parse_json
from cvlab.copywriting.writer import CopyWriter

ATTRS = {"category": "Tops", "colour": "Black", "gender": "Women", "usage": "Sports"}


def rules(alt, desc, attrs=ATTRS):
    return check_copy(alt, desc, attrs).rules()


def test_clean_copy_passes():
    assert rules("Women's black sports top", "A black top for training days.") == []


def test_hallucinated_material_flagged_unless_attribute_or_implied():
    assert "hallucinated_material" in rules("Black top", "Soft cotton jersey top.")
    assert rules("Black top", "Soft cotton top.", {**ATTRS, "materials": "cotton"}) == []
    jeans = {"category": "Jeans", "colour": "Blue"}
    assert "hallucinated_material" not in rules("Blue jeans", "Classic denim jeans.", jeans)


def test_unsupported_claims_and_competitors():
    r = rules("Black top", "Moisture-wicking and sustainable. Better than Lululemon.")
    assert {"unsupported_claim", "competitor_term"} <= set(r)


def test_colour_mismatch_and_synonyms():
    assert "colour_mismatch" in rules("Black top", "A black top with red trim.")
    navy = {"category": "Shirts", "colour": "Navy Blue"}
    assert rules("Navy blue shirt", "A navy shirt.", navy) == []
    assert rules("Grey tee", "A gray tee.", {"category": "Tshirts", "colour": "Grey"}) == []


def test_word_boundaries_avoid_false_positives():
    # "silky" is not "silk", "mesh-like" is not "mesh", "organically" is not "organic"
    assert rules("Black top", "A silky feel and mesh-like look, organically shaped.") == []


def test_alt_text_rules():
    assert "alt_text_length" in rules("Black " + "x" * ALT_TEXT_MAX, "ok")
    assert "alt_text_style" in rules("Image of a black top", "ok")
    assert "alt_text_missing_colour" in rules("A sports top", "A black top.")


def test_parse_json_extracts_object_from_chatter():
    assert parse_json('Sure! {"alt_text": "a", "description": "b"} hope that helps') == {
        "alt_text": "a", "description": "b"}


class ScriptedProvider:
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, system, user):
        self.calls.append(user)
        return self.outputs.pop(0)


def j(alt, desc):
    return json.dumps({"alt_text": alt, "description": desc})


def test_writer_accepts_clean_first_draft():
    res = CopyWriter(ScriptedProvider([j("Black sports top", "A black top.")])).write(ATTRS)
    assert res.source == "llm" and not res.needs_review


def test_writer_repairs_with_violation_feedback():
    p = ScriptedProvider([j("Black top", "Organic cotton top."), j("Black top", "A black top.")])
    res = CopyWriter(p).write(ATTRS)
    assert res.source == "llm_repaired"
    assert "hallucinated_material: cotton" in p.calls[1]
    assert res.first_draft_violations


def test_writer_falls_back_to_template_and_flags_review():
    p = ScriptedProvider([j("Red top", "Silk."), "not json at all"])
    res = CopyWriter(p).write(ATTRS)
    assert res.source == "fallback_template" and res.needs_review
    assert res.final_violations == []
    assert check_copy(res.alt_text, res.description, ATTRS).ok


def test_stub_provider_output_always_passes_guardrails():
    for attrs in [ATTRS, {"category": "Jeans", "colour": "Navy Blue", "pattern": "solid"},
                  {"category": "Heels", "colour": "Off White", "gender": "Women"}]:
        d = parse_json(StubProvider().complete("", "Product attributes:\n" + json.dumps(attrs)))
        assert check_copy(d["alt_text"], d["description"], attrs).ok, (attrs, d)

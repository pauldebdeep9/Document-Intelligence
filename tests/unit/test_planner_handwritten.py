"""data/gold/planner/handwritten.json: the hand-written planner cases,
checked for structure only (no model). Paraphrases the gold does not use,
near-misses and unsupported aggregations that must route "none", and the two
truncation shapes AG-02 left fail-open -- the check that a planner prompt was
not tuned to the gold's exact wording."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isc.aggregate.plan import QueryPlanRaw, validate_plan

ROOT = Path(__file__).resolve().parents[2]
HANDWRITTEN = ROOT / "data" / "gold" / "planner" / "handwritten.json"
QUESTIONS = ROOT / "data" / "gold" / "retrieval" / "questions.json"
PROMPT = ROOT / "config" / "prompts" / "aggregate" / "query_plan.v1.md"
MASTERS = ROOT / "data" / "masters"

DATA = json.loads(HANDWRITTEN.read_text())
CASES = {c["id"]: c for c in DATA["cases"]}
FIELDS = ("supplier", "part_number", "currency")


def test_ids_and_classes():
    assert [c["id"] for c in DATA["cases"]] == [f"hw_{i:02d}" for i in range(1, 21)]
    by_class: dict[str, list[str]] = {}
    for c in DATA["cases"]:
        by_class.setdefault(c["class"], []).append(c["id"])
    assert by_class == {
        "paraphrase": [f"hw_{i:02d}" for i in range(1, 10)],
        "near_miss": [f"hw_{i:02d}" for i in range(10, 16)],
        "unsupported": ["hw_16", "hw_17", "hw_18"],
        "truncation": ["hw_19", "hw_20"],
    }


def test_no_case_repeats_a_gold_question():
    gold = {q["text"] for q in json.loads(QUESTIONS.read_text())["questions"]}
    assert [c["id"] for c in DATA["cases"] if c["text"] in gold] == []


@pytest.mark.parametrize("cid", [c["id"] for c in DATA["cases"]
                                 if c["expected"]["operation"] == "none"])
def test_none_case_has_no_parameters(cid):
    case = CASES[cid]
    assert {f: case["expected"][f] for f in FIELDS} == dict.fromkeys(FIELDS)
    assert case["expected_supplier_ids"] == []


@pytest.mark.parametrize("cid", [c["id"] for c in DATA["cases"]
                                 if c["expected"]["operation"] != "none"])
def test_aggregate_case_is_verbatim_and_validates(cid):
    case = CASES[cid]
    exp, text = case["expected"], case["text"]
    if exp["supplier"] is not None:
        assert exp["supplier"].casefold() in text.casefold(), "supplier not in the text"
    for field in ("part_number", "currency"):
        if exp[field] is not None:
            assert exp[field] in text, f"{field} not in the text"
    plan, reason = validate_plan(QueryPlanRaw(**exp), text, MASTERS)
    assert plan is not None, reason
    assert plan.supplier_ids == tuple(case["expected_supplier_ids"])


def test_resolved_supplier_ids():
    assert CASES["hw_04"]["expected_supplier_ids"] == ["V102337"]
    assert CASES["hw_05"]["expected_supplier_ids"] == ["V100781", "V100782"]
    assert CASES["hw_08"]["expected_supplier_ids"] == ["V103014"]
    assert CASES["hw_20"]["expected_supplier_ids"] == ["V100781"]


def test_provenance_forbids_copying_into_a_prompt():
    note = DATA["provenance"]["note"]
    assert "never copy" in note and "prompt" in note


def test_no_case_text_is_in_the_planner_prompt():
    prompt = PROMPT.read_text().casefold()
    assert [c["id"] for c in DATA["cases"] if c["text"].casefold() in prompt] == []

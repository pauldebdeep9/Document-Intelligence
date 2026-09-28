"""scripts/gen_planner_gold.py: the expected plan for every retrieval-gold
question, derived from gold structure. Derived cases come from the
generator's own functions (no files written); the committed cases.json is
checked against them byte for byte."""

from __future__ import annotations

import hashlib
import json

import pytest

from isc.aggregate.plan import QueryPlanRaw, validate_plan
from scripts.gen_planner_gold import MASTERS, OUT, QUESTIONS, build, check, render

AG, PNEU = "V100781", "V100782"


@pytest.fixture(scope="module")
def built():
    return build()


@pytest.fixture(scope="module")
def cases(built):
    return {c["id"]: c for c in built["cases"]}


def _plan(case):
    e = case["expected"]
    return e["operation"], e["supplier"], e["part_number"], e["currency"]


def test_counts(cases):
    ops = [c["expected"]["operation"] for c in cases.values()]
    assert len(ops) == 56
    assert ops.count("none") == 47
    assert sorted(i for i, c in cases.items() if c["expected"]["operation"] == "total_spend") == [
        "q_am_04", "q_cd_01", "q_cd_02", "q_cd_03", "q_cd_04"]
    assert sorted(i for i, c in cases.items() if c["expected"]["operation"] == "part_prices") == [
        "q_cd_05", "q_cd_06", "q_cd_07", "q_cd_08"]


def test_single_legal_entity_total(cases):
    assert _plan(cases["q_cd_04"]) == ("total_spend", "Kestrel Industrial AG", None, "USD")
    assert cases["q_cd_04"]["expected_supplier_ids"] == [AG]


def test_ambiguous_total_is_the_common_prefix_with_no_currency(cases):
    assert _plan(cases["q_am_04"]) == ("total_spend", "Kestrel Industrial", None, None)
    assert cases["q_am_04"]["expected_supplier_ids"] == [AG, PNEU]


def test_part_prices(cases):
    assert _plan(cases["q_cd_07"]) == ("part_prices", None, "ENC-INC-1024", None)


@pytest.mark.parametrize("qid", ["q_am_01", "q_am_02", "q_am_03"])
def test_kestrel_near_misses_route_none(cases, qid):
    assert _plan(cases[qid]) == ("none", None, None, None)
    assert cases[qid]["expected_supplier_ids"] == []


def test_provenance_pins_the_current_questions_file():
    committed = json.loads(OUT.read_text())
    assert committed["provenance"]["questions_sha256"] == \
        hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(), \
        "questions.json changed -- run `make planner-gold`"


def test_regeneration_is_byte_identical(built):
    assert render(built) == OUT.read_text(), "cases.json is stale or hand-edited -- regenerate"


def test_every_expected_plan_passes_validate_plan(built, cases):
    assert check(built) == []
    for case in cases.values():
        if case["expected"]["operation"] == "none":
            continue
        plan, reason = validate_plan(QueryPlanRaw(**case["expected"]), case["text"], MASTERS)
        assert plan is not None, f"{case['id']}: {reason}"
        assert list(plan.supplier_ids) == case["expected_supplier_ids"], case["id"]

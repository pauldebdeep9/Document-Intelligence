"""aggregate/resolve.py and aggregate/plan.py: a question's supplier mention
resolves to every candidate, and a plan is only trusted when every
identifier in it was copied from the question."""

from __future__ import annotations

import json

import pytest

from isc.aggregate.plan import Operation, QueryPlanRaw, validate_plan
from isc.aggregate.resolve import resolve_mention
from isc.extract.masters import supplier_id_for_printed_name
from isc.llm.schema import to_strict_schema
from tests.aggregate_world import MASTERS

AG, PNEU = "V100781", "V100782"


# -- resolve_mention: a question wants every candidate ----------------------

@pytest.mark.parametrize("mention, expected", [
    ("Kestrel Industrial", (AG, PNEU)),              # the ambiguous case: both
    ("Kestrel", (AG, PNEU)),
    ("Kestrel Industrial AG", (AG,)),                # full legal name: that one only
    ("kestrel   industrial ag", (AG,)),              # case/whitespace-insensitive
    ("Kestrel Industrial Pneumatics", (PNEU,)),
    ("Kestrel Industrial Pneumatics GmbH", (PNEU,)),
    ("Keyence", ("V103014",)),
    ("Fastenal Industrial", ("V100234", "V100235")),
    ("Kes", ()),                                     # whole tokens only, no substrings
    ("Siemens", ()),
    ("", ()),
    (None, ()),
])
def test_resolve_mention(mention, expected):
    assert resolve_mention(mention, MASTERS) == expected


def test_same_string_resolves_differently_printed_vs_asked():
    """The reason resolve_mention() exists: the extraction-side policy
    (exact-or-nothing, for a name PRINTED on a document) maps "Kestrel
    Industrial" to AG alone, because both normalise to "kestrel
    industrial". Reusing it for a question would silently drop the
    Pneumatics GmbH -- q_am_*'s whole failure mode."""
    assert supplier_id_for_printed_name("Kestrel Industrial", MASTERS) == AG
    assert resolve_mention("Kestrel Industrial", MASTERS) == (AG, PNEU)


def test_printed_name_needs_a_unique_match():
    assert supplier_id_for_printed_name("Omron Electronics Asia", MASTERS) == "V102337"
    assert supplier_id_for_printed_name("Omron", MASTERS) is None
    assert supplier_id_for_printed_name(None, MASTERS) is None


def test_printed_name_that_normalises_to_two_masters_is_a_miss(tmp_path):
    """No corpus supplier pair normalises to the same string, so the
    unique-match guard needs its own master to be tested at all."""
    (tmp_path / "suppliers.json").write_text(json.dumps([
        {"supplier_id": "V1", "name": "Acme Tools GmbH", "country": "DE"},
        {"supplier_id": "V2", "name": "Acme Tools AG", "country": "CH"},
    ]))
    assert supplier_id_for_printed_name("Acme Tools", tmp_path) is None
    assert supplier_id_for_printed_name("Acme Tools GmbH", tmp_path) == "V1"
    assert resolve_mention("Acme Tools", tmp_path) == ("V1", "V2")


# -- validate_plan: every identifier must come from the question ------------

def _v(question, **raw):
    return validate_plan(QueryPlanRaw(**raw), question, MASTERS)


def test_none_falls_through():
    plan, reason = _v("Who is the buyer on PO 4513180299?", operation="none")
    assert plan is None and "not an aggregate" in reason


def test_valid_total_spend():
    plan, reason = _v("What did we spend with Omron Electronics Asia in total, in SGD?",
                      operation="total_spend", supplier="Omron Electronics Asia", currency="SGD")
    assert reason == ""
    assert plan.operation is Operation.TOTAL_SPEND
    assert plan.supplier_ids == ("V102337",)
    assert plan.currency == "SGD"


def test_ambiguous_mention_keeps_every_candidate():
    plan, _ = _v("How much did we spend with Kestrel Industrial in total?",
                 operation="total_spend", supplier="Kestrel Industrial")
    assert plan.supplier_ids == (AG, PNEU)


def test_planner_completing_a_supplier_name_is_rejected():
    """The planner 'helpfully' expanding the user's words changes the
    question -- here it would collapse the ambiguous case to one entity."""
    plan, reason = _v("How much did we spend with Kestrel Industrial in total?",
                      operation="total_spend", supplier="Kestrel Industrial AG")
    assert plan is None and "verbatim" in reason


def test_inferred_currency_is_rejected():
    plan, reason = _v("What did we spend with Omron Electronics Asia in total?",
                      operation="total_spend", supplier="Omron Electronics Asia", currency="SGD")
    assert plan is None and "currency" in reason


def test_hallucinated_part_number_is_rejected():
    plan, reason = _v("What did we pay for the terminal blocks across our orders?",
                      operation="part_prices", part_number="TRM-BLK-2P5")
    assert plan is None and "verbatim" in reason


def test_part_number_must_be_a_whole_token_of_the_question():
    plan, _ = _v("What did we pay for part TRM-BLK-2P5 across our purchase orders?",
                 operation="part_prices", part_number="TRM-BLK-2")
    assert plan is None


def test_part_number_case_is_normalised():
    plan, _ = _v("what did we pay for part trm-blk-2p5 across our purchase orders?",
                 operation="part_prices", part_number="trm-blk-2p5")
    assert plan is not None and plan.part_number == "TRM-BLK-2P5"


def test_supplier_not_in_master_falls_through():
    plan, reason = _v("What did we spend with Siemens in total?",
                      operation="total_spend", supplier="Siemens")
    assert plan is None and "master" in reason


@pytest.mark.parametrize("raw", [
    dict(operation="total_spend"),                      # no supplier
    dict(operation="part_prices", supplier="Keyence"),  # no part number
])
def test_operation_missing_its_required_parameter(raw):
    plan, _ = _v("What did we spend with Keyence in total?", **raw)
    assert plan is None


def test_truncated_supplier_mention_is_rejected():
    """The mirror of completing a name: dropping the user's "AG" widens one
    entity to both Kestrels, and every word left is still in the question."""
    plan, reason = _v("What did we spend with Kestrel Industrial AG in total, in USD?",
                      operation="total_spend", supplier="Kestrel Industrial", currency="USD")
    assert plan is None and reason


def test_truncation_toward_a_longer_master_name_is_rejected():
    plan, reason = _v("What did we spend with Kestrel Industrial Pneumatics in total?",
                      operation="total_spend", supplier="Kestrel Industrial")
    assert plan is None and reason


def test_supplier_mention_must_end_on_a_word_boundary():
    plan, reason = _v("What did we spend with Kestrel Industrials in total?",
                      operation="total_spend", supplier="Kestrel Industrial")
    assert plan is None and reason


def test_currency_must_be_written_as_a_code_in_the_question():
    """An English word that is also an ISO code is not a currency. Fail-safe
    cost: a lower-case code ("sgd") falls through to the chunk path too."""
    plan, reason = _v("What did we spend with Keyence across all orders?",
                      operation="total_spend", supplier="Keyence", currency="ALL")
    assert plan is None and reason
    plan, reason = _v("What did we spend with Keyence in sgd?",
                      operation="total_spend", supplier="Keyence", currency="SGD")
    assert plan is None and reason


def test_part_number_shape_is_checked():
    plan, reason = _v("What did we pay for part ABC across our purchase orders?",
                      operation="part_prices", part_number="ABC")
    assert plan is None and "shape" in reason


def test_currency_shape_is_checked():
    plan, reason = _v("What did we spend with Keyence in SGDX?",
                      operation="total_spend", supplier="Keyence", currency="SGDX")
    assert plan is None and "3-letter" in reason


def test_total_spend_with_a_part_number_is_rejected():
    """total_spend sums whole-order totals; it cannot restrict to one part,
    so accepting the plan would answer a different question."""
    plan, reason = _v("What did we spend with Keyence on part TRM-BLK-2P5 in total?",
                      operation="total_spend", supplier="Keyence", part_number="TRM-BLK-2P5")
    assert plan is None and reason


def test_plan_schema_is_strict_mode_compatible():
    schema = to_strict_schema(QueryPlanRaw)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"operation", "supplier", "part_number", "currency"}
    assert set(schema["properties"]["operation"]["enum"]) == {
        "total_spend", "part_prices", "none"}

"""aggregate/ end to end on small worlds: selection, the confidence gate,
the deterministic renderer, and the orchestrator hand-off."""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from isc.aggregate.answerer import RecordAnswerer
from isc.aggregate.execute import execute
from isc.aggregate.plan import QueryPlanRaw, validate_plan
from isc.aggregate.source import visible_records
from isc.answer.citations import _split_sentences, bind_citations, verify_attribution
from isc.answer.orchestrator import AnswerOrchestrator
from isc.common.confidence import Thresholds
from isc.extract.masters import supplier_ids_by_name
from isc.models.acl import Principal
from isc.models.answer import AbstentionReason
from tests.aggregate_world import MASTERS, Line, PlanChat, Po, build, plan, settings

D = Decimal
ANYONE = Principal(id="u_any")
OMRON, KEYENCE = "Omron Electronics Asia", "Keyence Singapore Pte Ltd"
AG, PNEU = "Kestrel Industrial AG", "Kestrel Industrial Pneumatics GmbH"


def _ask(tmp_path, pos, the_plan, question, principal=ANYONE):
    docs, store = build(tmp_path, pos)
    return RecordAnswerer(PlanChat(the_plan), docs, store, settings()).try_answer(
        question, principal)


def _omron_q(cur="SGD"):
    return f"What did we spend with {OMRON} in total, in {cur}?"


# -- selection ---------------------------------------------------------------

def test_sums_only_the_requested_supplier_and_currency(tmp_path):
    pos = [
        Po("d1", "4500000001", OMRON, "V102337", "SGD", D("100.10")),
        Po("d2", "4500000002", OMRON, None, "SGD", D("200.20")),   # no vendor code printed
        Po("d3", "4500000003", OMRON, "V102337", "USD", D("999.99")),
        Po("d4", "4500000004", KEYENCE, "V103014", "SGD", D("5000.00")),
    ]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert not ans.abstained and ans.route == "records"
    assert "300.30 SGD" in ans.text.splitlines()[0]
    # d2 has no supplier_id: selected by its printed name via the master.
    assert "4500000002" in ans.text
    # Other currency: counted, never its amount; other supplier: absent.
    assert "999.99" not in ans.text and "1 order in USD" in ans.text
    assert "5,000.00" not in ans.text and "4500000004" not in ans.text


def test_decimal_arithmetic_is_exact(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("0.10")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", D("0.20"))]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert ": 0.30 SGD" in ans.text   # float would give 0.30000000000000004


def test_ambiguous_mention_reports_each_supplier_and_never_adds_them(tmp_path):
    pos = [Po("d1", "4500000001", AG, "V100781", "EUR", D("20298.30")),
           Po("d2", "4500000002", PNEU, "V100782", "EUR", D("1325721.35"))]
    q = "How much did we spend with Kestrel Industrial in total?"
    ans = _ask(tmp_path, pos, plan("total_spend", "Kestrel Industrial"), q)
    assert "matches 2 suppliers" in ans.text
    assert f"{AG} (V100781), EUR: 20,298.30 EUR" in ans.text
    assert f"{PNEU} (V100782), EUR: 1,325,721.35 EUR" in ans.text
    assert "1,346,019.65" not in ans.text   # the conflated sum q_cd_04 produced


def test_ambiguous_mention_with_one_visible_candidate_still_says_so(tmp_path):
    pos = [Po("d1", "4500000001", AG, "V100781", "EUR", D("10.00"))]
    q = "How much did we spend with Kestrel Industrial in total?"
    ans = _ask(tmp_path, pos, plan("total_spend", "Kestrel Industrial"), q)
    assert "matches 2 suppliers" in ans.text
    assert "Only 1 of the 2 has orders visible to you." in ans.text


def test_no_currency_in_question_gives_per_currency_totals(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("1.00")),
           Po("d2", "4500000002", OMRON, "V102337", "EUR", D("2.00"))]
    q = f"What did we spend with {OMRON} in total?"
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON), q)
    assert "not converted" in ans.text
    assert "EUR: 2.00 EUR" in ans.text and "SGD: 1.00 SGD" in ans.text
    assert "3.00" not in ans.text


# -- the confidence gate -----------------------------------------------------

def test_review_band_value_is_included_and_named(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", D("5.00"), total_conf=0.70)]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert ": 15.00 SGD" in ans.text
    assert re.search(r"PO 4500000002 order total is included but is still in the "
                     r"extraction review queue \(confidence 0\.70\)", ans.text)
    assert ans.confidence.score == 0.70   # weakest contributing value


def test_low_confidence_value_is_excluded_and_named(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", D("5.00"), total_conf=0.40)]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert ": 10.00 SGD" in ans.text
    assert "Not included: PO 4500000002, because its extracted total is below" in ans.text


def test_route_is_the_weakest_selection_field_not_just_the_value(tmp_path):
    """A confident total under an unreadable currency is not a confident
    SGD total."""
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", D("5.00"), currency_conf=0.40)]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert ": 10.00 SGD" in ans.text and "Not included: PO 4500000002" in ans.text


def test_absent_total_is_named_not_treated_as_zero(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", None)]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert "across the 1 purchase order visible to you: 10.00 SGD" in ans.text
    assert "Not included: PO 4500000002, because no order total is printed" in ans.text


def test_value_not_found_in_indexed_text_is_not_cited_to_the_wrong_place(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", D("5.00"), printed_total=D("7.00"))]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert ": 10.00 SGD" in ans.text
    assert "could not be located in the indexed text" in ans.text


def test_unextracted_visible_document_is_reported(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
           Po("d2", "4500000002", OMRON, "V102337", "SGD", D("5.00"), extracted=False)]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert "Not checked: 1 purchase order visible to you has no extracted record" in ans.text


def test_nothing_summable_abstains_instead_of_answering_zero(tmp_path):
    pos = [Po("d1", "4500000001", OMRON, "V102337", "SGD", None)]
    ans = _ask(tmp_path, pos, plan("total_spend", OMRON, currency="SGD"), _omron_q())
    assert ans.abstained and ans.abstention_reason is AbstentionReason.INSUFFICIENT_CONTEXT
    assert ans.route == "records"


def test_every_visible_matching_order_is_accounted_for_exactly_once(tmp_path):
    """The executor's bookkeeping, as identities: every visible Omron SGD
    order is either included or excluded-with-a-reason, never both, never
    neither -- whatever mix of gate outcomes the orders hit."""
    omron_sgd = [
        Po("d_ok", "4500000001", OMRON, "V102337", "SGD", D("10.00")),
        Po("d_review", "4500000002", OMRON, "V102337", "SGD", D("20.00"), total_conf=0.70),
        Po("d_low", "4500000003", OMRON, "V102337", "SGD", D("30.00"), total_conf=0.40),
        Po("d_absent", "4500000004", OMRON, "V102337", "SGD", None),
        Po("d_unprinted", "4500000005", OMRON, "V102337", "SGD", D("40.00"),
           printed_total=D("41.00")),
        Po("d_no_code", "4500000006", OMRON, None, "SGD", D("50.00")),
    ]
    eur = Po("d_eur", "4500000007", OMRON, "V102337", "EUR", D("60.00"))
    keyence = Po("d_keyence", "4500000008", KEYENCE, "V103014", "SGD", D("70.00"))
    docs, store = build(tmp_path, [*omron_sgd, eur, keyence])
    the_plan, reason = validate_plan(
        QueryPlanRaw(operation="total_spend", supplier=OMRON, currency="SGD"),
        _omron_q(), MASTERS)
    assert the_plan is not None, reason

    result = execute(the_plan, visible_records(ANYONE, docs, store), Thresholds(),
                     frozenset({"accept", "review"}), MASTERS)

    included_ids = {r.document_id for r in result.included}
    excluded_ids = {r.document_id for r in result.excluded}
    assert included_ids & excluded_ids == set()
    assert included_ids | excluded_ids == {po.doc_id for po in omron_sgd}
    assert all(r.reason for r in result.excluded)
    assert not any(r.reason for r in result.included)
    assert result.other_currencies == {"EUR": 1}
    assert keyence.doc_id not in included_ids | excluded_ids


# -- part prices ---------------------------------------------------------------

def test_part_prices_lists_every_visible_priced_line(tmp_path):
    pos = [
        Po("d1", "4500000001", OMRON, "V102337", "SGD", D("1"), lines=[
            Line(10, "TRM-BLK-2P5", D("979.93")), Line(20, "PSU-24V-10A", D("5.00")),
            Line(30, "TRM-BLK-2P5", None)]),
        Po("d2", "4500000002", KEYENCE, "V103014", "SGD", D("1"), lines=[
            Line(10, "TRM-BLK-2P5", D("2106.20"))]),
    ]
    q = "What did we pay for part TRM-BLK-2P5 across our purchase orders?"
    ans = _ask(tmp_path, pos, plan("part_prices", part="TRM-BLK-2P5"), q)
    assert "PO 4500000001 line 10: unit price 979.93 SGD" in ans.text
    assert "PO 4500000002 line 10: unit price 2,106.20 SGD" in ans.text
    assert "5.00" not in ans.text
    assert "Not included: PO 4500000001 line 30, because no unit price is printed" in ans.text
    # line citations resolve to the table chunk covering that line
    assert {c.chunk_id for c in ans.citations} >= {"chk_d1_1", "chk_d2_1"}


# -- grounding: the renderer's output passes the same checks as a draft -------

_GROUNDING_CASES = {
    "single_supplier": (
        [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("1.00")),
         Po("d2", "4500000002", OMRON, None, "SGD", D("2.00"), total_conf=0.7),
         Po("d3", "4500000003", OMRON, "V102337", "SGD", None)],
        plan("total_spend", OMRON, currency="SGD"), _omron_q()),
    "ambiguous_supplier": (
        [Po("d1", "4500000001", AG, "V100781", "EUR", D("1.00")),
         Po("d2", "4500000002", PNEU, "V100782", "EUR", D("2.00"), total_conf=0.7),
         Po("d3", "4500000003", PNEU, None, "EUR", None)],
        plan("total_spend", "Kestrel Industrial"),
        "How much did we spend with Kestrel Industrial in total?"),
    "part_prices": (
        [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("1"), lines=[
            Line(10, "TRM-BLK-2P5", D("3.00")), Line(20, "TRM-BLK-2P5", None)]),
         Po("d2", "4500000002", KEYENCE, None, "SGD", D("1"), lines=[
            Line(10, "TRM-BLK-2P5", D("4.00"), price_conf=0.7)])],
        plan("part_prices", part="TRM-BLK-2P5"),
        "What did we pay for part TRM-BLK-2P5 across our purchase orders?"),
}


@pytest.mark.parametrize("case", sorted(_GROUNDING_CASES))
def test_rendered_text_binds_and_verifies(tmp_path, case):
    pos, the_plan, q = _GROUNDING_CASES[case]
    ans = _ask(tmp_path, pos, the_plan, q)
    assert not ans.abstained, ans.text
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", ans.text)}
    assert markers == set(range(1, len(ans.supporting) + 1))   # all resolve, none unused
    assert len(bind_citations(ans.text, ans.supporting)) == len(ans.supporting)
    attribution = verify_attribution(ans.text, ans.supporting, supplier_ids_by_name(MASTERS))
    assert not attribution.mismatches
    # one sentence per line, so a PO's number is only ever vouched for by its own chunks
    assert len(_split_sentences(ans.text)) == len(ans.text.splitlines())
    for line in ans.text.splitlines():
        m = re.match(r"PO (\d+)", line)
        if m:
            cited = {ans.supporting[int(n) - 1].chunk.filters["po_number"]
                     for n in re.findall(r"\[(\d+)\]", line)}
            assert cited == {m.group(1)}


# -- orchestrator hand-off -----------------------------------------------------

class _ExplodingRetriever:
    def retrieve(self, question, principal):
        raise AssertionError("chunk path must not run when the records path answered")


class _RecordingRetriever:
    def __init__(self):
        self.called = False

    def retrieve(self, question, principal):
        self.called = True
        return []


def test_orchestrator_uses_records_answer_and_skips_retrieval(tmp_path):
    docs, store = build(tmp_path, [Po("d1", "4500000001", OMRON, "V102337", "SGD", D("1.00"))])
    s = settings()
    agg = RecordAnswerer(PlanChat(plan("total_spend", OMRON, currency="SGD")), docs, store, s)
    ans = AnswerOrchestrator(_ExplodingRetriever(), PlanChat("unused"), s,
                             aggregator=agg).ask(_omron_q(), ANYONE)
    assert ans.route == "records" and not ans.abstained


def test_orchestrator_falls_through_when_plan_is_none(tmp_path):
    docs, store = build(tmp_path, [])
    s = settings()
    agg = RecordAnswerer(PlanChat(plan("none")), docs, store, s)
    retriever = _RecordingRetriever()
    ans = AnswerOrchestrator(retriever, PlanChat("unused"), s, aggregator=agg).ask(
        "Who is the buyer on PO 4513180299?", ANYONE)
    assert retriever.called and ans.route == "chunks"
    assert ans.abstention_reason is AbstentionReason.NO_RESULTS


def test_planner_that_never_validates_falls_through(tmp_path):
    docs, store = build(tmp_path, [])
    agg = RecordAnswerer(PlanChat("not json at all"), docs, store, settings())
    assert agg.try_answer(_omron_q(), ANYONE) is None

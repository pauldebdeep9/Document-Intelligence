"""EV-02's centrepiece: a checker that walks report.json's payload and fails
on any float in [0,1] that isn't accompanied by a sibling "n" in the same
JSON object -- enforced mechanically, not left as a convention someone has
to remember (answer_accuracy's {n, correct, accuracy} already got this
right; recall@5/recall@8/mrr/ndcg@8/abstention_precision/abstention_recall,
restricted.<subtype>.primary_recall@8, and extraction.auto_accept_error_rate
did not, until eval/report.py's EV-02 changes).

"Adjacent" is purely structural: the float's immediate parent dict must
contain a key literally named "n" whose value is a non-negative int (not a
bool -- bool is an int subclass in Python, excluded explicitly). The checker
cannot verify "n" is semantically the CORRECT denominator -- only that a
plausible one is sitting right next to the rate, not several keys away in a
namespace shared with other rates that have different denominators (that
ambiguity, not bare absence, was n_answerable's actual problem: it sat in
the same flat dict as abstention_precision/abstention_recall too, whose
denominators are different populations entirely).

Two fixtures, both committed, neither carrying any real per-principal data:

  fixtures/ev01_pre_ev02_report.json
    A real report.json from a live 74-outcome retrieval run (see
    runs/ev01_gate_source, produced under EV-01), captured before EV-02's
    report.py changes -- committed because it is pure aggregate numbers (no
    retrieved_ids/citations/answer_text; see EV-01's A5 on why
    outcomes.jsonl itself is never committed). --harness retrieval only, so
    it has no extraction subtree at all -- 8 bare rates, not 9.

  fixtures/ev02_synthetic_old_format_combined_report.json
    Genuinely EXECUTED, not hand-typed: the real pre-EV-02 report.py (git
    show fa5f2b0:src/isc/eval/report.py) run against a small synthetic
    ExtractionReport + RetrievalReport, so it has both subtrees and
    demonstrates all 9 values this item fixes in one file, including
    auto_accept_error_rate (retrieval-only ev01_gate_source never exercised
    the extraction path, so it alone cannot demonstrate that one).

  fixtures/ev02_synthetic_outcomes/{outcomes.jsonl,failed.jsonl}
    Hand-built QuestionOutcome objects -- a None-vs-False pair on
    answer_correct, a restricted pair sharing one question_id, a leaked
    chunk, and a subtype (restricted_unfiltered) with zero outcomes at all.
    Nothing here corresponds to a real principal's real view, so unlike
    outcomes.jsonl from a live run it carries no ACL semantics and is freely
    committable -- the permanent, clone-safe input for this item's tests and
    for EV-04's future diff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from isc.eval import outcomes as eval_outcomes
from isc.eval.extraction import ExtractionReport, FieldOutcome
from isc.eval.report import (
    REPORT_SCHEMA_VERSION,
    ReportSchemaMismatch,
    validate_schema_version,
    write,
)
from isc.eval.retrieval import QuestionOutcome, RetrievalReport

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Not a rate at all -- a float in [0,1] that isn't a ratio over a
# population, so "needs an n" never applied to it in the first place. Every
# entry needs a written reason; this is not a place to quietly park a real
# gap (see the TODO below the walker for that).
NOT_A_RATE = {
    "bin_low", "bin_high",  # calibration_bins()'s fixed bin boundaries, not computed values
    "gap",                  # accuracy - bin_midpoint: a signed difference, not a ratio;
                             # its sibling "accuracy" already carries its own adjacent "n"
    "confidence",            # one FieldOutcome's own score -- a single item, not an
                             # aggregate over a population; there is no "n" to have
}


def _walk(node: Any, path: str, violations: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key

            # TODO: EV-06 -- extraction.by_field's PRF shape (precision,
            # recall, f1) is a KNOWN, UNFIXED instance of exactly the defect
            # this checker exists to catch: recall's true denominator sits
            # under "support", not "n"; precision's and f1's true
            # denominators (tp+fp, and a composite of two different counts)
            # are not present under ANY name. NOT excluded because it isn't
            # a rate -- it plainly is one -- but because fixing it means
            # restructuring metrics.py's shared PRF dataclass (~18 fields x
            # up to 3 rates each in the real corpus, far more than this
            # item's 9), not a report.py presentation change. See EV-02
            # Stage 1's inventory. Remove this skip when EV-06 lands, not
            # before -- a checker that quietly tolerates the thing it was
            # built to catch is worse than no checker.
            if path == "extraction.by_field" or path.startswith("extraction.by_field."):
                if key in {"precision", "recall", "f1"}:
                    continue

            if isinstance(value, float) and 0.0 <= value <= 1.0:
                if key in NOT_A_RATE:
                    continue
                n = node.get("n")
                if not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
                    violations.append(child_path)
            else:
                _walk(value, child_path, violations)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", violations)


def find_bare_rates(payload: dict) -> list[str]:
    """Every JSON path to a float in [0,1] whose immediate parent object has
    no sibling "n", excluding NOT_A_RATE and the EV-06 TODO above. Empty
    means every in-scope rate in the payload can be sized by a reader
    without cross-referencing anything else in the file."""
    violations: list[str] = []
    _walk(payload, "", violations)
    return violations


# -- the checker against real (or genuinely executed) old-format files -----

def test_old_pre_ev02_retrieval_report_has_exactly_8_bare_rates():
    """Regression proof, not a synthetic example: this is an actual
    report.json from a real 74-outcome live run, captured before EV-02's
    report.py changes. --harness retrieval only -- no extraction subtree,
    so no auto_accept_error_rate here; the 9th offender is demonstrated
    separately below. If this ever stops failing without report.py's shape
    actually being fixed, the checker itself has gone blind."""
    payload = json.loads((FIXTURES / "ev01_pre_ev02_report.json").read_text())

    violations = find_bare_rates(payload)

    assert len(violations) == 8
    assert set(violations) == {
        "retrieval.recall@5",
        "retrieval.recall@8",
        "retrieval.mrr",
        "retrieval.ndcg@8",
        "retrieval.abstention_precision",
        "retrieval.abstention_recall",
        "retrieval.restricted.restricted_filtered.primary_recall@8",
        "retrieval.restricted.restricted_unfiltered.primary_recall@8",
    }


def test_old_format_combined_report_has_exactly_9_bare_rates():
    """fixtures/ev02_synthetic_old_format_combined_report.json is the real
    pre-EV-02 report.py, actually executed (not hand-typed JSON) against a
    small synthetic sample carrying both subtrees -- the one place all 9
    values this item fixes appear together, including
    extraction.auto_accept_error_rate."""
    payload = json.loads(
        (FIXTURES / "ev02_synthetic_old_format_combined_report.json").read_text()
    )

    violations = find_bare_rates(payload)

    assert len(violations) == 9
    assert set(violations) == {
        "retrieval.recall@5",
        "retrieval.recall@8",
        "retrieval.mrr",
        "retrieval.ndcg@8",
        "retrieval.abstention_precision",
        "retrieval.abstention_recall",
        "retrieval.restricted.restricted_filtered.primary_recall@8",
        "retrieval.restricted.restricted_unfiltered.primary_recall@8",
        "extraction.auto_accept_error_rate",
    }


# -- the checker against the current format --------------------------------

def _sample_retrieval_report() -> RetrievalReport:
    return RetrievalReport(outcomes=[
        QuestionOutcome(question_id="q_cd_02", question_class="answerable",
                         subtype="cross_document", retrieved_ids=["chk_a", "chk_b"],
                         gold_ids={"chk_a", "chk_b"}, answer_correct=False,
                         abstained=True, abstention_reason="attribution_mismatch"),
        QuestionOutcome(question_id="q_li_01", question_class="answerable",
                         subtype="line_item", retrieved_ids=["chk_x"],
                         gold_ids={"chk_x"}, answer_correct=True),
        QuestionOutcome(question_id="q_ua_01", question_class="unanswerable",
                         subtype="absent", abstained=True,
                         abstention_reason="insufficient_context", abstention_correct=True),
        QuestionOutcome(question_id="q_ua_07", question_class="unanswerable",
                         subtype="underspecified", abstained=False, abstention_correct=False),
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", principal_id="u_alice",
                         is_gold_principal=True, retrieved_ids=["chk_r"],
                         gold_ids={"chk_r"}, answer_correct=True),
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", principal_id="u_ben",
                         is_gold_principal=False, retrieved_ids=[], abstained=True),
        QuestionOutcome(question_id="q_re_09", question_class="restricted",
                         subtype="no_reader", principal_id="u_alice", retrieved_ids=[]),
    ])


def _sample_extraction_report() -> ExtractionReport:
    return ExtractionReport(outcomes=[
        FieldOutcome("doc_1", "po_number", "correct", "normalisation", "123", "123", 0.95),
        FieldOutcome("doc_1", "supplier_name", "wrong", "normalisation", "Acme", "Acmee", 0.95),
        FieldOutcome("doc_1", "lines[20]", "dropped_line", "normalisation",
                     None, {"line_number": 20}, 0.92),
    ])


def test_current_format_has_no_bare_rates_in_retrieval(tmp_path):
    write(tmp_path, None, _sample_retrieval_report())
    payload = json.loads((tmp_path / "report.json").read_text())

    assert find_bare_rates(payload) == []


def test_current_format_has_no_bare_rates_in_extraction(tmp_path):
    write(tmp_path, _sample_extraction_report(), None, threshold=0.90)
    payload = json.loads((tmp_path / "report.json").read_text())

    assert find_bare_rates(payload) == []


def test_current_format_has_no_bare_rates_with_both_harnesses(tmp_path):
    write(tmp_path, _sample_extraction_report(), _sample_retrieval_report(), threshold=0.90)
    payload = json.loads((tmp_path / "report.json").read_text())

    assert find_bare_rates(payload) == []


# -- the synthetic outcomes fixture (amendment B) ---------------------------

def test_synthetic_outcomes_fixture_round_trips_and_has_no_bare_rates(tmp_path):
    """Loads the committed, hand-built outcomes fixture (no live run, no
    real principal's data), rescores it through report.write(), and checks
    the result -- this is the "permanent, clone-safe input" amendment B
    asked for, actually exercised end to end."""
    result = eval_outcomes.load(FIXTURES / "ev02_synthetic_outcomes" / "outcomes.jsonl")
    assert len(result.report.outcomes) == 5

    write(tmp_path, None, result.report)
    payload = json.loads((tmp_path / "report.json").read_text())

    assert find_bare_rates(payload) == []


def test_synthetic_outcomes_fixture_exercises_a_zero_denominator_subtype(tmp_path):
    """The fixture deliberately has restricted_filtered pairs but NO
    restricted_unfiltered outcomes at all -- restricted_unfiltered's
    primary_recall@8 must come back as {"n": 0, ...: 0.0}, not a bare float
    a reader could mistake for a real measured score."""
    result = eval_outcomes.load(FIXTURES / "ev02_synthetic_outcomes" / "outcomes.jsonl")

    write(tmp_path, None, result.report)
    payload = json.loads((tmp_path / "report.json").read_text())
    restricted = payload["retrieval"]["restricted"]

    assert restricted["restricted_filtered"]["n_pairs"] == 1
    assert "restricted_unfiltered" not in restricted or restricted["restricted_unfiltered"] == {
        "n_pairs": 0,
        "primary_recall@8": {"n": 0, "primary_recall@8": 0.0},
        "primary_answer_correct": 0,
        "secondary_n": 0,
        "secondary_correctly_empty_and_abstained": 0,
        "leaks": 0,
    }


def test_synthetic_outcomes_fixture_carries_a_leak_and_a_none_vs_false_pair():
    """Confirms the fixture actually has the structural cases amendment B
    asked for -- not asserting report.json shape here, just that the
    committed file still contains what it's supposed to."""
    result = eval_outcomes.load(FIXTURES / "ev02_synthetic_outcomes" / "outcomes.jsonl")
    outcomes = [o for o in result.report.outcomes if o.question_id == "syn_re_01"]

    # (2) one question_id, two principals -- the restricted pair.
    assert {o.principal_id for o in outcomes} == {"p_alice", "p_ben"}
    # (3) the denied side leaked a chunk.
    assert result.report.leaks_by_subtype() == {"restricted_filtered": 1}
    # (1) None (not applicable) vs False (checked, wrong) on answer_correct.
    none_correct = [o.answer_correct for o in result.report.outcomes
                     if o.question_id == "syn_ans_01"][0]
    false_correct = [o.answer_correct for o in result.report.outcomes
                      if o.question_id == "syn_ans_02"][0]
    assert none_correct is None
    assert false_correct is False


# -- n=0: the guard value is still shown next to its own denominator -------

def test_n_zero_is_emitted_next_to_abstention_precisions_zero_guard(tmp_path):
    """abstention_precision() returns 0.0 when nothing abstained
    (retrieval.py's explicit `if not abstained: return 0.0`) -- not touching
    that, only making sure a reader sees n=0 right next to it instead of a
    bare 0.0 that reads identically to '0 of 5, all wrong'."""
    report = RetrievalReport(outcomes=[
        QuestionOutcome(question_id="q1", question_class="answerable", subtype="single_hop",
                         retrieved_ids=["c1"], gold_ids={"c1"}, answer_correct=True),
    ])
    write(tmp_path, None, report)
    payload = json.loads((tmp_path / "report.json").read_text())

    ap = payload["retrieval"]["abstention_precision"]
    assert ap == {"n": 0, "expected_to_abstain": 0, "abstention_precision": 0.0}


def test_n_zero_is_emitted_next_to_abstention_recalls_vacuous_one(tmp_path):
    """abstention_recall() returns 1.0 (not 0.0) when there are zero
    unanswerable questions -- a DIFFERENT guard convention, deliberately
    left as-is. The fix is that n=0 sits right next to it either way."""
    report = RetrievalReport(outcomes=[
        QuestionOutcome(question_id="q1", question_class="answerable", subtype="single_hop",
                         retrieved_ids=["c1"], gold_ids={"c1"}, answer_correct=True),
    ])
    write(tmp_path, None, report)
    payload = json.loads((tmp_path / "report.json").read_text())

    ar = payload["retrieval"]["abstention_recall"]
    assert ar == {"n": 0, "correct": 0, "abstention_recall": 1.0}


def test_n_zero_is_emitted_for_recall_mrr_ndcg_when_no_answerable_outcomes(tmp_path):
    report = RetrievalReport(outcomes=[
        QuestionOutcome(question_id="q1", question_class="unanswerable", subtype="absent",
                         abstained=True, abstention_reason="no_results", abstention_correct=True),
    ])
    write(tmp_path, None, report)
    payload = json.loads((tmp_path / "report.json").read_text())
    r = payload["retrieval"]

    assert r["recall@5"] == {"n": 0, "recall@5": 0.0}
    assert r["recall@8"] == {"n": 0, "recall@8": 0.0}
    assert r["mrr"] == {"n": 0, "mrr": 0.0}
    assert r["ndcg@8"] == {"n": 0, "ndcg@8": 0.0}


def test_n_zero_is_emitted_for_auto_accept_error_rate_when_nothing_clears_threshold(tmp_path):
    report = ExtractionReport(outcomes=[
        FieldOutcome("doc_1", "po_number", "correct", "normalisation", "123", "123", 0.10),
    ])
    write(tmp_path, report, None, threshold=0.90)
    payload = json.loads((tmp_path / "report.json").read_text())

    assert payload["extraction"]["auto_accept_error_rate"] == {
        "n": 0, "wrong": 0, "auto_accept_error_rate": 0.0,
    }


# -- auto_accept_band(): consistent with auto_accept_error_rate ------------

def test_auto_accept_band_is_consistent_with_auto_accept_error_rate():
    report = _sample_extraction_report()
    threshold = 0.90

    rate = report.auto_accept_error_rate(threshold)
    wrong, total = report.auto_accept_band(threshold)

    assert total > 0
    assert wrong / total == rate


def test_auto_accept_band_matches_the_rate_across_several_thresholds():
    """Same population, same check, for a range of thresholds -- not just
    the one value happened to be picked above."""
    report = ExtractionReport(outcomes=[
        FieldOutcome("doc_1", "a", "correct", "normalisation", "1", "1", 0.95),
        FieldOutcome("doc_1", "b", "wrong", "normalisation", "2", "3", 0.92),
        FieldOutcome("doc_1", "c", "missed", "normalisation", None, "4", 1.0),
        FieldOutcome("doc_1", "d", "correct", "normalisation", "5", "5", 0.40),
    ])
    for threshold in (0.0, 0.5, 0.9, 1.0, 1.01):
        rate = report.auto_accept_error_rate(threshold)
        wrong, total = report.auto_accept_band(threshold)
        if total == 0:
            assert rate == 0.0
        else:
            assert wrong / total == rate


# -- schema_version: present, and a wrong version is rejected loudly -------

def test_schema_version_is_present_and_correct(tmp_path):
    write(tmp_path, None, _sample_retrieval_report())
    payload = json.loads((tmp_path / "report.json").read_text())

    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    validate_schema_version(payload)  # must not raise


def test_wrong_schema_version_is_rejected_loudly(tmp_path):
    write(tmp_path, None, _sample_retrieval_report())
    payload = json.loads((tmp_path / "report.json").read_text())
    payload["schema_version"] = REPORT_SCHEMA_VERSION + 1

    with pytest.raises(ReportSchemaMismatch, match="schema_version"):
        validate_schema_version(payload)


def test_missing_schema_version_is_rejected_too():
    """The pre-EV-02 fixture has no schema_version key at all -- absence
    must be rejected exactly like a wrong version, not treated as some
    implicit version 0 that passes by default."""
    payload = json.loads((FIXTURES / "ev01_pre_ev02_report.json").read_text())
    assert "schema_version" not in payload

    with pytest.raises(ReportSchemaMismatch):
        validate_schema_version(payload)

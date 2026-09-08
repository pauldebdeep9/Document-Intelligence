"""EV-02's centrepiece: a checker that walks report.json's payload and fails
on any float in [0,1] that isn't accompanied by a sibling "n" in the same
JSON object -- enforced mechanically, not left as a convention someone has
to remember (answer_accuracy's {n, correct, accuracy} already got this
right; recall@5/recall@8/mrr/ndcg@8/abstention_precision/abstention_recall
and restricted.<subtype>.primary_recall@8 did not, until eval/report.py's
EV-02 changes).

"Adjacent" is purely structural: the float's immediate parent dict must
contain a key literally named "n" whose value is a non-negative int (not a
bool -- bool is an int subclass in Python, excluded explicitly). The checker
cannot verify "n" is semantically the CORRECT denominator -- only that a
plausible one is sitting right next to the rate, not several keys away in a
namespace shared with other rates that have different denominators (that
ambiguity, not bare absence, was n_answerable's actual problem: it sat in
the same flat dict as abstention_precision/abstention_recall too, whose
denominators are different populations entirely).

fixtures/ev01_pre_ev02_report.json is a real report.json produced by a live
74-outcome retrieval run (see runs/ev01_gate_source, produced under EV-01),
captured before EV-02's report.py changes -- committed because it is pure
aggregate numbers (no retrieved_ids/citations/answer_text; see EV-01's A5 on
why outcomes.jsonl itself is NOT committed). Kept as a permanent regression
fixture: proof the checker detects a real historical instance of the defect
it exists to catch, not just a synthetic example built to please it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from isc.eval.extraction import ExtractionReport, FieldOutcome
from isc.eval.report import (
    REPORT_SCHEMA_VERSION,
    ReportSchemaMismatch,
    validate_schema_version,
    write,
)
from isc.eval.retrieval import QuestionOutcome, RetrievalReport

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Every entry below needs a written reason. Two different KINDS of entry --
# conflating them would let a real gap hide behind a "this was reviewed and
# is fine" reading it doesn't deserve:
#
#   NOT_A_RATE   -- not a ratio over a population at all, so "needs an n"
#                   never applied to it in the first place.
#   DEFERRED_GAP -- IS a rate that needs this same fix. Deliberately out of
#                   scope for THIS item (see EV-02 Stage 1 inventory:
#                   extraction.by_field has ~18 fields x up to 3 rates each,
#                   far more than retrieval's 8, and fixing it means
#                   restructuring metrics.py's shared PRF dataclass, not a
#                   report.py presentation change). A known, tracked gap,
#                   not a false positive.
NOT_A_RATE = {
    "bin_low", "bin_high",  # calibration_bins()'s fixed bin boundaries, not computed values
    "gap",                  # accuracy - bin_midpoint: a signed difference, not a ratio;
                             # its sibling "accuracy" already carries its own adjacent "n"
    "confidence",            # one FieldOutcome's own score -- a single item, not an
                             # aggregate over a population; there is no "n" to have
}
DEFERRED_GAP = {"precision", "recall", "f1"}  # extraction.by_field's PRF shape -- see above


def _walk(node: Any, path: str, violations: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, float) and 0.0 <= value <= 1.0:
                if key in NOT_A_RATE or key in DEFERRED_GAP:
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
    no sibling "n", excluding the written allowlist above. Empty means every
    rate in the payload can be sized by a reader without cross-referencing
    anything else in the file."""
    violations: list[str] = []
    _walk(payload, "", violations)
    return violations


# -- the checker against a real, historical broken file -------------------

def test_old_pre_ev02_report_has_bare_rates_without_denominators():
    """Regression proof, not a synthetic example: this is an actual
    report.json from a real 74-outcome live run, captured before EV-02's
    report.py changes. If this ever stops failing without report.py's shape
    actually being fixed, the checker itself has gone blind."""
    payload = json.loads((FIXTURES / "ev01_pre_ev02_report.json").read_text())

    violations = find_bare_rates(payload)

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


def test_n_zero_is_emitted_for_restricted_summary_with_no_pairs(tmp_path):
    """No restricted_unfiltered outcomes at all -- n_pairs=0, and the
    checker must not mistake the resulting 0.0 for a real measured score."""
    report = RetrievalReport(outcomes=[
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", principal_id="u_alice",
                         is_gold_principal=True, retrieved_ids=["c1"],
                         gold_ids={"c1"}, answer_correct=True),
    ])
    write(tmp_path, None, report)
    payload = json.loads((tmp_path / "report.json").read_text())

    assert payload["retrieval"]["restricted"]["restricted_unfiltered"]["primary_recall@8"] == {
        "n": 0, "primary_recall@8": 0.0,
    }


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

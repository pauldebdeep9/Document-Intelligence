"""Unit test for eval/report.py's write().

Regression for a real bug: `vars(v)` on a PRF raises TypeError, because PRF
is a slots=True dataclass (no __dict__) -- never caught by any test because
`isc eval` was a stub (`raise typer.Exit`) until it was wired up for real
here. Nothing exercised write() end to end before that.
"""

from __future__ import annotations

import json

from isc.eval.extraction import ExtractionReport, FieldOutcome
from isc.eval.report import write
from isc.eval.retrieval import QuestionOutcome, RetrievalReport


def test_write_handles_prf_and_line_outcomes_without_raising(tmp_path):
    outcomes = [
        FieldOutcome("doc_1", "po_number", "correct", "normalisation", "123", "123", 0.95),
        FieldOutcome("doc_1", "supplier_name", "wrong", "normalisation", "Acme", "Acmee", 0.95),
        FieldOutcome("doc_1", "lines[20]", "dropped_line", "normalisation",
                     None, {"line_number": 20}, 0.92),
    ]
    report = ExtractionReport(outcomes)

    md_path = write(tmp_path, report, None, threshold=0.90)

    assert md_path.exists()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["extraction"]["by_field"]["po_number"]["support"] == 1
    assert payload["extraction"]["line_outcomes"][0]["outcome"] == "dropped_line"
    text = md_path.read_text()
    assert "po_number" in text
    assert "dropped_line" in text


def test_write_renders_retrieval_report_without_raising(tmp_path):
    """P1-09 smoke test, same reasoning as the extraction one above:
    nothing exercised the new retrieval section of write() end to end
    before this -- recall_by_subtype/answer_accuracy/answerable_failures/
    restricted_summary/no_reader_summary/leaks_by_subtype all have to
    render, including the empty-dict/empty-list cases, without raising."""
    outcomes = [
        QuestionOutcome(question_id="q_cd_02", question_class="answerable",
                         subtype="cross_document", retrieved_ids=["chk_a", "chk_b"],
                         gold_ids={"chk_a", "chk_b"}, answer_correct=False,
                         abstained=True, abstention_reason="attribution_mismatch"),
        QuestionOutcome(question_id="q_li_01", question_class="answerable",
                         subtype="line_item", retrieved_ids=["chk_x"],
                         gold_ids={"chk_x"}, answer_correct=True),
        QuestionOutcome(question_id="q_ua_01", question_class="unanswerable",
                         subtype="absent", abstained=True, abstention_reason="insufficient_context",
                         abstention_correct=True),
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
    ]
    report = RetrievalReport(outcomes=outcomes)

    md_path = write(tmp_path, None, report)

    assert md_path.exists()
    payload = json.loads((tmp_path / "report.json").read_text())
    r = payload["retrieval"]
    assert r["recall_by_subtype"]["cross_document"]["n"] == 1
    assert r["answer_accuracy"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert len(r["answerable_failures"]) == 1
    assert r["answerable_failures"][0]["question_id"] == "q_cd_02"
    assert r["restricted"]["restricted_filtered"]["n_pairs"] == 1
    assert r["no_reader"]["all_empty"] is True
    assert r["passed"] is True

    text = md_path.read_text()
    assert "cross_document" in text
    assert "line_item" in text
    assert "generation/grounding" in text

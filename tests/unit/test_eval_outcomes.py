"""Unit tests for eval/outcomes.py (EV-01): round-tripping
RetrievalReport.outcomes and RetrievalEvalResult.failed through
outcomes.jsonl/failed.jsonl with no live model, index, or docstore access.

Scoring logic itself (RetrievalReport's methods, eval/metrics.py) is out of
scope here and already covered by test_eval_retrieval.py -- this file is
purely about persistence fidelity: does what comes back out equal what went
in, byte for byte where that matters and field for field everywhere else.
"""

from __future__ import annotations

import json

import pytest

from isc.eval.outcomes import (
    FAILED_FILENAME,
    OUTCOMES_FILENAME,
    OUTCOMES_SCHEMA_VERSION,
    OutcomesSchemaMismatch,
    load,
    write,
)
from isc.eval.retrieval import QuestionOutcome, RetrievalEvalResult, RetrievalReport


def _outcome(**kw) -> QuestionOutcome:
    base = dict(question_id="q1", question_class="answerable", subtype="single_hop")
    base.update(kw)
    return QuestionOutcome(**base)


# -- round trip: every field, including the None-vs-False distinction -----

def test_round_trip_preserves_every_field_and_the_none_vs_false_distinction(tmp_path):
    outcomes = [
        _outcome(
            question_id="q_sh_01", question_class="answerable", subtype="single_hop",
            principal_id="u_alice", is_gold_principal=True,
            retrieved_ids=["c3", "c1", "c2"], gold_ids={"c9", "c1", "c2"},
            abstained=False, abstention_reason=None, answer_text="The total is 100.",
            citations=["c1", "c2"], answer_correct=True, abstention_correct=None,
            citations_valid=True, leaked_chunk_ids=[],
        ),
        # answer_correct=None ("not applicable") vs False ("checked, wrong")
        # are different outcomes for answer_accuracy()'s denominator -- must
        # not collapse to the same thing on reload.
        _outcome(
            question_id="q_sh_02", question_class="answerable", subtype="single_hop",
            principal_id="u_chen", is_gold_principal=True,
            retrieved_ids=[], gold_ids=set(), answer_correct=None,
        ),
        _outcome(
            question_id="q_sh_03", question_class="answerable", subtype="single_hop",
            principal_id="u_chen", is_gold_principal=True,
            retrieved_ids=["c5"], gold_ids={"c5"}, answer_correct=False,
        ),
        # abstention_correct=None vs False, same distinction, different field.
        _outcome(
            question_id="q_ua_01", question_class="unanswerable", subtype="absent",
            principal_id="u_ewan", abstained=True, abstention_reason="insufficient_context",
            abstention_correct=None,
        ),
        _outcome(
            question_id="q_ua_02", question_class="unanswerable", subtype="absent",
            principal_id="u_ewan", abstained=True, abstention_reason="attribution_mismatch",
            abstention_correct=False,
        ),
        # A leak, and a non-gold-principal denial.
        _outcome(
            question_id="q_re_01", question_class="restricted", subtype="restricted_filtered",
            principal_id="u_ben", is_gold_principal=False, retrieved_ids=["c_secret"],
            leaked_chunk_ids=["c_secret"],
        ),
    ]
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=outcomes), failed=[])

    outcomes_path, _ = write(tmp_path, result)
    loaded = load(outcomes_path)

    assert loaded.report.outcomes == outcomes
    # dataclass __eq__ already distinguishes None from False (None == False
    # is False in Python), but assert the load-bearing ones explicitly so a
    # future refactor that weakens the comparison still gets caught.
    assert loaded.report.outcomes[1].answer_correct is None
    assert loaded.report.outcomes[2].answer_correct is False
    assert loaded.report.outcomes[3].abstention_correct is None
    assert loaded.report.outcomes[4].abstention_correct is False
    # gold_ids round-trips as a set, not a list -- membership, not order.
    assert loaded.report.outcomes[0].gold_ids == {"c1", "c2", "c9"}
    assert isinstance(loaded.report.outcomes[0].gold_ids, set)
    # retrieved_ids order IS preserved -- mrr()/ndcg_at_k() score position.
    assert loaded.report.outcomes[0].retrieved_ids == ["c3", "c1", "c2"]


def test_gold_ids_round_trips_by_membership_regardless_of_insertion_order(tmp_path):
    a = _outcome(question_id="qa", gold_ids={"z", "a", "m"})
    b = _outcome(question_id="qb", gold_ids={"m", "z", "a"})  # same set, built differently
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[a, b]), failed=[])

    outcomes_path, _ = write(tmp_path, result)
    loaded = load(outcomes_path)

    assert loaded.report.outcomes[0].gold_ids == loaded.report.outcomes[1].gold_ids == {"a", "m", "z"}


# -- failed list survives reload -------------------------------------------

def test_failed_list_survives_reload(tmp_path):
    result = RetrievalEvalResult(
        report=RetrievalReport(outcomes=[_outcome()]),
        failed=[
            ("q_sh_05", "u_alice", "provider timeout"),
            ("q_sh_09", "u_chen", "rate limited"),
        ],
    )

    outcomes_path, _ = write(tmp_path, result)
    loaded = load(outcomes_path)

    assert loaded.failed == result.failed


def test_failed_list_survives_reload_when_empty(tmp_path):
    """The empty case is the one that risks silently vanishing -- see
    OutcomesSchemaMismatch's docstring on why failed.jsonl always carries a
    header line even with zero failures."""
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[_outcome()]), failed=[])

    outcomes_path, _ = write(tmp_path, result)
    loaded = load(outcomes_path)

    assert loaded.failed == []


# -- a file written twice from the same report is byte-identical ----------

def test_write_twice_from_the_same_report_is_byte_identical(tmp_path):
    outcomes = [
        _outcome(question_id="q1", retrieved_ids=["c2", "c1"], gold_ids={"c1", "c9"},
                 answer_correct=False, abstention_correct=None),
        _outcome(question_id="q2", question_class="restricted", subtype="restricted_filtered",
                 principal_id="u_ben", is_gold_principal=False, leaked_chunk_ids=["c3"]),
    ]
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=outcomes),
                                  failed=[("q3", "u_x", "timeout")])

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    outcomes_a, failed_a = write(dir_a, result)
    outcomes_b, failed_b = write(dir_b, result)

    assert outcomes_a.read_bytes() == outcomes_b.read_bytes()
    assert failed_a.read_bytes() == failed_b.read_bytes()


# -- schema_version: present, and a wrong version is rejected loudly ------

def test_schema_version_is_present_on_every_outcome_line(tmp_path):
    result = RetrievalEvalResult(
        report=RetrievalReport(outcomes=[_outcome(question_id="q1"), _outcome(question_id="q2")]),
        failed=[],
    )
    outcomes_path, failed_path = write(tmp_path, result)

    for line in outcomes_path.read_text().splitlines():
        assert json.loads(line)["schema_version"] == OUTCOMES_SCHEMA_VERSION
    header = json.loads(failed_path.read_text().splitlines()[0])
    assert header["schema_version"] == OUTCOMES_SCHEMA_VERSION


def test_wrong_schema_version_on_an_outcome_line_is_rejected_loudly(tmp_path):
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[_outcome()]), failed=[])
    outcomes_path, _ = write(tmp_path, result)

    record = json.loads(outcomes_path.read_text())
    record["schema_version"] = OUTCOMES_SCHEMA_VERSION + 1
    outcomes_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(OutcomesSchemaMismatch, match="schema_version"):
        load(outcomes_path)


def test_wrong_schema_version_on_the_failed_header_is_rejected_loudly(tmp_path):
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[_outcome()]), failed=[])
    outcomes_path, failed_path = write(tmp_path, result)

    header = json.loads(failed_path.read_text().splitlines()[0])
    header["schema_version"] = OUTCOMES_SCHEMA_VERSION + 1
    failed_path.write_text(json.dumps(header) + "\n")

    with pytest.raises(OutcomesSchemaMismatch, match="schema_version"):
        load(outcomes_path)


def test_a_truly_empty_failed_file_is_rejected_not_read_as_zero_failures(tmp_path):
    """write() always emits a header line even with zero failures -- a
    zero-byte failed.jsonl means truncation or hand-editing, not a clean
    empty run, and must not be silently treated as 'nothing failed'."""
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[_outcome()]), failed=[])
    outcomes_path, failed_path = write(tmp_path, result)

    failed_path.write_text("")

    with pytest.raises(OutcomesSchemaMismatch):
        load(outcomes_path)


def test_failed_header_count_mismatch_is_rejected(tmp_path):
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[_outcome()]),
                                  failed=[("q1", "u_x", "timeout")])
    outcomes_path, failed_path = write(tmp_path, result)

    lines = failed_path.read_text().splitlines()
    header = json.loads(lines[0])
    header["count"] = 5
    failed_path.write_text(json.dumps(header) + "\n" + lines[1] + "\n")

    with pytest.raises(ValueError, match="count"):
        load(outcomes_path)


def test_missing_failed_file_is_rejected_not_read_as_zero_failures(tmp_path):
    result = RetrievalEvalResult(report=RetrievalReport(outcomes=[_outcome()]), failed=[])
    outcomes_path, failed_path = write(tmp_path, result)

    failed_path.unlink()

    with pytest.raises(FileNotFoundError, match=FAILED_FILENAME):
        load(outcomes_path)


def test_missing_outcomes_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / OUTCOMES_FILENAME)


# -- re-score from file reproduces the same metrics as scoring in memory --

def test_rescore_from_file_reproduces_the_same_metrics_as_in_memory(tmp_path):
    """The actual point of this module: a corrected metric definition can be
    re-run against a file instead of a live re-run. Exercises every
    RetrievalReport method across every question class so a metric that
    depends on a field this module forgot to persist would show up here as
    a mismatch, not just as an equal outcomes list."""
    outcomes = [
        _outcome(question_id="q_sh_01", question_class="answerable", subtype="single_hop",
                 principal_id="u_alice", is_gold_principal=True,
                 retrieved_ids=["c1", "c2"], gold_ids={"c1"}, answer_correct=True),
        _outcome(question_id="q_sh_02", question_class="answerable", subtype="single_hop",
                 principal_id="u_chen", is_gold_principal=True,
                 retrieved_ids=["c5"], gold_ids={"c9"}, answer_correct=False),
        _outcome(question_id="q_cd_01", question_class="answerable", subtype="cross_document",
                 principal_id="u_alice", is_gold_principal=True,
                 retrieved_ids=["c7", "c8"], gold_ids={"c7", "c8"}, answer_correct=True),
        _outcome(question_id="q_ua_01", question_class="unanswerable", subtype="absent",
                 principal_id="u_ewan", abstained=True, abstention_reason="no_results",
                 abstention_correct=True),
        _outcome(question_id="q_ua_02", question_class="unanswerable", subtype="underspecified",
                 principal_id="u_gita", abstained=False, abstention_correct=False),
        _outcome(question_id="q_re_01", question_class="restricted", subtype="restricted_filtered",
                 principal_id="u_alice", is_gold_principal=True,
                 retrieved_ids=["c10"], gold_ids={"c10"}, answer_correct=True),
        _outcome(question_id="q_re_01", question_class="restricted", subtype="restricted_filtered",
                 principal_id="u_ben", is_gold_principal=False,
                 retrieved_ids=[], abstained=True),
        _outcome(question_id="q_re_10", question_class="restricted", subtype="restricted_unfiltered",
                 principal_id="u_ben", is_gold_principal=False,
                 retrieved_ids=["c_leak"], leaked_chunk_ids=["c_leak"]),
        _outcome(question_id="q_re_09", question_class="restricted", subtype="no_reader",
                 principal_id="u_dara", is_gold_principal=False, retrieved_ids=[]),
    ]
    original = RetrievalReport(outcomes=outcomes)
    result = RetrievalEvalResult(report=original,
                                  failed=[("q_sh_99", "u_frank", "provider timeout")])

    outcomes_path, _ = write(tmp_path, result)
    rescored = load(outcomes_path).report

    assert rescored.recall_at(5) == original.recall_at(5)
    assert rescored.recall_at(8) == original.recall_at(8)
    assert rescored.mean_mrr() == original.mean_mrr()
    assert rescored.ndcg(8) == original.ndcg(8)
    assert rescored.abstention_precision() == original.abstention_precision()
    assert rescored.abstention_recall() == original.abstention_recall()
    assert rescored.abstention_by_subtype() == original.abstention_by_subtype()
    assert rescored.recall_by_subtype() == original.recall_by_subtype()
    assert rescored.answer_accuracy() == original.answer_accuracy()
    assert rescored.answerable_failures() == original.answerable_failures()
    assert rescored.restricted_summary() == original.restricted_summary()
    assert rescored.no_reader_summary() == original.no_reader_summary()
    assert [o.question_id for o in rescored.leaks()] == [o.question_id for o in original.leaks()]
    assert rescored.leaks_by_subtype() == original.leaks_by_subtype()
    assert rescored.passed() == original.passed()

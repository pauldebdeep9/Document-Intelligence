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
from pathlib import Path

import pytest

from isc.common.errors import IscError
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
    # route defaults to "chunks": an outcome built in memory came from run(),
    # which always knows which path answered. route=None ("not recorded")
    # only ever comes from reading a schema-v1 file.
    base = dict(question_id="q1", question_class="answerable", subtype="single_hop",
                route="chunks")
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


# -- schema v2: route is recorded; v1 reads as "not recorded" --------------

V1_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ev02_synthetic_outcomes"


def _v1_expected(line: str) -> QuestionOutcome:
    """A v1 fixture line as QuestionOutcome, built by hand from the JSON --
    not via load() -- so this is an independent snapshot of what v1 means."""
    r = json.loads(line)
    assert r["schema_version"] == 1
    return QuestionOutcome(
        question_id=r["question_id"], question_class=r["question_class"],
        subtype=r["subtype"], principal_id=r["principal_id"],
        is_gold_principal=r["is_gold_principal"], retrieved_ids=list(r["retrieved_ids"]),
        gold_ids=set(r["gold_ids"]), abstained=r["abstained"],
        abstention_reason=r["abstention_reason"], answer_text=r["answer_text"],
        citations=list(r["citations"]), answer_correct=r["answer_correct"],
        abstention_correct=r["abstention_correct"], citations_valid=r["citations_valid"],
        leaked_chunk_ids=list(r["leaked_chunk_ids"]), route=None,
    )


def test_v2_round_trip_preserves_route(tmp_path):
    outcomes = [_outcome(question_id="q_a", route="chunks"),
                _outcome(question_id="q_b", route="records")]
    outcomes_path, _ = write(tmp_path, RetrievalEvalResult(
        report=RetrievalReport(outcomes=outcomes), failed=[]))
    records = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
    assert [r["route"] for r in records] == ["chunks", "records"]
    assert {r["schema_version"] for r in records} == {2}
    assert [o.route for o in load(outcomes_path).report.outcomes] == ["chunks", "records"]


def test_committed_v1_fixture_reads_route_as_not_recorded():
    lines = (V1_FIXTURE / "outcomes.jsonl").read_text().splitlines()
    expected = [_v1_expected(line) for line in lines]
    loaded = load(V1_FIXTURE / "outcomes.jsonl").report.outcomes
    assert loaded == expected
    assert [o.route for o in loaded] == [None] * len(lines)


def _written_record(tmp_path) -> tuple[Path, dict]:
    outcomes_path, _ = write(tmp_path, RetrievalEvalResult(
        report=RetrievalReport(outcomes=[_outcome()]), failed=[]))
    return outcomes_path, json.loads(outcomes_path.read_text())


@pytest.mark.parametrize("route", ["missing", "other", None])
def test_v2_record_without_a_valid_route_is_malformed(tmp_path, route):
    """A v2 record with no route is malformed, not "not recorded" -- only a
    v1 file means that."""
    outcomes_path, record = _written_record(tmp_path)
    record["schema_version"] = 2
    if route == "missing":
        del record["route"]
    else:
        record["route"] = route
    outcomes_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(IscError, match="route"):
        load(outcomes_path)


def test_schema_version_3_is_rejected(tmp_path):
    outcomes_path, record = _written_record(tmp_path)
    record["schema_version"] = 3
    outcomes_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(OutcomesSchemaMismatch, match="schema_version"):
        load(outcomes_path)


def test_a_file_mixing_v1_and_v2_records_is_rejected_naming_the_line(tmp_path):
    outcomes_path, v2 = _written_record(tmp_path)
    v1 = json.loads((V1_FIXTURE / "outcomes.jsonl").read_text().splitlines()[0])
    v2["schema_version"] = 2
    v2["route"] = "records"
    outcomes_path.write_text(json.dumps(v1) + "\n" + json.dumps(v2) + "\n")
    with pytest.raises(IscError, match=r":2:.*mix"):
        load(outcomes_path)


def test_writer_refuses_an_outcome_whose_route_was_not_recorded(tmp_path):
    """A v2 file never claims "not recorded": re-writing outcomes loaded from
    a v1 file would otherwise produce records the v2 reader rejects."""
    result = RetrievalEvalResult(
        report=RetrievalReport(outcomes=[_outcome(route=None)]), failed=[])
    with pytest.raises(ValueError, match="route"):
        write(tmp_path, result)


def test_retrieval_run_copies_the_answer_route():
    from isc.eval.retrieval import run
    from isc.models.acl import Principal
    from isc.models.answer import Answer

    class RecordsOrchestrator:
        def ask(self, question, principal):
            return Answer(question=question, text="Total spend: 1.00 SGD.", route="records")

    q = {"id": "q_cd_01", "text": "What did we spend?", "question_class": "answerable",
         "subtype": "cross_document", "gold_chunk_ids": [], "gold_answer": None,
         "principal": "u_alice"}
    result = run([q], {"u_alice": Principal(id="u_alice")}, RecordsOrchestrator())
    assert [o.route for o in result.report.outcomes] == ["records"]

"""EV-04: mutation tests for eval/diff.py's per-outcome differ.

Every case is built from the committed 5-outcome fixture
(tests/fixtures/ev02_synthetic_outcomes/outcomes.jsonl) plus programmatic
variants -- constructed with dataclasses.replace()/direct QuestionOutcome
and RetrievalEvalResult construction, never a second committed fixture.

Cases follow EV-04's own class list:
  identity, one test per behavioural class (leak_change, verdict_flip,
  abstention_change, retrieval_set, retrieval_order, text_change),
  None-vs-False verdict_flip, leak-above-text-change ranking, added/removed,
  status_change, both-failed, key collision, citations_valid mismatch,
  gold_change (data-complete + renderer-suppressed), and the composite
  priority-contract case.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from isc.eval import outcomes as eval_outcomes
from isc.eval.diff import (
    CitationsValidMismatch,
    DiffClass,
    FieldDiff,
    KeyCollision,
    diff,
    render,
)
from isc.eval.retrieval import QuestionOutcome, RetrievalEvalResult, RetrievalReport

FIXTURES = Path(__file__).parent.parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE_PATH = FIXTURES / "ev02_synthetic_outcomes" / "outcomes.jsonl"


def _load_fixture() -> RetrievalEvalResult:
    """Fresh load every call -- QuestionOutcome instances must never be
    shared between the two sides of a diff in these tests, or a mutation
    meant for one side would silently apply to both."""
    return eval_outcomes.load(FIXTURE_PATH)


def _with_replacement(result: RetrievalEvalResult, key: tuple[str, str], **changes) -> RetrievalEvalResult:
    outcomes = [
        replace(o, **changes) if (o.question_id, o.principal_id) == key else o
        for o in result.report.outcomes
    ]
    assert any((o.question_id, o.principal_id) == key for o in outcomes), f"no such key: {key}"
    return RetrievalEvalResult(report=RetrievalReport(outcomes=outcomes), failed=list(result.failed))


def _result(outcomes: list[QuestionOutcome], failed: list[tuple[str, str, str]] | None = None) -> RetrievalEvalResult:
    return RetrievalEvalResult(report=RetrievalReport(outcomes=outcomes), failed=failed or [])


# -- case 1: identity ---------------------------------------------------------

def test_identity_fixture_diffed_against_itself_is_empty():
    a = _load_fixture()
    b = _load_fixture()

    result = diff(a, b)

    assert result.is_identical is True
    assert result.changed == ()
    assert result.status_changes == ()
    assert result.added == ()
    assert result.removed == ()
    assert render(result) == "No differences."


def test_identity_real_74_outcome_run_diffed_against_itself_is_empty():
    path = REPO_ROOT / "runs" / "ev01_gate_source" / "eval" / "outcomes.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not present locally (evaluator-only, gitignored)")
    a = eval_outcomes.load(path)
    b = eval_outcomes.load(path)

    result = diff(a, b)

    assert result.is_identical is True


# -- case 2: one test per behavioural class -----------------------------------

def test_leak_change_single_record():
    a = _load_fixture()
    b = _with_replacement(_load_fixture(), ("syn_ans_01", "p_alice"), leaked_chunk_ids=["c_new_leak"])

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.key == ("syn_ans_01", "p_alice")
    assert rd.primary_class == DiffClass.LEAK_CHANGE
    assert len(rd.field_diffs) == 1
    assert rd.field_diffs[0].field == "leaked_chunk_ids"


def test_verdict_flip_single_record():
    a = _load_fixture()
    b = _with_replacement(_load_fixture(), ("syn_re_01", "p_alice"), answer_correct=False)

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.key == ("syn_re_01", "p_alice")
    assert rd.primary_class == DiffClass.VERDICT_FLIP
    assert rd.field_diffs == (FieldDiff("answer_correct", True, False, DiffClass.VERDICT_FLIP),)


def test_abstention_change_single_record():
    a = _load_fixture()
    b = _with_replacement(_load_fixture(), ("syn_ua_01", "p_chen"), abstention_reason="low_support")

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.key == ("syn_ua_01", "p_chen")
    assert rd.primary_class == DiffClass.ABSTENTION_CHANGE
    assert rd.field_diffs[0].field == "abstention_reason"


def test_retrieval_set_single_record():
    a = _load_fixture()
    b = _with_replacement(_load_fixture(), ("syn_ans_01", "p_alice"), retrieved_ids=["c99"])

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.primary_class == DiffClass.RETRIEVAL_SET
    assert rd.field_diffs[0].field == "retrieved_ids"


def test_text_change_single_record():
    a = _load_fixture()
    b = _with_replacement(_load_fixture(), ("syn_ans_02", "p_alice"), answer_text="a different draft")

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.primary_class == DiffClass.TEXT_CHANGE
    assert rd.field_diffs[0].field == "answer_text"


def test_citations_order_only_is_not_a_difference():
    """R2: citations are compared as a set, not positionally -- nothing in
    retrieval.py's scoring reads citation order today. Permuting citations
    with the same membership must report zero differences for this record."""
    a = _with_replacement(_load_fixture(), ("syn_ans_02", "p_alice"), citations=["c2", "c9"])
    b = _with_replacement(_load_fixture(), ("syn_ans_02", "p_alice"), citations=["c9", "c2"])

    result = diff(a, b)

    assert result.is_identical is True


# -- case 3: order-only ---------------------------------------------------

def test_retrieval_order_only_is_retrieval_order_not_retrieval_set():
    a = _with_replacement(_load_fixture(), ("syn_ans_01", "p_alice"), retrieved_ids=["c1", "c5"])
    b = _with_replacement(_load_fixture(), ("syn_ans_01", "p_alice"), retrieved_ids=["c5", "c1"])

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.primary_class == DiffClass.RETRIEVAL_ORDER
    assert rd.field_diffs[0].before == ["c1", "c5"]
    assert rd.field_diffs[0].after == ["c5", "c1"]


# -- case 4: None-vs-False verdict_flip ---------------------------------------

def test_none_to_false_answer_correct_is_a_verdict_flip():
    """The distinction that silently changes answer_accuracy()'s denominator
    (retrieval.py:231-238 filters on `answer_correct is not None`) -- must
    not classify as no-change or as presentational."""
    a = _load_fixture()  # syn_ans_01's answer_correct is None in the fixture
    b = _with_replacement(_load_fixture(), ("syn_ans_01", "p_alice"), answer_correct=False)

    result = diff(a, b)

    assert result.is_identical is False
    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.key == ("syn_ans_01", "p_alice")
    assert rd.primary_class == DiffClass.VERDICT_FLIP
    assert len(rd.field_diffs) == 1
    fd = rd.field_diffs[0]
    assert fd.field == "answer_correct"
    assert fd.before is None
    assert fd.after is False


# -- case 5: leak_change ranks above text_change ------------------------------

def test_leak_appears_ranks_above_text_change_on_the_same_record():
    a = _load_fixture()
    b = _with_replacement(
        _load_fixture(), ("syn_ans_01", "p_alice"),
        leaked_chunk_ids=["c_new_leak"], answer_text="different text too",
    )

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.primary_class == DiffClass.LEAK_CHANGE
    fields = {fd.field for fd in rd.field_diffs}
    assert fields == {"leaked_chunk_ids", "answer_text"}


# -- case 6: added / removed ---------------------------------------------------

def _extra_outcome(question_id: str, principal_id: str) -> QuestionOutcome:
    return QuestionOutcome(
        question_id=question_id, question_class="answerable", subtype="single_hop",
        principal_id=principal_id, is_gold_principal=True,
        retrieved_ids=["cX"], gold_ids={"cX"}, answer_correct=True,
    )


def test_added_record_present_in_b_only():
    a = _load_fixture()
    b_outcomes = list(_load_fixture().report.outcomes) + [_extra_outcome("q_new", "p_new")]
    b = _result(b_outcomes)

    result = diff(a, b)

    assert result.added == (result.added[0],)
    assert result.added[0].key == ("q_new", "p_new")
    assert result.removed == ()
    assert result.status_changes == ()


def test_removed_record_present_in_a_only():
    a_outcomes = list(_load_fixture().report.outcomes) + [_extra_outcome("q_old", "p_old")]
    a = _result(a_outcomes)
    b = _load_fixture()

    result = diff(a, b)

    assert result.removed == (result.removed[0],)
    assert result.removed[0].key == ("q_old", "p_old")
    assert result.added == ()
    assert result.status_changes == ()


# -- case 7: status_change -----------------------------------------------------

def test_status_change_failed_to_succeeded_is_not_added():
    key = ("q_flaky", "p_x")
    a = _result(list(_load_fixture().report.outcomes),
                failed=[(key[0], key[1], "provider timeout")])
    b_outcomes = list(_load_fixture().report.outcomes) + [_extra_outcome(*key)]
    b = _result(b_outcomes)

    result = diff(a, b)

    assert result.added == ()
    assert result.removed == ()
    assert len(result.status_changes) == 1
    sc = result.status_changes[0]
    assert sc.key == key
    assert sc.direction == "failed_to_succeeded"
    assert sc.failure_reason == "provider timeout"


def test_status_change_succeeded_to_failed_is_not_removed():
    key = ("q_flaky", "p_x")
    a_outcomes = list(_load_fixture().report.outcomes) + [_extra_outcome(*key)]
    a = _result(a_outcomes)
    b = _result(list(_load_fixture().report.outcomes),
                failed=[(key[0], key[1], "provider timeout")])

    result = diff(a, b)

    assert result.added == ()
    assert result.removed == ()
    assert len(result.status_changes) == 1
    sc = result.status_changes[0]
    assert sc.key == key
    assert sc.direction == "succeeded_to_failed"


# -- case 8: both-failed --------------------------------------------------

def test_key_failed_on_both_sides_produces_no_output():
    key = ("q_dead", "p_y")
    a = _result([], failed=[(key[0], key[1], "timeout")])
    b = _result([], failed=[(key[0], key[1], "different timeout message")])

    result = diff(a, b)

    assert result.is_identical is True


# -- case 9: key collision -----------------------------------------------------

def test_key_collision_raises_and_does_not_return_partial_results():
    dup_key = ("syn_ans_01", "p_alice")
    base_outcomes = list(_load_fixture().report.outcomes)
    dup_source = next(o for o in base_outcomes if (o.question_id, o.principal_id) == dup_key)
    a = _result(base_outcomes + [replace(dup_source)])
    # b differs from a on OTHER, non-colliding keys too -- if diff() ever
    # started catching the collision internally and returning a partial
    # result, those other differences would show up here. A raised
    # exception makes that structurally impossible: there is no return
    # path once diff() raises, so this assertion is the proof, not just
    # an observation.
    b = _with_replacement(_load_fixture(), ("syn_ans_02", "p_alice"), answer_text="unrelated change")

    with pytest.raises(KeyCollision) as exc_info:
        diff(a, b)

    assert exc_info.value.key == dup_key


# -- case 10: citations_valid mismatch -----------------------------------------

def test_citations_valid_mismatch_raises_naming_the_record():
    key = ("syn_ans_01", "p_alice")
    a = _load_fixture()
    b = _with_replacement(_load_fixture(), key, citations_valid=False)

    with pytest.raises(CitationsValidMismatch) as exc_info:
        diff(a, b)

    assert exc_info.value.key == key
    assert exc_info.value.before is True
    assert exc_info.value.after is False


# -- case 11: gold_change -------------------------------------------------

def test_gold_change_is_complete_at_the_data_layer_and_alone_in_the_render():
    gold_key = ("syn_re_01", "p_alice")
    text_key = ("syn_ans_02", "p_alice")

    a = _load_fixture()
    b = _load_fixture()
    # gold_key: gold_ids moves (gold_change) AND leaked_chunk_ids changes
    # too (would be leak_change on its own) -- proving the data layer keeps
    # BOTH field diffs even though primary_class is forced to gold_change.
    b = _with_replacement(b, gold_key, gold_ids={"c3", "c4"}, leaked_chunk_ids=["c_extra"])
    # text_key: an unrelated record with an ordinary text_change -- proving
    # other records are still fully classified even when gold_change fires
    # elsewhere in the same pair.
    b = _with_replacement(b, text_key, answer_text="unrelated text change")

    result = diff(a, b)

    assert result.has_gold_change is True
    by_key = {rd.key: rd for rd in result.changed}

    gold_rd = by_key[gold_key]
    assert gold_rd.primary_class == DiffClass.GOLD_CHANGE
    fields = {fd.field for fd in gold_rd.field_diffs}
    assert fields == {"gold_ids", "leaked_chunk_ids"}  # complete, not truncated

    text_rd = by_key[text_key]
    assert text_rd.primary_class == DiffClass.TEXT_CHANGE  # still computed, unaffected

    rendered = render(result)
    assert "GOLD SET DIFFERS" in rendered
    assert "gold_ids" in rendered
    assert "syn_re_01" in rendered
    # the renderer shows ONLY the gold_change section -- the unrelated
    # text_change record must not leak into this output.
    assert "syn_ans_02" not in rendered
    assert "text_change" not in rendered


# -- case 12: composite -- the priority scheme's actual contract --------------

def test_composite_leak_and_verdict_and_retrieval_set_keeps_all_three_diffs():
    """One record, three simultaneous field changes. Every case above tests
    classification in isolation; this is the one that proves grouping does
    not DROP the lower-priority detail -- primary_class picks leak_change,
    but answer_correct and retrieved_ids must both still be present in
    field_diffs, not silently subsumed."""
    key = ("syn_re_01", "p_alice")
    a = _load_fixture()
    b = _with_replacement(
        _load_fixture(), key,
        leaked_chunk_ids=["c_new_leak"],
        answer_correct=False,
        retrieved_ids=["c_other"],
    )

    result = diff(a, b)

    assert len(result.changed) == 1
    rd = result.changed[0]
    assert rd.key == key
    assert rd.primary_class == DiffClass.LEAK_CHANGE

    by_field = {fd.field: fd for fd in rd.field_diffs}
    assert set(by_field) == {"leaked_chunk_ids", "answer_correct", "retrieved_ids"}
    assert by_field["leaked_chunk_ids"].diff_class == DiffClass.LEAK_CHANGE
    assert by_field["answer_correct"].diff_class == DiffClass.VERDICT_FLIP
    assert by_field["retrieved_ids"].diff_class == DiffClass.RETRIEVAL_SET

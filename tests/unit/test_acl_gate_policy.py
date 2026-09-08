"""EV-03: mutation tests for the declared ACL gate policy (evaluate_acl_gate()).

Four required cases, asserted directly against evaluate_acl_gate() -- not
through report.write() -- so a failure here points at the policy itself,
not at JSON plumbing:

  1. clean               -- zero leaks                        -> PASS
  2. leak injected        -- one leak, otherwise the clean set  -> FAIL,
                             reason names the condition, not the chunk
  3. floor metrics, zero leaks -- every retrieval metric at its
     worst possible value, no leaks                            -> PASS
  4. perfect metrics, one leak -- every metric at its best,
     exactly one leak                                          -> FAIL

Cases 3 and 4 are the actual content of "a single leak fails the run
regardless of every other metric" (see retrieval.py's evaluate_acl_gate()
docstring) -- nothing before this file asserted either one. Every case is
also checked against RetrievalReport.passed(), the pre-existing gate, to
confirm the policy only added a name and a reason -- it did not change the
decision.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from isc.eval import outcomes as eval_outcomes
from isc.eval.retrieval import QuestionOutcome, RetrievalReport, evaluate_acl_gate

FIXTURES = Path(__file__).parent.parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent.parent


def _clean(report: RetrievalReport) -> RetrievalReport:
    """Same outcomes, every leaked_chunk_ids cleared -- case 1 derived from
    the same committed fixture case 2 uses, rather than hand-built twice."""
    return RetrievalReport(outcomes=[replace(o, leaked_chunk_ids=[]) for o in report.outcomes])


# -- case 1: clean, zero leaks -----------------------------------------------

def test_clean_report_zero_leaks_gate_passes():
    fixture = eval_outcomes.load(FIXTURES / "ev02_synthetic_outcomes" / "outcomes.jsonl").report
    report = _clean(fixture)
    assert report.leaks() == []

    gate = evaluate_acl_gate(report)

    assert gate.name == "acl_leak_gate"
    assert gate.passed is True
    assert gate.passed == report.passed()


# -- case 2: leak injected ----------------------------------------------------

def test_leak_injected_gate_fails_and_reason_names_condition_not_chunk():
    """The fixture as committed already carries the leak: syn_re_01/p_ben
    retrieves c_secret (tests/fixtures/ev02_synthetic_outcomes/outcomes.jsonl:4)."""
    report = eval_outcomes.load(FIXTURES / "ev02_synthetic_outcomes" / "outcomes.jsonl").report
    assert len(report.leaks()) == 1

    gate = evaluate_acl_gate(report)

    assert gate.passed is False
    assert gate.passed == report.passed()
    assert "1 chunk" in gate.reason
    assert "restricted_filtered" in gate.reason
    # The condition, not the evidence -- report.json is a committed,
    # clone-safe fixture with no per-principal data (EV-02's A5/B); a
    # reason string is held to the same line. The chunk/principal ids live
    # in outcomes.jsonl, not here.
    assert "c_secret" not in gate.reason
    assert "p_ben" not in gate.reason
    assert "p_alice" not in gate.reason


# -- case 3: floor metrics, zero leaks ---------------------------------------

def _floor_metrics_zero_leaks_report() -> RetrievalReport:
    """Every headline retrieval metric report.md prints -- recall@5/8, mrr,
    ndcg@8, answer_accuracy, abstention_precision, abstention_recall,
    restricted primary_recall@8/primary_answer_correct -- driven to its
    worst possible value (0.0/0), with acl_leaks == 0 throughout. Half of
    "regardless of every other metric": metrics as bad as they can be, gate
    must still say PASS."""
    return RetrievalReport(outcomes=[
        # answerable: wrong chunk, wrong answer, AND an abstain that should
        # never have happened -- an answerable gold-principal side
        # abstaining is a real failure (_expected_to_abstain() returns
        # False for it), which is what drags abstention_precision to 0 too.
        QuestionOutcome(
            question_id="q_floor_ans", question_class="answerable", subtype="single_hop",
            principal_id="p1", is_gold_principal=True,
            retrieved_ids=["wrong_chunk"], gold_ids={"c1"},
            answer_correct=False, abstained=True, abstention_reason="ungrounded_draft",
            leaked_chunk_ids=[],
        ),
        # unanswerable: did NOT abstain at all -- abstention_correct False,
        # driving abstention_recall to 0. abstained=False keeps it OUT of
        # abstention_precision's population, so that floor stays isolated
        # to the one outcome above.
        QuestionOutcome(
            question_id="q_floor_ua", question_class="unanswerable", subtype="absent",
            principal_id="p2", is_gold_principal=True,
            abstained=False, abstention_correct=False,
            leaked_chunk_ids=[],
        ),
        # restricted gold-principal side: wrong chunk, wrong answer.
        QuestionOutcome(
            question_id="q_floor_re", question_class="restricted", subtype="restricted_filtered",
            principal_id="p_alice", is_gold_principal=True,
            retrieved_ids=["wrong_chunk_2"], gold_ids={"c2"},
            answer_correct=False, abstained=False,
            leaked_chunk_ids=[],
        ),
        # restricted denied side: retrieves nothing, does NOT leak -- zero
        # leaks is case 3's other half of the claim. abstained=False keeps
        # it out of abstention_precision's population too.
        QuestionOutcome(
            question_id="q_floor_re", question_class="restricted", subtype="restricted_filtered",
            principal_id="p_ben", is_gold_principal=False,
            retrieved_ids=[], abstained=False,
            leaked_chunk_ids=[],
        ),
    ])


def test_floor_metrics_zero_leaks_gate_passes():
    report = _floor_metrics_zero_leaks_report()

    # The floor, metric by metric -- if this ever drifts because a helper
    # changed, this says WHICH metric stopped being 0, not just that the
    # gate's verdict changed.
    assert report.recall_at(5) == 0.0
    assert report.recall_at(8) == 0.0
    assert report.mean_mrr() == 0.0
    assert report.ndcg(8) == 0.0
    assert report.answer_accuracy() == {"n": 1, "correct": 0, "accuracy": 0.0}
    assert report.abstention_precision() == 0.0
    assert report.abstention_recall() == 0.0
    restricted = report.restricted_summary()
    assert restricted["restricted_filtered"]["primary_recall@8"] == 0.0
    assert restricted["restricted_filtered"]["primary_answer_correct"] == 0
    assert report.leaks() == []

    gate = evaluate_acl_gate(report)

    assert gate.passed is True, f"floor metrics must not affect the gate -- reason: {gate.reason!r}"
    assert gate.passed == report.passed()


# -- case 4: perfect metrics, one leak ---------------------------------------

def _perfect_metrics_one_leak_report() -> RetrievalReport:
    """Every headline retrieval metric at its best possible value (1.0),
    plus exactly one leak -- isolated on a no_reader outcome that no rate
    metric reads (recall/mrr/ndcg/answer_accuracy only look at
    question_class == "answerable"; abstention_recall only at
    "unanswerable"; restricted_summary only at restricted_filtered/
    restricted_unfiltered -- see each method's own filter, retrieval.py).
    The other half of "regardless of every other metric": metrics as good
    as they can be, gate must still say FAIL."""
    return RetrievalReport(outcomes=[
        QuestionOutcome(
            question_id="q_perfect_ans", question_class="answerable", subtype="single_hop",
            principal_id="p1", is_gold_principal=True,
            retrieved_ids=["c1"], gold_ids={"c1"},
            answer_correct=True, abstained=False,
            leaked_chunk_ids=[],
        ),
        QuestionOutcome(
            question_id="q_perfect_ua", question_class="unanswerable", subtype="absent",
            principal_id="p2", is_gold_principal=True,
            abstained=True, abstention_reason="no_results", abstention_correct=True,
            leaked_chunk_ids=[],
        ),
        QuestionOutcome(
            question_id="q_perfect_re_f", question_class="restricted", subtype="restricted_filtered",
            principal_id="p_alice", is_gold_principal=True,
            retrieved_ids=["c2"], gold_ids={"c2"},
            answer_correct=True, abstained=False,
            leaked_chunk_ids=[],
        ),
        QuestionOutcome(
            question_id="q_perfect_re_f", question_class="restricted", subtype="restricted_filtered",
            principal_id="p_ben", is_gold_principal=False,
            retrieved_ids=[], abstained=True,
            leaked_chunk_ids=[],
        ),
        QuestionOutcome(
            question_id="q_perfect_re_u", question_class="restricted", subtype="restricted_unfiltered",
            principal_id="p_alice", is_gold_principal=True,
            retrieved_ids=["c3"], gold_ids={"c3"},
            answer_correct=True, abstained=False,
            leaked_chunk_ids=[],
        ),
        QuestionOutcome(
            question_id="q_perfect_re_u", question_class="restricted", subtype="restricted_unfiltered",
            principal_id="p_ben", is_gold_principal=False,
            retrieved_ids=[], abstained=True,
            leaked_chunk_ids=[],
        ),
        # the one leak -- a no_reader principal, checked by no rate metric
        # asserted above.
        QuestionOutcome(
            question_id="q_perfect_nr", question_class="restricted", subtype="no_reader",
            principal_id="p_dan", is_gold_principal=False,
            retrieved_ids=["leak_chunk"], abstained=False,
            leaked_chunk_ids=["leak_chunk"],
        ),
    ])


def test_perfect_metrics_one_leak_gate_fails():
    report = _perfect_metrics_one_leak_report()

    assert report.recall_at(5) == 1.0
    assert report.recall_at(8) == 1.0
    assert report.mean_mrr() == 1.0
    assert report.ndcg(8) == 1.0
    assert report.answer_accuracy() == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert report.abstention_precision() == 1.0
    assert report.abstention_recall() == 1.0
    restricted = report.restricted_summary()
    assert restricted["restricted_filtered"]["primary_recall@8"] == 1.0
    assert restricted["restricted_filtered"]["primary_answer_correct"] == 1
    assert restricted["restricted_unfiltered"]["primary_recall@8"] == 1.0
    assert restricted["restricted_unfiltered"]["primary_answer_correct"] == 1
    assert len(report.leaks()) == 1

    gate = evaluate_acl_gate(report)

    assert gate.passed is False, f"perfect metrics must not save the gate -- reason: {gate.reason!r}"
    assert gate.passed == report.passed()


# -- parity against the real, non-synthetic run ------------------------------

def test_gate_verdict_matches_passed_on_the_real_74_outcome_rescore():
    """runs/ev01_gate_source/eval/outcomes.jsonl -- EV-01's real 74-outcome
    live run, the same source Stage 5's gate rescore reads -- confirming
    the policy did not change the decision on the one real (non-synthetic)
    run this repo has. Skipped, not failed, when absent: outcomes.jsonl is
    evaluator-only and gitignored (see eval/outcomes.py's module docstring
    and 3a410f8's commit message) -- it will not exist after a fresh clone,
    and this file's job is proving the policy, not regenerating local run
    state."""
    path = REPO_ROOT / "runs" / "ev01_gate_source" / "eval" / "outcomes.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not present locally (evaluator-only, gitignored)")
    report = eval_outcomes.load(path).report
    assert len(report.outcomes) == 74

    gate = evaluate_acl_gate(report)

    assert gate.passed == report.passed()

"""Unit tests for eval/retrieval.py's P1-09 additions: the provenance
guard, gold-question -> principal(s) expansion, the runner, and (step 2)
per-class scoring -- answer accuracy, reason-aware abstention correctness,
and precise ACL-conformance leak detection.

FakeOrchestrator/FakeStore/FakeDocs stand in for the live index, docstore,
and a real AnswerOrchestrator -- no network calls, and scripted enough to
pin exact behaviour. recall_at/mean_mrr/ndcg already have their own
coverage via the metric primitives in eval/metrics.py; this file is about
the NEW pieces.
"""

from __future__ import annotations

import pytest

from isc.common.config import Settings
from isc.common.errors import GoldProvenanceMismatch
from isc.eval.retrieval import (
    QuestionOutcome,
    RetrievalEvalResult,
    RetrievalReport,
    _abstention_correct,
    _expected_to_abstain,
    _principals_for,
    answer_contains_gold,
    check_provenance,
    run,
)
from isc.index.chunker import settings_fingerprint
from isc.models.acl import AclSet, Principal, Sensitivity
from isc.models.answer import AbstentionReason, Answer
from isc.models.chunk import Chunk, ScoredChunk


# -- fakes --------------------------------------------------------------

class FakeStore:
    def __init__(self, settings_fp: str | None) -> None:
        self._fp = settings_fp
        self.path = "fake_store.pkl"

    def settings_fingerprint(self) -> str | None:
        return self._fp


class FakeDocs:
    def __init__(self, hashes: list[str]) -> None:
        self._hashes = hashes

    def content_hashes(self) -> list[str]:
        return self._hashes


class FakeOrchestrator:
    """Records every (question, principal.id) it was asked, and returns a
    scripted Answer per principal id -- default is a bare grounded answer
    if the caller doesn't care what comes back for a given principal."""

    def __init__(self, answers: dict[str, Answer] | None = None) -> None:
        self._answers = answers or {}
        self.calls: list[tuple[str, str]] = []

    def ask(self, question: str, principal: Principal) -> Answer:
        self.calls.append((question, principal.id))
        if principal.id in self._answers:
            return self._answers[principal.id]
        return Answer(question=question, text="a grounded answer", abstained=False)


def _users(*ids: str) -> dict[str, Principal]:
    return {uid: Principal(id=uid) for uid in ids}


# -- check_provenance(): settings_fingerprint ---------------------------

def _matching_provenance() -> dict:
    settings = Settings()
    fp = settings_fingerprint(settings.chunk)
    return {"settings_fingerprint": fp, "corpus_fingerprint": "corpus:abc123"}, settings


def test_check_provenance_passes_when_both_fingerprints_match():
    provenance, settings = _matching_provenance()
    store = FakeStore(settings_fp=provenance["settings_fingerprint"])
    docs = FakeDocs(hashes=["h1", "h2"])

    # No exception -- but corpus_fingerprint must also match, so patch the
    # expected corpus fingerprint computation via matching input: easiest
    # is to compute what corpus_fingerprint(["h1","h2"]) actually is and
    # record that instead of a made-up string.
    from isc.common.ids import corpus_fingerprint
    provenance["corpus_fingerprint"] = corpus_fingerprint(docs.content_hashes())

    check_provenance(provenance, settings, store, docs)  # must not raise


def test_check_provenance_raises_on_settings_fingerprint_mismatch():
    provenance, settings = _matching_provenance()
    provenance["settings_fingerprint"] = "chunk_settings:not_the_real_one"
    store = FakeStore(settings_fp="chunk_settings:not_the_real_one")
    docs = FakeDocs(hashes=["h1"])

    with pytest.raises(GoldProvenanceMismatch, match="settings_fingerprint"):
        check_provenance(provenance, settings, store, docs)


def test_check_provenance_raises_when_gold_settings_fingerprint_disagrees_with_live_index():
    """The gold's recorded value matches the CURRENT settings, but the live
    index on disk was built under a different settings_fingerprint -- e.g.
    the index was rebuilt with a changed ChunkSettings after gold was last
    regenerated. This must fail even though provenance["settings_fingerprint"]
    == settings_fingerprint(settings.chunk)."""
    provenance, settings = _matching_provenance()
    store = FakeStore(settings_fp="chunk_settings:stale_index_value")
    docs = FakeDocs(hashes=["h1"])

    with pytest.raises(GoldProvenanceMismatch, match="live index"):
        check_provenance(provenance, settings, store, docs)


def test_check_provenance_raises_when_live_index_has_no_settings_fingerprint():
    provenance, settings = _matching_provenance()
    store = FakeStore(settings_fp=None)
    docs = FakeDocs(hashes=["h1"])

    with pytest.raises(GoldProvenanceMismatch, match="no settings_fingerprint"):
        check_provenance(provenance, settings, store, docs)


def test_check_provenance_raises_on_corpus_fingerprint_mismatch():
    provenance, settings = _matching_provenance()
    store = FakeStore(settings_fp=provenance["settings_fingerprint"])
    docs = FakeDocs(hashes=["h1", "h2"])
    provenance["corpus_fingerprint"] = "corpus:definitely_not_the_real_one"

    with pytest.raises(GoldProvenanceMismatch, match="corpus_fingerprint"):
        check_provenance(provenance, settings, store, docs)


def test_check_provenance_corpus_mismatch_fires_even_when_settings_match():
    """Exactly the gap docs/adr/0007 describes: a content-only corpus edit
    (e.g. renaming a supplier across 5 PDFs) never moves
    settings_fingerprint at all -- only corpus_fingerprint catches it."""
    provenance, settings = _matching_provenance()
    store = FakeStore(settings_fp=provenance["settings_fingerprint"])
    docs_before = FakeDocs(hashes=["h1", "h2", "h3"])
    from isc.common.ids import corpus_fingerprint
    provenance["corpus_fingerprint"] = corpus_fingerprint(docs_before.content_hashes())

    # The corpus changed (one document's content_sha256 is now different)
    # but chunk settings did not.
    docs_after = FakeDocs(hashes=["h1", "h2", "CHANGED"])

    with pytest.raises(GoldProvenanceMismatch, match="corpus_fingerprint"):
        check_provenance(provenance, settings, store, docs_after)


# -- _principals_for() ---------------------------------------------------

def test_principals_for_answerable_question_is_just_its_principal():
    q = {"question_class": "answerable", "subtype": "single_hop", "principal": "u_alice"}
    assert _principals_for(q) == ["u_alice"]


def test_principals_for_unanswerable_question_is_just_its_principal():
    q = {"question_class": "unanswerable", "subtype": "absent", "principal": "u_chen"}
    assert _principals_for(q) == ["u_chen"]


def test_principals_for_restricted_filtered_returns_principal_then_principal_b():
    q = {
        "question_class": "restricted", "subtype": "restricted_filtered",
        "principal": "u_alice", "principal_b": "u_ben",
    }
    assert _principals_for(q) == ["u_alice", "u_ben"]


def test_principals_for_restricted_unfiltered_returns_principal_then_principal_b():
    q = {
        "question_class": "restricted", "subtype": "restricted_unfiltered",
        "principal": "u_alice", "principal_b": "u_ben",
    }
    assert _principals_for(q) == ["u_alice", "u_ben"]


def test_principals_for_no_reader_returns_all_seven_principals_checked():
    checked = ["u_alice", "u_ben", "u_chen", "u_dara", "u_ewan", "u_frank", "u_gita"]
    q = {
        "question_class": "restricted", "subtype": "no_reader",
        "principal": None, "principals_checked": checked,
    }
    result = _principals_for(q)
    assert result == checked
    assert len(result) == 7


# -- run() ----------------------------------------------------------------

def _q(qid: str, cls: str, subtype: str, **extra) -> dict:
    base = {
        "id": qid, "text": f"question text for {qid}", "question_class": cls,
        "subtype": subtype, "gold_chunk_ids": [], "principal": "u_alice",
    }
    base.update(extra)
    return base


def test_run_asks_each_question_as_its_gold_principal():
    """The specific thing P1-09's WBS item warns about: running as an
    unrestricted superuser instead of the gold principal would make every
    number look better and test nothing. Verified by inspecting the
    Principal actually passed to orchestrator.ask(), not just that some
    answer came back."""
    questions = [
        _q("q_sh_01", "answerable", "single_hop", principal="u_alice"),
        _q("q_sh_02", "answerable", "single_hop", principal="u_chen"),
        _q("q_ua_01", "unanswerable", "absent", principal="u_ewan",
           expected_behavior="abstain"),
    ]
    orch = FakeOrchestrator()
    users = _users("u_alice", "u_chen", "u_ewan")

    result = run(questions, users, orch)

    assert orch.calls == [
        ("question text for q_sh_01", "u_alice"),
        ("question text for q_sh_02", "u_chen"),
        ("question text for q_ua_01", "u_ewan"),
    ]
    assert len(result.report.outcomes) == 3
    assert result.failed == []


def test_run_asks_restricted_question_as_both_principal_and_principal_b():
    q = _q("q_re_01", "restricted", "restricted_filtered",
           principal="u_alice", principal_b="u_ben", gold_chunk_ids=["chk_a"])
    orch = FakeOrchestrator()
    users = _users("u_alice", "u_ben")

    result = run([q], users, orch)

    assert orch.calls == [
        ("question text for q_re_01", "u_alice"),
        ("question text for q_re_01", "u_ben"),
    ]
    outcomes = result.report.outcomes
    assert len(outcomes) == 2
    assert {o.principal_id for o in outcomes} == {"u_alice", "u_ben"}
    # Both outcomes share the same question_id and gold_ids -- they are the
    # same gold question, run as two different principals.
    assert all(o.question_id == "q_re_01" for o in outcomes)
    assert all(o.gold_ids == {"chk_a"} for o in outcomes)


def test_run_asks_no_reader_question_as_all_seven_checked_principals():
    checked = ["u_alice", "u_ben", "u_chen", "u_dara", "u_ewan", "u_frank", "u_gita"]
    q = _q("q_re_09", "restricted", "no_reader", principal=None, principals_checked=checked)
    orch = FakeOrchestrator()
    users = _users(*checked)

    result = run([q], users, orch)

    assert {p for _, p in orch.calls} == set(checked)
    assert len(result.report.outcomes) == 7
    assert {o.principal_id for o in result.report.outcomes} == set(checked)


def test_run_records_retrieved_ids_and_reason_even_when_abstained():
    """The whole point of Answer.abstain()'s new supporting= parameter:
    retrieval happened before the model declined, and the runner must not
    lose that just because the outcome was an abstention."""
    from isc.models.acl import AclSet
    from isc.models.chunk import Chunk, ScoredChunk

    chunk = Chunk(
        id="chk_x", document_id="doc_x", ordinal=0, text="some text",
        acl=AclSet(allow_terms=frozenset({"everyone:*"})),
    )
    hit = ScoredChunk(chunk=chunk, score=0.5)
    abstained_answer = Answer.abstain(
        "q", AbstentionReason.INSUFFICIENT_CONTEXT, supporting=[hit],
    )
    q = _q("q_ua_01", "unanswerable", "absent", gold_chunk_ids=[], expected_behavior="abstain")
    orch = FakeOrchestrator(answers={"u_alice": abstained_answer})
    users = _users("u_alice")

    result = run([q], users, orch)

    outcome = result.report.outcomes[0]
    assert outcome.abstained is True
    assert outcome.abstention_reason == "insufficient_context"
    assert outcome.retrieved_ids == ["chk_x"]


def test_run_isolates_one_questions_failure_from_the_rest():
    """A provider error on one question must not abort scoring the other
    fifty-five -- same per-item isolation discipline as index/pipeline.py's
    IndexResult.failed and eval/pipeline.py's ExtractionEvalResult.skipped."""

    class FlakyOrchestrator:
        def ask(self, question, principal):
            if principal.id == "u_chen":
                raise RuntimeError("provider timeout")
            return Answer(question=question, text="ok", abstained=False)

    questions = [
        _q("q_sh_01", "answerable", "single_hop", principal="u_alice"),
        _q("q_sh_02", "answerable", "single_hop", principal="u_chen"),
        _q("q_sh_03", "answerable", "single_hop", principal="u_alice"),
    ]
    users = _users("u_alice", "u_chen")

    result = run(questions, users, FlakyOrchestrator())

    assert isinstance(result, RetrievalEvalResult)
    assert len(result.report.outcomes) == 2
    assert [o.question_id for o in result.report.outcomes] == ["q_sh_01", "q_sh_03"]
    assert result.failed == [("q_sh_02", "u_chen", "provider timeout")]


# -- answer_contains_gold(): reuses eval/normalise.py, adds containment ----

def test_answer_contains_gold_returns_none_for_null_gold_answer():
    assert answer_contains_gold(None, "any text at all") is None


def test_answer_contains_gold_matches_a_plain_string_value():
    assert answer_contains_gold("A. Tan", "The buyer contact is A. Tan [1].") is True


def test_answer_contains_gold_false_when_plain_string_value_is_absent():
    assert answer_contains_gold("A. Tan", "The buyer contact is J. Ruiz [1].") is False


def test_answer_contains_gold_matches_decimal_value_despite_different_grouping():
    assert answer_contains_gold("1,536.37", "We paid 1536.37 SGD per unit [1].") is True


def test_answer_contains_gold_matches_decimal_value_verbatim():
    assert answer_contains_gold("1,536.37", "We paid 1,536.37 SGD per unit [1].") is True


def test_answer_contains_gold_false_when_decimal_value_differs():
    assert answer_contains_gold("1,536.37", "We paid 1,536.38 SGD per unit [1].") is False


def test_answer_contains_gold_matches_a_date_value():
    assert answer_contains_gold("26/04/2025", "The PO date was 26/04/2025 [1].") is True


def test_answer_contains_gold_false_when_date_value_differs():
    assert answer_contains_gold("26/04/2025", "The PO date was 27/04/2025 [1].") is False


def test_answer_contains_gold_checks_the_total_not_the_breakdown():
    """Regression for the real q_cd_01 bug found running the P1-07 sample:
    the model summed only ONE of two documents' totals into its answer. A
    check that also accepted a breakdown entry as a match would have let
    this pass by coincidence -- "breakdown" is excluded from what counts
    as required (see _leaf_values()'s docstring), so only "total" is
    checked."""
    gold_answer = {
        "total": "2,972,338.10", "currency": "SGD",
        "breakdown": [
            {"document": "po_000.pdf", "total_amount": "392,589.57"},
            {"document": "po_017.pdf", "total_amount": "2,579,748.53"},
        ],
    }
    wrong = "The total spent with Omron Electronics Asia is 2,579,748.53 SGD [8]."
    assert answer_contains_gold(gold_answer, wrong) is False

    correct = "The total spent with Omron Electronics Asia is 2,972,338.10 SGD [1][8]."
    assert answer_contains_gold(gold_answer, correct) is True


def test_answer_contains_gold_requires_every_value_in_a_multi_entry_list():
    gold_answer = [
        {"document": "po_000.pdf", "line_number": 10, "unit_price": "1,536.37", "currency": "SGD"},
        {"document": "po_014.pdf", "line_number": 30, "unit_price": "174.53", "currency": "SGD"},
    ]
    only_one = "We paid 1,536.37 per unit on the first order [1]."
    assert answer_contains_gold(gold_answer, only_one) is False

    both = "We paid 1,536.37 on the first order and 174.53 on the second [1][2]."
    assert answer_contains_gold(gold_answer, both) is True


def test_answer_contains_gold_ambiguous_suppliers_shape_requires_both_names():
    """Matches the ambiguous class's own explicit design intent: surface
    both entities, not silently pick one."""
    gold_answer = {
        "note": "two distinct Kestrel Industrial entities placed orders",
        "suppliers": [
            {"supplier_name": "Kestrel Industrial Pneumatics GmbH", "document": "po_004.pdf"},
            {"supplier_name": "Kestrel Industrial AG", "document": "po_007.pdf"},
        ],
    }
    only_one = "We ordered from Kestrel Industrial AG [1]."
    assert answer_contains_gold(gold_answer, only_one) is False

    both = "We ordered from both Kestrel Industrial Pneumatics GmbH and Kestrel Industrial AG [1][2]."
    assert answer_contains_gold(gold_answer, both) is True


def test_answer_contains_gold_ignores_document_filenames():
    """A filename like "po_004.pdf" never appears in natural-language
    prose -- requiring it would fail every correct answer."""
    gold_answer = {"total": "100.00", "currency": "SGD",
                    "breakdown": [{"document": "po_004.pdf", "total_amount": "100.00"}]}
    assert answer_contains_gold(gold_answer, "The total is 100.00 SGD [1].") is True


# -- _abstention_correct(): reason-aware, not just abstained=True ----------

def test_abstention_correct_true_when_absent_question_declines_with_insufficient_context():
    assert _abstention_correct("abstain", True, "insufficient_context") is True


def test_abstention_correct_true_for_no_results_or_low_support_too():
    assert _abstention_correct("abstain", True, "no_results") is True
    assert _abstention_correct("abstain", True, "low_support") is True


def test_abstention_correct_false_when_abstained_for_an_unrelated_reason():
    """An ATTRIBUTION_MISMATCH (or UNGROUNDED_DRAFT) abstention happened to
    abstain, but for a reason unrelated to the question genuinely being
    unanswerable -- the model tried to answer and failed, it did not
    correctly recognise "no answer here". Only one of the two looks-like-
    abstained outcomes is a pass."""
    assert _abstention_correct("abstain", True, "attribution_mismatch") is False
    assert _abstention_correct("abstain", True, "ungrounded") is False


def test_abstention_correct_false_when_not_abstained_at_all():
    assert _abstention_correct("abstain", False, None) is False


def test_abstention_correct_always_false_for_underspecified():
    """Scored as a failure honestly, per the explicit instruction not to
    soften this -- even a "good" reason does not count, because nothing in
    AbstentionReason represents "asked for clarification", so there is
    nothing to check FOR."""
    assert _abstention_correct("abstain_with_clarification", True, "insufficient_context") is False
    assert _abstention_correct("abstain_with_clarification", False, None) is False


# -- RetrievalReport: reason-aware abstention_recall(), by-subtype views ---

def test_abstention_recall_uses_reason_aware_correctness_not_bare_abstained():
    outcomes = [
        QuestionOutcome(question_id="q1", question_class="unanswerable", subtype="absent",
                         abstained=True, abstention_correct=True),
        # Abstained, but for the wrong reason -- must not count as a hit.
        QuestionOutcome(question_id="q2", question_class="unanswerable", subtype="absent",
                         abstained=True, abstention_correct=False),
    ]
    report = RetrievalReport(outcomes=outcomes)
    assert report.abstention_recall() == 0.5


def test_abstention_by_subtype_reports_underspecified_separately_from_absent():
    outcomes = [
        QuestionOutcome(question_id="q1", question_class="unanswerable", subtype="absent",
                         abstention_correct=True),
        QuestionOutcome(question_id="q2", question_class="unanswerable", subtype="out_of_scope",
                         abstention_correct=True),
        QuestionOutcome(question_id="q3", question_class="unanswerable", subtype="underspecified",
                         abstention_correct=False),
    ]
    report = RetrievalReport(outcomes=outcomes)
    by_subtype = report.abstention_by_subtype()
    assert by_subtype["absent"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert by_subtype["out_of_scope"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert by_subtype["underspecified"] == {"n": 1, "correct": 0, "accuracy": 0.0}


# -- leak detection: principal.may_read() against its OWN AclSet -----------

def _chunk_with_acl(cid: str, allow_terms: frozenset, **kw) -> Chunk:
    return Chunk(id=cid, document_id=f"doc_{cid}", ordinal=0, text="some chunk text",
                 acl=AclSet(allow_terms=allow_terms, **kw))


def test_run_flags_a_leak_when_a_retrieved_chunk_fails_the_principals_own_acl_check():
    """Constructed directly rather than through a live store: whatever the
    reason a chunk this principal cannot read ends up in answer.supporting,
    run() must catch it. This is a direct may_read() check against the
    chunk's own AclSet -- not a comparison against what any other
    principal retrieved."""
    forbidden = _chunk_with_acl("chk_secret", frozenset({"user:someone_else"}))
    answer = Answer(question="q", text="an answer naming a real chunk", abstained=False,
                     supporting=[ScoredChunk(chunk=forbidden, score=0.5)])
    q = _q("q_re_01", "restricted", "restricted_filtered",
           principal="u_ben", principal_b="u_other", gold_chunk_ids=[])
    orch = FakeOrchestrator(answers={"u_ben": answer})

    result = run([q], _users("u_ben", "u_other"), orch)

    outcome = next(o for o in result.report.outcomes if o.principal_id == "u_ben")
    assert outcome.leaked_chunk_ids == ["chk_secret"]
    assert result.report.leaks() == [outcome]
    assert result.report.passed() is False


def test_run_does_not_flag_leak_for_a_chunk_the_principal_may_legitimately_read():
    readable = _chunk_with_acl("chk_ok", frozenset({"user:u_alice"}))
    answer = Answer(question="q", text="an answer", abstained=False,
                     supporting=[ScoredChunk(chunk=readable, score=0.9)])
    q = _q("q_re_01", "restricted", "restricted_filtered",
           principal="u_alice", principal_b="u_ben", gold_chunk_ids=["chk_ok"])
    orch = FakeOrchestrator(answers={"u_alice": answer})

    result = run([q], _users("u_alice", "u_ben"), orch)

    alice_outcome = next(o for o in result.report.outcomes if o.principal_id == "u_alice")
    assert alice_outcome.leaked_chunk_ids == []
    assert result.report.passed() is True


def test_run_does_not_treat_bens_smaller_result_set_as_a_leak():
    """The thing this check must NOT be: 'a chunk alice got that ben did
    not' is correct, expected behaviour for a restricted question, not a
    leak. Ben legitimately retrieving nothing at all must score clean."""
    alice_chunk = _chunk_with_acl("chk_alice_only", frozenset({"user:u_alice"}))
    alice_answer = Answer(question="q", text="alice's answer", abstained=False,
                           supporting=[ScoredChunk(chunk=alice_chunk, score=0.9)])
    ben_answer = Answer.abstain("q", AbstentionReason.NO_RESULTS)  # supporting=[]
    q = _q("q_re_01", "restricted", "restricted_filtered",
           principal="u_alice", principal_b="u_ben", gold_chunk_ids=["chk_alice_only"])
    orch = FakeOrchestrator(answers={"u_alice": alice_answer, "u_ben": ben_answer})

    result = run([q], _users("u_alice", "u_ben"), orch)

    ben_outcome = next(o for o in result.report.outcomes if o.principal_id == "u_ben")
    assert ben_outcome.retrieved_ids == []
    assert ben_outcome.leaked_chunk_ids == []
    assert result.report.passed() is True


def test_leaks_by_subtype_groups_leak_counts_separately_by_retrieval_path():
    """The whole point of the unfiltered variant: a pre-search filter
    degrading into a post-search one would only show up here, not on the
    filtered path -- so the two must be counted separately, not folded
    into one aggregate."""
    forbidden = _chunk_with_acl("chk_x", frozenset({"user:nobody"}))
    leaking_answer = Answer(question="q", text="oops", abstained=False,
                             supporting=[ScoredChunk(chunk=forbidden, score=0.5)])
    q_filtered = _q("q_re_01", "restricted", "restricted_filtered",
                     principal="u_ben", principal_b="u_other", gold_chunk_ids=[])
    q_unfiltered = _q("q_re_10", "restricted", "restricted_unfiltered",
                       principal="u_ben", principal_b="u_other", gold_chunk_ids=[])
    orch = FakeOrchestrator(answers={"u_ben": leaking_answer})

    result = run([q_filtered, q_unfiltered], _users("u_ben", "u_other"), orch)

    assert result.report.leaks_by_subtype() == {
        "restricted_filtered": 1, "restricted_unfiltered": 1,
    }


# -- report-facing aggregates: recall_by_subtype, answer_accuracy, --------
# -- answerable_failures, restricted_summary, no_reader_summary -----------

def _answerable_outcome(qid, subtype, retrieved_ids, gold_ids, answer_correct) -> QuestionOutcome:
    return QuestionOutcome(
        question_id=qid, question_class="answerable", subtype=subtype,
        retrieved_ids=retrieved_ids, gold_ids=set(gold_ids), answer_correct=answer_correct,
    )


def test_recall_by_subtype_reports_cross_document_and_line_item_separately():
    outcomes = [
        _answerable_outcome("q_cd_01", "cross_document", ["chk_a"], {"chk_a", "chk_b"}, True),
        _answerable_outcome("q_li_01", "line_item", ["chk_x"], {"chk_x"}, True),
    ]
    report = RetrievalReport(outcomes=outcomes)
    by_subtype = report.recall_by_subtype()

    assert by_subtype["cross_document"]["n"] == 1
    assert by_subtype["cross_document"]["recall@8"] == 0.5  # only 1 of 2 gold chunks retrieved
    assert by_subtype["line_item"]["n"] == 1
    assert by_subtype["line_item"]["recall@8"] == 1.0


def test_answer_accuracy_counts_only_checkable_answerable_outcomes():
    outcomes = [
        _answerable_outcome("q1", "single_hop", ["chk_a"], {"chk_a"}, True),
        _answerable_outcome("q2", "single_hop", ["chk_b"], {"chk_b"}, False),
        # answer_correct=None (e.g. gold_answer was null) must not count
        # toward the denominator either way.
        _answerable_outcome("q3", "single_hop", ["chk_c"], {"chk_c"}, None),
    ]
    report = RetrievalReport(outcomes=outcomes)
    assert report.answer_accuracy() == {"n": 2, "correct": 1, "accuracy": 0.5}


def test_answerable_failures_diagnoses_retrieval_vs_generation():
    """Two different defects, and the report must be able to tell them
    apart: gold chunks retrieved but the answer still wrong points at
    generation/grounding; gold chunks missing points at retrieval."""
    outcomes = [
        # Gold chunk WAS retrieved -- a generation/grounding failure.
        QuestionOutcome(question_id="q_cd_02", question_class="answerable",
                         subtype="cross_document", retrieved_ids=["chk_a", "chk_b"],
                         gold_ids={"chk_a", "chk_b"}, answer_correct=False,
                         abstained=True, abstention_reason="attribution_mismatch"),
        # Gold chunk was NOT retrieved -- a retrieval failure.
        QuestionOutcome(question_id="q_li_02", question_class="answerable",
                         subtype="line_item", retrieved_ids=["chk_x"],
                         gold_ids={"chk_missing"}, answer_correct=False),
        # Correct answer -- must not appear in the failure list at all.
        QuestionOutcome(question_id="q_sh_01", question_class="answerable",
                         subtype="single_hop", retrieved_ids=["chk_y"],
                         gold_ids={"chk_y"}, answer_correct=True),
    ]
    report = RetrievalReport(outcomes=outcomes)
    failures = {f["question_id"]: f for f in report.answerable_failures()}

    assert set(failures) == {"q_cd_02", "q_li_02"}
    assert failures["q_cd_02"]["gold_chunks_retrieved"] is True
    assert failures["q_cd_02"]["diagnosis"].startswith("generation/grounding")
    assert failures["q_cd_02"]["abstention_reason"] == "attribution_mismatch"
    assert failures["q_li_02"]["gold_chunks_retrieved"] is False
    assert failures["q_li_02"]["diagnosis"].startswith("retrieval")


def test_restricted_summary_reports_filtered_and_unfiltered_separately():
    outcomes = [
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", principal_id="u_alice",
                         is_gold_principal=True, retrieved_ids=["chk_a"],
                         gold_ids={"chk_a"}, answer_correct=True),
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", principal_id="u_ben",
                         is_gold_principal=False, retrieved_ids=[], abstained=True),
        QuestionOutcome(question_id="q_re_10", question_class="restricted",
                         subtype="restricted_unfiltered", principal_id="u_alice",
                         is_gold_principal=True, retrieved_ids=["chk_b"],
                         gold_ids={"chk_b"}, answer_correct=True),
        # Unfiltered ben: leaked a chunk his own ACL should have excluded.
        QuestionOutcome(question_id="q_re_10", question_class="restricted",
                         subtype="restricted_unfiltered", principal_id="u_ben",
                         is_gold_principal=False, retrieved_ids=["chk_c"],
                         leaked_chunk_ids=["chk_c"]),
    ]
    report = RetrievalReport(outcomes=outcomes)
    summary = report.restricted_summary()

    assert summary["restricted_filtered"]["n_pairs"] == 1
    assert summary["restricted_filtered"]["primary_recall@8"] == 1.0
    assert summary["restricted_filtered"]["primary_answer_correct"] == 1
    assert summary["restricted_filtered"]["secondary_correctly_empty_and_abstained"] == 1
    assert summary["restricted_filtered"]["leaks"] == 0

    assert summary["restricted_unfiltered"]["primary_answer_correct"] == 1
    assert summary["restricted_unfiltered"]["secondary_correctly_empty_and_abstained"] == 0
    assert summary["restricted_unfiltered"]["leaks"] == 1


def test_no_reader_summary_flags_any_principal_with_results():
    outcomes = [
        QuestionOutcome(question_id="q_re_09", question_class="restricted",
                         subtype="no_reader", principal_id=uid, retrieved_ids=[])
        for uid in ["u_alice", "u_ben", "u_chen"]
    ]
    report = RetrievalReport(outcomes=outcomes)
    assert report.no_reader_summary() == {
        "n_principals_checked": 3, "all_empty": True, "principals_with_results": [],
    }

    outcomes[1].retrieved_ids = ["chk_leaked"]
    report2 = RetrievalReport(outcomes=outcomes)
    summary = report2.no_reader_summary()
    assert summary["all_empty"] is False
    assert summary["principals_with_results"] == ["u_ben"]


# -- _expected_to_abstain() / abstention_precision(): fixed definition -----

def test_expected_to_abstain_true_for_any_unanswerable_subtype():
    o = QuestionOutcome(question_id="q1", question_class="unanswerable", subtype="absent")
    assert _expected_to_abstain(o) is True


def test_expected_to_abstain_true_for_restricted_non_gold_principal():
    """Ben, denied by design -- or any of no_reader's 7 checked principals,
    which all have is_gold_principal=False for the same reason (see
    run(): q["principal"] is None for no_reader, and no principal_id ever
    equals None)."""
    ben = QuestionOutcome(question_id="q_re_01", question_class="restricted",
                           subtype="restricted_filtered", is_gold_principal=False)
    no_reader = QuestionOutcome(question_id="q_re_09", question_class="restricted",
                                 subtype="no_reader", is_gold_principal=False)
    assert _expected_to_abstain(ben) is True
    assert _expected_to_abstain(no_reader) is True


def test_expected_to_abstain_false_for_restricted_gold_principal():
    """Alice IS supposed to get a real answer -- an abstention on her side
    (e.g. a retrieval miss) is a real failure, not a legitimate denial."""
    alice = QuestionOutcome(question_id="q_re_13", question_class="restricted",
                             subtype="restricted_unfiltered", is_gold_principal=True)
    assert _expected_to_abstain(alice) is False


def test_expected_to_abstain_false_for_answerable():
    o = QuestionOutcome(question_id="q_cd_02", question_class="answerable",
                         subtype="cross_document")
    assert _expected_to_abstain(o) is False


def test_abstention_precision_does_not_penalise_correct_denials():
    """Regression for the exact bug found on the P1-09 full run: the old
    definition (question_class == "unanswerable" only) scored every one
    of ben's correct denials as imprecision. 8 ben-side abstentions + 1
    genuinely unanswerable abstention, all legitimate, must read as 1.0,
    not be dragged down for denying restricted questions correctly."""
    outcomes = [
        QuestionOutcome(question_id=f"q_re_{i}", question_class="restricted",
                         subtype="restricted_filtered", is_gold_principal=False, abstained=True)
        for i in range(8)
    ] + [
        QuestionOutcome(question_id="q_ua_01", question_class="unanswerable",
                         subtype="absent", abstained=True),
    ]
    report = RetrievalReport(outcomes=outcomes)
    assert report.abstention_precision() == 1.0


def test_abstention_precision_penalises_an_unexpected_abstention():
    """An answerable question that abstained, and a restricted
    gold-principal (alice) who abstained instead of answering, both count
    against precision -- these are real failures, not legitimate denials."""
    outcomes = [
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", is_gold_principal=False, abstained=True),
        QuestionOutcome(question_id="q_cd_02", question_class="answerable",
                         subtype="cross_document", abstained=True),
        QuestionOutcome(question_id="q_re_13", question_class="restricted",
                         subtype="restricted_unfiltered", is_gold_principal=True, abstained=True),
    ]
    report = RetrievalReport(outcomes=outcomes)
    # 1 of 3 abstentions was legitimate (ben's denial); the answerable
    # abstention and alice's unexpected one were not.
    assert report.abstention_precision() == pytest.approx(1 / 3)


def test_abstention_precision_matches_the_full_run_shape():
    """Reproduces the P1-09 full run's abstained-outcome mix at reduced
    scale: old definition would read 2/3 (only the unanswerable one and
    ben's denial... no -- old definition counted ONLY question_class ==
    "unanswerable", so 1/3); this definition credits ben's denial too."""
    outcomes = [
        QuestionOutcome(question_id="q_ua_01", question_class="unanswerable",
                         subtype="absent", abstained=True),
        QuestionOutcome(question_id="q_re_01", question_class="restricted",
                         subtype="restricted_filtered", is_gold_principal=False, abstained=True),
        QuestionOutcome(question_id="q_cd_02", question_class="answerable",
                         subtype="cross_document", abstained=True),
    ]
    report = RetrievalReport(outcomes=outcomes)
    assert report.abstention_precision() == pytest.approx(2 / 3)

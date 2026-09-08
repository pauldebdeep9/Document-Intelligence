"""Retrieval + answering harness.

Runs each gold question as its gold principal, so ACL correctness is measured,
not assumed. Four question classes, scored separately (see docs/WBS-P1.md's
P1-09 item):

  answerable    gold passage ids exist; score recall@k, MRR, nDCG, answer accuracy
  unanswerable  ~15% of the set; the only correct behaviour is abstention (or,
                for underspecified, abstain_with_clarification specifically)
  restricted    answerable for user A, invisible to user B; any leak is a hard
                failure that fails the whole run regardless of other metrics
  no_reader     answerable for nobody in the named identity graph; every
                principal checked must come back empty

check_provenance() must be called, and must pass, before run() runs a single
question -- see its own docstring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from isc.answer.orchestrator import AnswerOrchestrator
from isc.common.config import Settings
from isc.common.errors import GoldProvenanceMismatch
from isc.common.ids import corpus_fingerprint
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.eval.metrics import mrr, ndcg_at_k, recall_at_k
from isc.eval.normalise import normalise_date, normalise_decimal, normalise_string
from isc.index.chunker import settings_fingerprint
from isc.models.acl import Principal
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("eval.retrieval")

# Abstention reasons that represent the system genuinely concluding "no
# answer is available" -- as opposed to UNGROUNDED_DRAFT/ATTRIBUTION_MISMATCH,
# which mean the model tried to answer and failed in a way that happens to
# come out as an abstention. Both look like abstained=True; only the first
# kind is a correct abstain on an absent/out_of_scope question. See
# docs/WBS-P1.md's P1-09 briefing and docs/adr/0009.
_CORRECT_ABSTAIN_REASONS = frozenset({"no_results", "low_support", "insufficient_context"})


@dataclass
class QuestionOutcome:
    question_id: str
    question_class: str
    subtype: str = ""
    # Which of the gold question's principal(s) this particular outcome was
    # run as -- a restricted question produces two outcomes sharing one
    # question_id (principal, then principal_b), a no_reader question up to
    # seven, one per name in principals_checked. Every OTHER class produces
    # exactly one outcome per question_id.
    principal_id: str = ""
    # Was this run as q["principal"] specifically -- the ONE principal gold
    # says should get a real answer? False for principal_b (restricted) and
    # for every name in principals_checked (no_reader), which exist to be
    # DENIED, not answered. Generic on purpose: nothing here hardcodes
    # "alice"/"ben" -- see run()'s docstring and the P1-09 briefing on why
    # leak detection must not either.
    is_gold_principal: bool = True
    retrieved_ids: list[str] = field(default_factory=list)
    gold_ids: set[str] = field(default_factory=set)
    abstained: bool = False
    abstention_reason: str | None = None
    answer_text: str = ""
    # Chunk ids the answer actually cited (post-binding -- see
    # answer/citations.py's bind_citations()), not the same thing as
    # retrieved_ids: a citation is a subset of what was retrieved, and the
    # two failing independently is exactly what "score retrieval and
    # answering separately" means.
    citations: list[str] = field(default_factory=list)
    # None = not applicable: gold_answer is null (unanswerable/no_reader),
    # or this outcome is not the gold-principal side of a restricted pair.
    # Otherwise True/False from answer_contains_gold() against answer_text
    # -- computed uniformly whether or not the answer abstained, so an
    # ATTRIBUTION_MISMATCH (or any other) abstention on an answerable
    # question correctly scores False rather than being silently excluded.
    answer_correct: bool | None = None
    # None unless question_class == "unanswerable": True/False against the
    # gold's own expected_behavior for this subtype (absent/out_of_scope
    # expect "abstain"; underspecified expects "abstain_with_clarification",
    # which nothing in this system can currently produce -- see
    # _abstention_correct()). Distinct from `abstained` -- the WRONG reason
    # can still set abstained=True.
    abstention_correct: bool | None = None
    citations_valid: bool = True
    # Chunk ids this principal retrieved that fail principal.may_read()
    # against their OWN AclSet -- not "a chunk alice got that this
    # principal didn't" (that is correct, expected behaviour), a direct
    # ACL-conformance violation. See run()'s computation and the P1-09
    # briefing's correction of the original (comparison-based) design.
    leaked_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalReport:
    outcomes: list[QuestionOutcome] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        rows = [o for o in self.outcomes if o.question_class == "answerable"]
        return sum(recall_at_k(o.retrieved_ids, o.gold_ids, k) for o in rows) / max(len(rows), 1)

    def mean_mrr(self) -> float:
        rows = [o for o in self.outcomes if o.question_class == "answerable"]
        return sum(mrr(o.retrieved_ids, o.gold_ids) for o in rows) / max(len(rows), 1)

    def ndcg(self, k: int = 8) -> float:
        rows = [o for o in self.outcomes if o.question_class == "answerable"]
        return sum(ndcg_at_k(o.retrieved_ids, o.gold_ids, k) for o in rows) / max(len(rows), 1)

    def abstention_precision(self) -> float:
        """Of the outcomes that abstained, how many SHOULD have -- the
        question's expected behaviour was abstention, whatever the class
        (see _expected_to_abstain()). Deliberately NOT scoped to
        question_class == "unanswerable" alone: that was the original
        P1-08/P1-09 framing and it was a bug, not a tuning choice. It
        scored every one of ben's correct denials, and every no_reader
        principal's correct denial, as imprecision -- on the P1-09 full
        run that put 22 of 29 abstained outcomes in the "imprecise"
        bucket for correctly declining a question they were never
        supposed to answer, reading as 0.241 for a run with 0 ACL leaks
        and a clean restricted pass. Recomputed under this definition on
        the exact same outcomes: 0.862 -- the number moved because the
        definition was wrong, not because behaviour changed."""
        abstained = [o for o in self.outcomes if o.abstained]
        if not abstained:
            return 0.0
        return sum(_expected_to_abstain(o) for o in abstained) / len(abstained)

    def abstention_precision_band(self) -> tuple[int, int]:
        """(expected, total) underlying abstention_precision() -- same
        population (every abstained outcome) and the same
        _expected_to_abstain() check, exposed as raw counts rather than a
        pre-divided ratio (EV-02: a rate with nowhere to show its own
        denominator is a rate a reader cannot size). Does not change what
        abstention_precision() means or returns -- an additional view onto
        the same computation, not a replacement."""
        abstained = [o for o in self.outcomes if o.abstained]
        if not abstained:
            return 0, 0
        return sum(_expected_to_abstain(o) for o in abstained), len(abstained)

    def abstention_recall(self) -> float:
        """Of the genuinely unanswerable questions, how many did we abstain
        on FOR THE RIGHT REASON? Uses abstention_correct, not the bare
        `abstained` flag -- abstaining for the wrong reason is not a pass
        (an ATTRIBUTION_MISMATCH abstention on an absent-field question
        happened to abstain, but for a reason unrelated to the field being
        absent; see _abstention_correct() and docs/adr/0009)."""
        unanswerable = [o for o in self.outcomes if o.question_class == "unanswerable"]
        if not unanswerable:
            return 1.0
        return sum(bool(o.abstention_correct) for o in unanswerable) / len(unanswerable)

    def abstention_recall_band(self) -> tuple[int, int]:
        """(correct, total) underlying abstention_recall() -- same
        population (every unanswerable-class outcome) and the same
        abstention_correct check, exposed as raw counts rather than a
        pre-divided ratio (EV-02, same reasoning as
        abstention_precision_band()). Note the n=0 case here is (0, 0), NOT
        (0, 1) or any stand-in for abstention_recall()'s own vacuous-truth
        1.0 return on empty -- this method only exposes counts, it does not
        recompute or reinterpret the rate."""
        unanswerable = [o for o in self.outcomes if o.question_class == "unanswerable"]
        if not unanswerable:
            return 0, 0
        return sum(bool(o.abstention_correct) for o in unanswerable), len(unanswerable)

    def abstention_by_subtype(self) -> dict[str, dict[str, float | int]]:
        """Per unanswerable subtype (absent, out_of_scope, underspecified):
        n, how many scored abstention_correct, and the resulting accuracy.
        Reported separately, not folded into one abstention_recall number,
        so the single known gap (underspecified: expects
        abstain_with_clarification, which the prompt cannot currently
        produce) does not drag down the two subtypes that work."""
        by_subtype: dict[str, list[QuestionOutcome]] = {}
        for o in self.outcomes:
            if o.question_class != "unanswerable":
                continue
            by_subtype.setdefault(o.subtype, []).append(o)
        out: dict[str, dict[str, float | int]] = {}
        for subtype, rows in by_subtype.items():
            correct = sum(bool(o.abstention_correct) for o in rows)
            out[subtype] = {
                "n": len(rows), "correct": correct,
                "accuracy": correct / len(rows) if rows else 0.0,
            }
        return out

    def recall_by_subtype(self) -> dict[str, dict[str, float | int]]:
        """Per answerable subtype: n, recall@5, recall@8, mrr, ndcg@8.
        cross_document and line_item are not singled out in code -- the
        data treats every subtype the same way; the report renderer is
        what decides to surface those two prominently, per the reasons
        they were built to stress in the first place (cross_document
        needs chunks from 2+ documents in one fused ranking; line_item
        needs the right slice of a split table)."""
        by_subtype: dict[str, list[QuestionOutcome]] = {}
        for o in self.outcomes:
            if o.question_class != "answerable":
                continue
            by_subtype.setdefault(o.subtype, []).append(o)
        out: dict[str, dict[str, float | int]] = {}
        for subtype, rows in by_subtype.items():
            n = len(rows)
            out[subtype] = {
                "n": n,
                "recall@5": sum(recall_at_k(o.retrieved_ids, o.gold_ids, 5) for o in rows) / n,
                "recall@8": sum(recall_at_k(o.retrieved_ids, o.gold_ids, 8) for o in rows) / n,
                "mrr": sum(mrr(o.retrieved_ids, o.gold_ids) for o in rows) / n,
                "ndcg@8": sum(ndcg_at_k(o.retrieved_ids, o.gold_ids, 8) for o in rows) / n,
            }
        return out

    def answer_accuracy(self) -> dict[str, float | int]:
        """Answerable-class accuracy: of the questions whose answer could
        be checked against gold (answer_correct is not None -- see
        answer_contains_gold()), how many actually contained the gold
        value. An abstention for ANY reason, including ATTRIBUTION_MISMATCH,
        scores as incorrect here: run() computes answer_correct against
        answer_text unconditionally, and the canned abstention text never
        contains a real gold value, so a mismatch-caused abstention on an
        answerable question correctly counts as an answer failure rather
        than being silently excluded."""
        rows = [
            o for o in self.outcomes
            if o.question_class == "answerable" and o.answer_correct is not None
        ]
        if not rows:
            return {"n": 0, "correct": 0, "accuracy": 0.0}
        correct = sum(1 for o in rows if o.answer_correct)
        return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows)}

    def answerable_failures(self) -> list[dict]:
        """Every answerable question whose answer did not contain the gold
        value, each tagged with which stage the failure implicates:

          gold chunks retrieved, answer wrong -> generation/grounding problem
          gold chunks missing                 -> retrieval problem

        Collapsing these into one "answering is weak" finding would point
        at the wrong stage to fix -- gold_ids is a subset check against
        retrieved_ids (whatever retrieve() actually returned, already
        capped to final_k), not a recall_at_k threshold, so this asks
        exactly "was the evidence even there", separately from whether the
        model used it correctly.
        """
        out = []
        for o in self.outcomes:
            if o.question_class != "answerable" or o.answer_correct is not False:
                continue
            gold_retrieved = bool(o.gold_ids) and o.gold_ids <= set(o.retrieved_ids)
            out.append({
                "question_id": o.question_id,
                "subtype": o.subtype,
                "gold_chunks_retrieved": gold_retrieved,
                "abstained": o.abstained,
                "abstention_reason": o.abstention_reason,
                "diagnosis": (
                    "generation/grounding -- gold chunks were retrieved" if gold_retrieved
                    else "retrieval -- gold chunks were not retrieved"
                ),
            })
        return out

    def restricted_summary(self) -> dict[str, dict]:
        """restricted_filtered and restricted_unfiltered, reported
        separately -- the whole point of the unfiltered variant is that a
        pre-search filter degrading into a post-search one would only be
        visible there (see docs on the P1-08 gold set's own split). For
        each: the gold-principal side's retrieval recall@8 and answer
        accuracy, and how many non-gold-principal outcomes came back
        correctly empty-and-abstained versus leaked."""
        out: dict[str, dict] = {}
        for subtype in ("restricted_filtered", "restricted_unfiltered"):
            rows = [o for o in self.outcomes if o.subtype == subtype]
            primary = [o for o in rows if o.is_gold_principal]
            secondary = [o for o in rows if not o.is_gold_principal]
            n = len(primary)
            out[subtype] = {
                "n_pairs": n,
                "primary_recall@8": (
                    sum(recall_at_k(o.retrieved_ids, o.gold_ids, 8) for o in primary) / n
                    if n else 0.0
                ),
                "primary_answer_correct": sum(1 for o in primary if o.answer_correct),
                "secondary_n": len(secondary),
                "secondary_correctly_empty_and_abstained": sum(
                    1 for o in secondary if not o.retrieved_ids and o.abstained
                ),
                "leaks": sum(1 for o in secondary if o.leaked_chunk_ids),
            }
        return out

    def no_reader_summary(self) -> dict:
        """po_002: every principal in principals_checked must come back
        with zero retrieved chunks -- there is no gold-principal side to
        this question at all, everyone is the secondary/denied side."""
        rows = [o for o in self.outcomes if o.subtype == "no_reader"]
        return {
            "n_principals_checked": len(rows),
            "all_empty": all(not o.retrieved_ids for o in rows),
            "principals_with_results": [o.principal_id for o in rows if o.retrieved_ids],
        }

    def leaks(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if o.leaked_chunk_ids]

    def leaks_by_subtype(self) -> dict[str, int]:
        """Leak count per subtype -- restricted_filtered and
        restricted_unfiltered reported separately (and no_reader too, if
        it ever fires), not as one aggregate. A pre-search filter that
        degrades into a post-search one would only show up on the
        unfiltered path; folding the two together would hide exactly the
        finding this split exists to surface."""
        out: dict[str, int] = {}
        for o in self.leaks():
            out[o.subtype] = out.get(o.subtype, 0) + 1
        return out

    def passed(self) -> bool:
        """A single leak fails the run. This is not a tunable metric."""
        return not self.leaks()


def check_provenance(
    provenance: dict, settings: Settings, store: LocalVectorStore, docs: SqliteDocStore,
) -> None:
    """Both fingerprints, checked against the LIVE index and corpus, before
    a single question runs. Either one drifting means gold_chunk_ids and
    gold_answer values may no longer correspond to what actually gets
    retrieved -- a recall number computed against that is not a low score,
    it is a meaningless one that looks like a real measurement. Raises
    GoldProvenanceMismatch and refuses to run rather than let that happen
    silently.

    settings_fingerprint alone (the guard P1-05/P1-08 already had) is blind
    to the corpus's own content -- a rename or any other content-only edit
    moves corpus_fingerprint without moving settings_fingerprint at all
    (docs/adr/0007's "a provenance guard only protects against the inputs
    it actually covers"). corpus_fingerprint alone would miss a chunk
    settings change. Both are required, and each is checked against BOTH
    what the gold recorded AND what is live right now: the gold's own
    recorded value could already disagree with current settings/corpus
    even before comparing to the index/docstore artifacts on disk.
    """
    expected_settings_fp = settings_fingerprint(settings.chunk)
    recorded_settings_fp = provenance.get("settings_fingerprint")
    live_settings_fp = store.settings_fingerprint()
    if recorded_settings_fp != expected_settings_fp:
        raise GoldProvenanceMismatch(
            f"gold settings_fingerprint {recorded_settings_fp!r} does not match current "
            f"chunk settings {expected_settings_fp!r} -- chunk settings changed since this "
            "gold set was generated. Regenerate gold (scripts/gen_gold.py) or restore the "
            "settings before evaluating."
        )
    if live_settings_fp is None:
        raise GoldProvenanceMismatch(
            f"live index at {store.path} has no settings_fingerprint -- was it built by "
            "`isc index`? An index built before the settings-fingerprint guard existed "
            "(or never built at all) cannot be trusted to match this gold set."
        )
    if recorded_settings_fp != live_settings_fp:
        raise GoldProvenanceMismatch(
            f"gold settings_fingerprint {recorded_settings_fp!r} does not match the live "
            f"index's {live_settings_fp!r} -- rebuild the index (`isc index`) or regenerate "
            "gold before evaluating."
        )

    expected_corpus_fp = corpus_fingerprint(docs.content_hashes())
    recorded_corpus_fp = provenance.get("corpus_fingerprint")
    if recorded_corpus_fp != expected_corpus_fp:
        raise GoldProvenanceMismatch(
            f"gold corpus_fingerprint {recorded_corpus_fp!r} does not match the live "
            f"corpus's {expected_corpus_fp!r} -- a document changed, was added, or was "
            "removed since this gold set was generated (regardless of which questions "
            "happen to reference it -- see docs/adr/0007/0008). Re-ingest and reindex, or "
            "regenerate gold, before evaluating."
        )


def _principals_for(q: dict) -> list[str]:
    """Every principal id this gold question must be run as. Most classes
    have exactly one (q["principal"]); restricted has two (principal, the
    one it IS answerable for, then principal_b, the one it must not be);
    no_reader has however many principals_checked lists, since its whole
    point is that none of them can read it."""
    if q["subtype"] == "no_reader":
        return list(q["principals_checked"])
    if q["question_class"] == "restricted":
        return [q["principal"], q["principal_b"]]
    return [q["principal"]]


def _abstention_correct(expected_behavior: str, abstained: bool, reason: str | None) -> bool:
    """Scores a single unanswerable-class outcome against its gold
    expected_behavior. Two behaviours, not one:

    "abstain" (absent, out_of_scope) -- correct iff the system actually
    abstained AND did so for a reason that represents genuinely concluding
    "no answer here" (_CORRECT_ABSTAIN_REASONS), not one that means "I
    tried to answer and failed" (UNGROUNDED_DRAFT, ATTRIBUTION_MISMATCH)
    landing on abstained=True by coincidence.

    "abstain_with_clarification" (underspecified) -- always False. Measured
    directly (P1-07's sample and the full 56-question run): the current
    prompt produces neither a plain abstention nor a clarifying question on
    these, it answers directly and hedges across candidates. There is no
    signal anywhere in AbstentionReason for "asked for clarification", so
    there is nothing to check FOR -- scored as a failure honestly rather
    than relaxed to accept a plain INSUFFICIENT_CONTEXT decline as
    "close enough". See docs/WBS-P1.md's P1-09 briefing.
    """
    if expected_behavior == "abstain_with_clarification":
        return False
    return abstained and reason in _CORRECT_ABSTAIN_REASONS


def _expected_to_abstain(o: QuestionOutcome) -> bool:
    """Is abstention the correct behaviour for THIS OUTCOME -- this
    question, run as this specific principal -- independent of whether the
    exact reason was right (that finer question is _abstention_correct()'s
    job, and only applies to question_class == "unanswerable"). Used by
    abstention_precision(), which needs a broader "was abstaining here
    legitimate at all" check across every class:

      unanswerable (any subtype)         -> True, always
      restricted, non-gold-principal side -> True (ben, denied by design)
      restricted, gold-principal side     -> False (alice IS supposed to
                                              get a real answer; abstaining
                                              here is a real failure, e.g.
                                              a retrieval miss, not a
                                              legitimate denial)
      answerable                          -> False

    no_reader falls out of the restricted branch for free: every one of
    its outcomes has is_gold_principal=False (q["principal"] is None for a
    no_reader question, and no principal_id ever equals None -- see
    run()), so all 7 checked principals count as "should abstain" without
    a separate subtype check.
    """
    if o.question_class == "unanswerable":
        return True
    if o.question_class == "restricted":
        return not o.is_gold_principal
    return False


# -- answer accuracy: did the answer contain the gold_answer value? ---------
#
# Reuses eval/normalise.py's normalise_string/date/decimal -- the same
# functions the extraction harness scores against gold with -- rather than
# a second normaliser. What's new here (not in normalise.py, which only
# normalises a single already-isolated value) is CONTAINMENT: scanning free
# answer text for a candidate substring that, once normalised the same way,
# equals the normalised gold value.

_NUMBER_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DATE_TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}")

# gold_answer's nested shapes (cross_document, ambiguous) carry structural
# bookkeeping alongside the actual answer values -- a filename ("document"),
# corpus-design commentary ("note"), a supporting breakdown that restates
# the headline "total" a second time, a currency code the question already
# names, and a line_number used only to locate the row. None of these is
# something a correct answer needs to restate; keeping them out is what
# lets q_cd_01's "summed the wrong single document's total" actually fail
# instead of accidentally matching one breakdown entry.
_SKIP_KEYS = frozenset({"document", "note", "breakdown", "currency", "line_number"})


def _leaf_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            if k in _SKIP_KEYS:
                continue
            out.extend(_leaf_values(v))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_leaf_values(item))
        return out
    if value is None:
        return []
    return [str(value)]


def _value_in_text(value: str, text: str) -> bool:
    """Type-appropriate containment: try decimal, then date, then a plain
    normalised substring -- in that order, since normalise_decimal/
    normalise_date both return None (not a guess) on input of the wrong
    shape, so trying the more specific parsers first is safe."""
    decimal = normalise_decimal(value)
    if decimal is not None:
        candidates = {normalise_decimal(tok) for tok in _NUMBER_TOKEN.findall(text)}
        return decimal in candidates
    date = normalise_date(value)
    if date is not None:
        candidates = {normalise_date(tok) for tok in _DATE_TOKEN.findall(text)}
        return date in candidates
    needle = normalise_string(value)
    haystack = normalise_string(text)
    return needle is not None and haystack is not None and needle in haystack


def answer_contains_gold(gold_answer: Any, answer_text: str) -> bool | None:
    """None means not applicable (gold_answer is null, or -- after
    stripping structural keys -- there was nothing left to check). True
    only if EVERY required value is found: for a multi-entity gold_answer
    (ambiguous's two suppliers, cross_document's per-part prices), a draft
    that names only one of them is not a complete, correct answer -- see
    _leaf_values()'s own docstring on why "document"/"note"/"breakdown"/
    "currency"/"line_number" are excluded from what counts as "required".
    """
    required = _leaf_values(gold_answer)
    if not required:
        return None
    return all(_value_in_text(v, answer_text) for v in required)


@dataclass
class RetrievalEvalResult:
    report: RetrievalReport = field(default_factory=RetrievalReport)
    # (question_id, principal_id, reason) -- same per-item isolation
    # discipline as index/pipeline.py's IndexResult.failed and
    # eval/pipeline.py's ExtractionEvalResult.skipped: one question's
    # provider error must not abort scoring the other fifty-five.
    failed: list[tuple[str, str, str]] = field(default_factory=list)


def run(
    questions: list[dict], users: dict[str, Principal], orchestrator: AnswerOrchestrator,
) -> RetrievalEvalResult:
    """Every gold question, asked as its own gold principal(s) -- never as
    an unrestricted superuser, which would make every recall number look
    better and test nothing about permissions (see this module's own
    docstring and docs/WBS-P1.md's P1-09 "Watch"). check_provenance() must
    already have been called and passed before this runs; it is not called
    from here, so a caller cannot accidentally skip it by only calling run().

    Scores each outcome as it is built, not in a later pass, because the
    leak check needs the live ScoredChunk objects (answer.supporting) --
    their .chunk.acl -- which are gone once reduced to plain retrieved_ids
    strings.
    """
    result = RetrievalEvalResult()
    for q in questions:
        for principal_id in _principals_for(q):
            principal = users[principal_id]
            try:
                with span("eval.retrieval.question", question=q["id"], principal=principal_id):
                    answer = orchestrator.ask(q["text"], principal)
            except Exception as exc:  # noqa: BLE001 - batch isolation boundary, by design
                log.warning("eval failed for %s as %s: %s", q["id"], principal_id, exc)
                result.failed.append((q["id"], principal_id, str(exc)))
                continue

            is_gold_principal = principal_id == q.get("principal")

            # Leak = a chunk THIS principal retrieved that fails THIS
            # principal's own may_read() against its own AclSet. Not "a
            # chunk alice got that ben didn't" -- that comparison is
            # correct behaviour, not a leak. Checked for every outcome,
            # not just principal_b/no_reader ones: the invariant "nobody
            # ever receives a chunk their own ACL denies" has to hold
            # universally, which is also what makes this work unmodified
            # for no_reader and any future principal, not just alice/ben.
            leaked = [
                sc.chunk.id for sc in answer.supporting
                if not principal.may_read(sc.chunk.acl)
            ]

            answer_correct = None
            if is_gold_principal:
                answer_correct = answer_contains_gold(q.get("gold_answer"), answer.text)

            abstention_correct = None
            if q["question_class"] == "unanswerable":
                abstention_correct = _abstention_correct(
                    q["expected_behavior"], answer.abstained,
                    answer.abstention_reason.value if answer.abstention_reason else None,
                )

            result.report.outcomes.append(QuestionOutcome(
                question_id=q["id"],
                question_class=q["question_class"],
                subtype=q["subtype"],
                principal_id=principal_id,
                is_gold_principal=is_gold_principal,
                retrieved_ids=[sc.chunk.id for sc in answer.supporting],
                gold_ids=set(q["gold_chunk_ids"]),
                abstained=answer.abstained,
                abstention_reason=(
                    answer.abstention_reason.value if answer.abstention_reason else None
                ),
                answer_text=answer.text,
                citations=[c.chunk_id for c in answer.citations],
                answer_correct=answer_correct,
                abstention_correct=abstention_correct,
                leaked_chunk_ids=leaked,
            ))
    log.info("ran %d questions (%d outcomes), %d failed",
              len(questions), len(result.report.outcomes), len(result.failed))
    return result

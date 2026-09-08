"""Per-outcome diff between two retrieval runs (EV-04).

report.json can tell you mrr moved by 0.014; it cannot tell you which
outcomes moved. This compares two runs' raw QuestionOutcome records
(outcomes.jsonl + failed.jsonl, loaded via eval/outcomes.py's load()) and
classifies every difference field by field, so "the number changed" becomes
"these three questions retrieved different chunks, and this one flipped
answer_correct."

Deliberately does NOT read report.json: the report is a downstream
aggregate, produced one layer above the records this module compares.
Reading it back would add a dependency on report.py's schema (already
bumped once, by EV-03) for nothing this module needs. It also does NOT
attribute a difference to a metric delta (e.g. "this is why recall@8 moved
by 0.014") -- that would duplicate RetrievalReport's own aggregation logic
(retrieval.py) in a second place that can silently drift from the
original. Both are out of scope by design, not by omission.

Record identity: (question_id, principal_id). This is OBSERVED unique in
every real gold set and the committed synthetic fixture, but never
GUARANTEED anywhere in code -- _principals_for() (retrieval.py:434-444)
does not check principal != principal_b for a restricted question, and the
generator (scripts/gen_gold.py:325) hardcodes principal_b="u_ben" without
comparing it to principal either. So a key colliding within one file
raises (KeyCollision) rather than falling back to positional pairing --
positional pairing would have no principled way to decide which of two
same-keyed records on one side corresponds to which on the other, and a
wrong guess is worse than a refusal.

A question that failed on BOTH sides (present in both runs' `failed` list,
absent from both runs' outcomes) appears in neither run's identity index
and produces no output here at all. That is correct, not a bug: nothing
changed for that question/principal between the two runs. It will read as
a gap to a reader expecting every failed key to show up somewhere in the
result -- it does not, deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from isc.common.errors import IscError
from isc.eval.retrieval import QuestionOutcome, RetrievalEvalResult

Key = tuple[str, str]  # (question_id, principal_id)


class DiffClass(str, Enum):
    # gold-derived fields disagree on a matched key -- the gold set moved,
    # or the pairing is wrong. Not a peer of the classes below it: its
    # presence anywhere makes every other difference in the pair suspect
    # (see render()'s gold-change-only banner).
    GOLD_CHANGE = "gold_change"
    LEAK_CHANGE = "leak_change"
    # One side has NO observation for this key at all (it failed) while the
    # other has a real outcome. Ranked above verdict_flip: a verdict_flip
    # means both runs produced a comparable observation and the metric that
    # moved is measuring something; a status_change means the failing
    # side's answer_accuracy (or whichever metric this outcome feeds) has
    # one fewer denominator entry than the other side's -- the two runs'
    # metrics are not directly comparable for this question at all, which
    # is nearer gold_change's category than verdict_flip's.
    STATUS_CHANGE = "status_change"
    VERDICT_FLIP = "verdict_flip"
    ABSTENTION_CHANGE = "abstention_change"
    RETRIEVAL_SET = "retrieval_set"
    RETRIEVAL_ORDER = "retrieval_order"
    TEXT_CHANGE = "text_change"
    ADDED = "added"
    REMOVED = "removed"


# Rendering/summary order, most severe first. Not the same list as
# _FIELD_PRIORITY below -- GOLD_CHANGE and STATUS_CHANGE are never assigned
# by _diff_record()'s per-field scan (GOLD_CHANGE short-circuits it;
# STATUS_CHANGE only ever comes from the added/removed cross-reference in
# diff()), so they only need a place in the render-time ordering, not in
# the matched-pair priority scan.
SEVERITY_ORDER: tuple[DiffClass, ...] = (
    DiffClass.GOLD_CHANGE,
    DiffClass.LEAK_CHANGE,
    DiffClass.STATUS_CHANGE,
    DiffClass.VERDICT_FLIP,
    DiffClass.ABSTENTION_CHANGE,
    DiffClass.RETRIEVAL_SET,
    DiffClass.RETRIEVAL_ORDER,
    DiffClass.TEXT_CHANGE,
    DiffClass.ADDED,
    DiffClass.REMOVED,
)

# Priority for _diff_record()'s primary_class, when gold-linked fields did
# NOT change (if they did, GOLD_CHANGE wins unconditionally -- see below).
_FIELD_PRIORITY: tuple[DiffClass, ...] = (
    DiffClass.LEAK_CHANGE,
    DiffClass.VERDICT_FLIP,
    DiffClass.ABSTENTION_CHANGE,
    DiffClass.RETRIEVAL_SET,
    DiffClass.RETRIEVAL_ORDER,
    DiffClass.TEXT_CHANGE,
)

_GOLD_LINKED_FIELDS = frozenset({"question_class", "subtype", "is_gold_principal", "gold_ids"})


class KeyCollision(IscError):
    """(question_id, principal_id) appears more than once within a single
    run's outcomes -- see this module's docstring on why identity is
    observed, not guaranteed. Aborts the whole diff: once one key's
    identity is unprovable, nothing else in that file can be safely paired
    either, so a partial result would be false confidence, not a partial
    truth."""

    def __init__(self, label: str, key: Key) -> None:
        super().__init__(
            f"{label}: key {key!r} appears more than once -- cannot establish a unique "
            "identity for diffing. Refusing to guess a pairing for it (or trust any other "
            "key in this file) rather than silently mispair records."
        )
        self.label = label
        self.key = key


class CitationsValidMismatch(IscError):
    """citations_valid differs between the two runs for this key. run()
    never assigns it (retrieval.py:638-655) -- every outcome it produces
    carries the dataclass default True -- so two files both produced by
    run() cannot legitimately disagree on it. A difference means one file
    was hand-edited, or a new code path has started assigning this field
    for real; either way this module's assumption that it is inert no
    longer holds, and that needs a human to look, not a silent
    classification."""

    def __init__(self, key: Key, before: bool, after: bool) -> None:
        super().__init__(
            f"{key!r}: citations_valid differs ({before!r} -> {after!r}), but no code path "
            "in this repo ever assigns it to anything but its default True -- this indicates "
            "a hand-edited file or a new writer, not a routine difference. Refusing to "
            "classify it alongside ordinary differences."
        )
        self.key = key
        self.before = before
        self.after = after


@dataclass(frozen=True)
class FieldDiff:
    field: str
    before: object
    after: object
    diff_class: DiffClass


@dataclass(frozen=True)
class RecordDiff:
    key: Key
    primary_class: DiffClass
    field_diffs: tuple[FieldDiff, ...]


@dataclass(frozen=True)
class StatusChange:
    key: Key
    direction: str  # "failed_to_succeeded" | "succeeded_to_failed"
    outcome: QuestionOutcome  # the succeeding side's outcome
    failure_reason: str  # the failing side's recorded reason


@dataclass(frozen=True)
class AddedRecord:
    key: Key
    outcome: QuestionOutcome


@dataclass(frozen=True)
class RemovedRecord:
    key: Key
    outcome: QuestionOutcome


@dataclass(frozen=True)
class DiffResult:
    changed: tuple[RecordDiff, ...]
    status_changes: tuple[StatusChange, ...]
    added: tuple[AddedRecord, ...]
    removed: tuple[RemovedRecord, ...]

    @property
    def has_gold_change(self) -> bool:
        return any(r.primary_class == DiffClass.GOLD_CHANGE for r in self.changed)

    @property
    def is_identical(self) -> bool:
        return not (self.changed or self.status_changes or self.added or self.removed)


def _classify_list_field(
    before: list[str], after: list[str], set_class: DiffClass, order_class: DiffClass,
) -> DiffClass | None:
    if before == after:
        return None
    if set(before) != set(after):
        return set_class
    return order_class


def _citations_differ(before: list[str], after: list[str]) -> bool:
    """Set comparison, not positional. Order is preserved on disk alongside
    retrieved_ids (outcomes.py:94-96), but -- unlike retrieved_ids, which
    mrr()/ndcg_at_k() iterate positionally (metrics.py:29-33, 36-42) --
    nothing in retrieval.py's scoring reads citations positionally today.
    This is a claim about the CURRENT scorers, not about the field itself:
    if a reranker or a citation-order metric is ever added, this needs to
    change to match retrieved_ids's treatment above it."""
    return set(before) != set(after)


def _diff_record(key: Key, a: QuestionOutcome, b: QuestionOutcome) -> RecordDiff | None:
    if a.citations_valid != b.citations_valid:
        raise CitationsValidMismatch(key, a.citations_valid, b.citations_valid)

    diffs: list[FieldDiff] = []

    if a.question_class != b.question_class:
        diffs.append(FieldDiff("question_class", a.question_class, b.question_class,
                                DiffClass.GOLD_CHANGE))
    if a.subtype != b.subtype:
        diffs.append(FieldDiff("subtype", a.subtype, b.subtype, DiffClass.GOLD_CHANGE))
    if a.is_gold_principal != b.is_gold_principal:
        diffs.append(FieldDiff("is_gold_principal", a.is_gold_principal, b.is_gold_principal,
                                DiffClass.GOLD_CHANGE))
    if a.gold_ids != b.gold_ids:
        diffs.append(FieldDiff("gold_ids", sorted(a.gold_ids), sorted(b.gold_ids),
                                DiffClass.GOLD_CHANGE))

    gold_changed = any(d.field in _GOLD_LINKED_FIELDS for d in diffs)

    if set(a.leaked_chunk_ids) != set(b.leaked_chunk_ids):
        diffs.append(FieldDiff("leaked_chunk_ids", list(a.leaked_chunk_ids),
                                list(b.leaked_chunk_ids), DiffClass.LEAK_CHANGE))

    if a.answer_correct != b.answer_correct:
        diffs.append(FieldDiff("answer_correct", a.answer_correct, b.answer_correct,
                                DiffClass.VERDICT_FLIP))
    if a.abstention_correct != b.abstention_correct:
        diffs.append(FieldDiff("abstention_correct", a.abstention_correct, b.abstention_correct,
                                DiffClass.VERDICT_FLIP))

    if a.abstained != b.abstained:
        diffs.append(FieldDiff("abstained", a.abstained, b.abstained, DiffClass.ABSTENTION_CHANGE))
    if a.abstention_reason != b.abstention_reason:
        diffs.append(FieldDiff("abstention_reason", a.abstention_reason, b.abstention_reason,
                                DiffClass.ABSTENTION_CHANGE))

    retrieved_class = _classify_list_field(
        a.retrieved_ids, b.retrieved_ids, DiffClass.RETRIEVAL_SET, DiffClass.RETRIEVAL_ORDER)
    if retrieved_class is not None:
        diffs.append(FieldDiff("retrieved_ids", list(a.retrieved_ids), list(b.retrieved_ids),
                                retrieved_class))

    if a.answer_text != b.answer_text:
        diffs.append(FieldDiff("answer_text", a.answer_text, b.answer_text, DiffClass.TEXT_CHANGE))
    if _citations_differ(a.citations, b.citations):
        diffs.append(FieldDiff("citations", list(a.citations), list(b.citations),
                                DiffClass.TEXT_CHANGE))

    if not diffs:
        return None

    if gold_changed:
        primary = DiffClass.GOLD_CHANGE
    else:
        primary = min(diffs, key=lambda d: _FIELD_PRIORITY.index(d.diff_class)).diff_class

    return RecordDiff(key=key, primary_class=primary, field_diffs=tuple(diffs))


def _index(result: RetrievalEvalResult, label: str) -> dict[Key, QuestionOutcome]:
    index: dict[Key, QuestionOutcome] = {}
    for o in result.report.outcomes:
        key = (o.question_id, o.principal_id)
        if key in index:
            raise KeyCollision(label, key)
        index[key] = o
    return index


def _failed_keys(result: RetrievalEvalResult) -> dict[Key, str]:
    return {(qid, pid): reason for qid, pid, reason in result.failed}


def diff(
    a: RetrievalEvalResult, b: RetrievalEvalResult, *, label_a: str = "A", label_b: str = "B",
) -> DiffResult:
    """Compare two runs' loaded outcomes (eval/outcomes.py's load() return
    value) and classify every difference. Raises KeyCollision or
    CitationsValidMismatch rather than returning a partial result -- see
    this module's docstring and each exception's own docstring for why."""
    index_a = _index(a, label_a)
    index_b = _index(b, label_b)
    failed_a = _failed_keys(a)
    failed_b = _failed_keys(b)

    changed: list[RecordDiff] = []
    for key in sorted(index_a.keys() & index_b.keys()):
        record_diff = _diff_record(key, index_a[key], index_b[key])
        if record_diff is not None:
            changed.append(record_diff)

    status_changes: list[StatusChange] = []
    added: list[AddedRecord] = []
    removed: list[RemovedRecord] = []

    for key in sorted(index_b.keys() - index_a.keys()):
        if key in failed_a:
            status_changes.append(
                StatusChange(key, "failed_to_succeeded", index_b[key], failed_a[key]))
        else:
            added.append(AddedRecord(key, index_b[key]))

    for key in sorted(index_a.keys() - index_b.keys()):
        if key in failed_b:
            status_changes.append(
                StatusChange(key, "succeeded_to_failed", index_a[key], failed_b[key]))
        else:
            removed.append(RemovedRecord(key, index_a[key]))

    return DiffResult(
        changed=tuple(changed), status_changes=tuple(status_changes),
        added=tuple(added), removed=tuple(removed),
    )


def _severity(cls: DiffClass) -> int:
    return SEVERITY_ORDER.index(cls)


def _render_gold_change_only(result: DiffResult) -> str:
    lines = [
        "GOLD SET DIFFERS BETWEEN THESE TWO RUNS -- comparison invalid.",
        "The record(s) below show gold-derived fields (question_class/subtype/"
        "is_gold_principal/gold_ids) disagreeing for the same (question_id, principal_id) "
        "key. Every other difference between these two runs may be an artifact of comparing "
        "against different gold, not a behaviour change -- not shown.",
        "",
    ]
    gold_records = sorted(
        (r for r in result.changed if r.primary_class == DiffClass.GOLD_CHANGE),
        key=lambda r: r.key,
    )
    for record_diff in gold_records:
        lines.append(f"  {record_diff.key[0]} / {record_diff.key[1]}:")
        for field_diff in record_diff.field_diffs:
            if field_diff.field in _GOLD_LINKED_FIELDS:
                lines.append(f"    {field_diff.field}: {field_diff.before!r} -> {field_diff.after!r}")
    return "\n".join(lines)


def render(result: DiffResult) -> str:
    """Human-readable rendering, the primary (and only) output form -- see
    this item's design note on why no machine-readable format is built.
    Records are grouped by SEVERITY_ORDER, most severe first; within a
    class, sorted by key for a stable, diffable-itself output."""
    if result.has_gold_change:
        return _render_gold_change_only(result)
    if result.is_identical:
        return "No differences."

    items: list[tuple[int, Key, str]] = []

    for record_diff in result.changed:
        header = f"[{record_diff.primary_class.value}] {record_diff.key[0]} / {record_diff.key[1]}"
        detail = "\n".join(
            f"    {fd.field}: {fd.before!r} -> {fd.after!r}" for fd in record_diff.field_diffs)
        items.append((_severity(record_diff.primary_class), record_diff.key, f"{header}\n{detail}"))

    for status_change in result.status_changes:
        header = (f"[status_change:{status_change.direction}] "
                  f"{status_change.key[0]} / {status_change.key[1]}")
        detail = f"    failure_reason: {status_change.failure_reason!r}"
        items.append((_severity(DiffClass.STATUS_CHANGE), status_change.key, f"{header}\n{detail}"))

    for added_record in result.added:
        items.append((_severity(DiffClass.ADDED), added_record.key,
                       f"[added] {added_record.key[0]} / {added_record.key[1]}"))

    for removed_record in result.removed:
        items.append((_severity(DiffClass.REMOVED), removed_record.key,
                       f"[removed] {removed_record.key[0]} / {removed_record.key[1]}"))

    items.sort(key=lambda item: (item[0], item[1]))

    counts: dict[str, int] = {}
    for rank, _key, _text in items:
        cls = SEVERITY_ORDER[rank].value
        counts[cls] = counts.get(cls, 0) + 1
    summary = f"{len(items)} difference(s): " + ", ".join(
        f"{n} {cls}" for cls, n in sorted(counts.items(), key=lambda kv: -counts[kv[0]]))

    return "\n".join([summary, ""] + [text for _rank, _key, text in items])

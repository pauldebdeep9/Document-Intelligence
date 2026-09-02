"""Pure scoring functions for the Document Intelligence PoC eval harness.

Everything here is pure: no file I/O, no network, no OpenAI client, no reading of
goldset.json, no imports from a runner. Every function takes data in and returns numbers or
verdicts out. That is what makes it testable with hand-built inputs, and what makes
re-scoring a saved run free while re-running the pipeline costs money.

Retrieval metrics are corpus-wide and doc_id aware: the harness pools all chunks in a split
into one retrieval candidate set (the only way the near_duplicate questions on po-004/po-005
test anything), so an anchor is satisfied only if its text is a substring of a retrieved
chunk's text *and* that chunk's doc_id matches the gold document. Text matched in a chunk
from a different document is a miss, not a hit — that distinction is the entire point of the
near_duplicate class.
"""

from decimal import Decimal
from enum import StrEnum
from typing import NamedTuple

from evals.gold.schema import Anchor
from isc.models import LineItem, PurchaseOrder, SourceEvidence

_HEADER_FIELDS = (
    "po_number",
    "po_date",
    "supplier_name",
    "ship_to_site",
    "payment_terms",
    "currency",
    "total_amount",
)


# --- retrieval -------------------------------------------------------------------------------


def anchors_satisfied(
    retrieved: list[SourceEvidence],
    gold_doc_id: str,
    anchors: list[Anchor],
) -> list[bool]:
    """One bool per anchor: True iff some retrieved chunk from gold_doc_id contains that
    anchor's exact text as a substring of that chunk's own text. A chunk from a different
    doc_id containing the same text does not satisfy the anchor, and text split across two
    retrieved chunks (present in neither individually) does not either — each chunk's text is
    checked on its own, never concatenated with another.
    """
    return [
        any(source.doc_id == gold_doc_id and anchor.text in source.text for source in retrieved)
        for anchor in anchors
    ]


def hit_at_k(
    retrieved: list[SourceEvidence],
    gold_doc_id: str,
    anchors: list[Anchor],
) -> bool:
    """True iff at least one anchor is satisfied — the standard IR hit@k definition.

    This is deliberately weaker than "every anchor satisfied" (which is what
    evals/gold/schema.py's docstring means by a RetrievalGold item "succeeding", relevant for
    e.g. a cross_page question whose evidence spans two pages): hit_at_k, first_hit_rank, and
    reciprocal_rank are the standard rank-based IR trio, and by definition measure whether
    *any* signal was found, not whether *all* required evidence was found. A future runner
    computing full per-question success should use all(anchors_satisfied(...)) instead.
    """
    return any(anchors_satisfied(retrieved, gold_doc_id, anchors))


def first_hit_rank(
    retrieved: list[SourceEvidence],
    gold_doc_id: str,
    anchors: list[Anchor],
) -> int | None:
    """One-based rank of the first retrieved chunk (in retrieved's given order) that is from
    gold_doc_id and contains at least one anchor's text, or None if no chunk does.
    """
    for rank, source in enumerate(retrieved, start=1):
        if source.doc_id == gold_doc_id and any(anchor.text in source.text for anchor in anchors):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    """1/rank, or 0.0 if rank is None (no hit)."""
    return 0.0 if rank is None else 1.0 / rank


# --- extraction ------------------------------------------------------------------------------


class FieldVerdict(StrEnum):
    """Four-way verdict for one extracted field, not a binary correct/incorrect.

    Null agreement is the point of the extraction prompt discipline — collapsing MISSED and
    HALLUCINATED into a single "incorrect" bucket hides the failure mode that matters most.
    """

    CORRECT = "correct"
    WRONG = "wrong"
    MISSED = "missed"  # expected a non-null value, got null
    HALLUCINATED = "hallucinated"  # expected null, got a non-null value


def score_field(expected: object | None, actual: object | None) -> FieldVerdict:
    """Score one field's expected vs. actual value.

    Both None is CORRECT (agreement that the field is genuinely absent), not MISSED — MISSED
    means expected a value and got none. Decimal values compare by numeric value (Decimal's
    own __eq__ already does this: Decimal("10.00") == Decimal("10") is True). String values
    compare exactly after stripping only surrounding whitespace — no case-folding, no
    punctuation normalization, so an ampersand vs. "and" or a case difference is WRONG, never
    silently treated as equivalent.
    """
    if expected is None and actual is None:
        return FieldVerdict.CORRECT
    if expected is None:
        return FieldVerdict.HALLUCINATED
    if actual is None:
        return FieldVerdict.MISSED

    if isinstance(expected, Decimal) and isinstance(actual, Decimal):
        return FieldVerdict.CORRECT if expected == actual else FieldVerdict.WRONG
    if isinstance(expected, str) and isinstance(actual, str):
        matches = expected.strip() == actual.strip()
        return FieldVerdict.CORRECT if matches else FieldVerdict.WRONG

    return FieldVerdict.CORRECT if expected == actual else FieldVerdict.WRONG


def score_purchase_order(
    expected: PurchaseOrder,
    actual: PurchaseOrder,
) -> dict[str, FieldVerdict]:
    """Score each of the seven header fields (everything but line_items)."""
    return {
        field: score_field(getattr(expected, field), getattr(actual, field))
        for field in _HEADER_FIELDS
    }


class LineItemScore(NamedTuple):
    """Counts from reconciling expected line items against actual ones."""

    matched: int
    missing: int
    spurious: int


def score_line_items(expected: list[LineItem], actual: list[LineItem]) -> LineItemScore:
    """Reconcile expected line items against actual ones by existence, not full field equality.

    Matching rule: walk expected items in order. For an expected item with a non-null
    part_number, search the *remaining unconsumed* actual items for one with the same
    part_number, anywhere in the list (order-independent) — if found, it counts as matched
    and that actual item is consumed. For an expected item with a null part_number, match by
    position instead: it counts as matched iff the actual item at that same index is still
    unconsumed. An expected item that finds nothing either way counts as missing. Whatever
    remains unconsumed in actual after every expected item has been processed counts as
    spurious. Consuming matched actual items prevents one actual item from being counted as a
    match for two different expected items.
    """
    unconsumed = list(range(len(actual)))
    matched = 0
    missing = 0

    for index, expected_item in enumerate(expected):
        found_at: int | None = None
        if expected_item.part_number is not None:
            for candidate in unconsumed:
                if actual[candidate].part_number == expected_item.part_number:
                    found_at = candidate
                    break
        elif index in unconsumed:
            found_at = index

        if found_at is None:
            missing += 1
        else:
            matched += 1
            unconsumed.remove(found_at)

    return LineItemScore(matched=matched, missing=missing, spurious=len(unconsumed))


# --- insufficiency compliance -----------------------------------------------------------------

# Deliberately pinned here, not imported from isc.llm. This string is the contract between
# what the model is asked to emit and what this eval scores as compliant. If it were
# imported, an edit to isc.llm's wording would silently change what this eval accepts — the
# metric would keep returning True and every prior run record would stay scored as
# compliant, with nothing anywhere going red. Pinning it here means a real wording change
# shows up as a failing test (see test_pinned_insufficiency_answer_matches_isc_llm_contract
# in tests/evals/test_metrics.py — the only place these two values are allowed to meet), not
# a silent pass-through.
INSUFFICIENCY_ANSWER = "I don't have enough information in the provided sources."


def is_insufficiency_response(answer: str) -> bool:
    """Exact-string compliance with the pinned INSUFFICIENCY_ANSWER contract above.

    No normalization, no fuzzy match, no startswith, no strip — this exact string is a
    control signal downstream, so a near-miss (trailing whitespace, a differing final
    period, a case difference) is a real failure and must be visible as one. This is binary
    string compliance, not answer grading.
    """
    return answer == INSUFFICIENCY_ANSWER


# --- reporting ---------------------------------------------------------------------------------


def format_kn(k: int, n: int) -> str:
    """Format a count as "k/n" — deliberately not a percentage.

    At n around 24 (this corpus's DEV split size), a percentage implies a precision the
    sample can't support. No percentage/rate formatter exists anywhere in this module; see
    test_metrics_module_has_no_percentage_or_rate_formatter, which enforces that structurally.
    """
    return f"{k}/{n}"

"""Explicit, hand-assigned DEV/TEST/HELD document split, with access guards on TEST/HELD.

Hash-based bucketing was tried first and rejected: with only 12 hand-constructed documents
covering 5 question classes, a handful of which (cross_page, near_duplicate) exist on exactly
one or two documents by construction, a hash can easily strand a class out of DEV entirely —
which is exactly what happened (near_duplicate landed 100% in TEST). Coverage per split is a
requirement here, not an emergent property worth leaving to a hash function.
"""

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from evals.gold.authoring import GOLDSET_PATH
from evals.gold.schema import GoldSet


class Split(StrEnum):
    """Which document subset a gold item belongs to."""

    DEV = "dev"
    TEST = "test"
    HELD = "held"


# Every doc_id must appear exactly once. cross_page (po-006 only) and near_duplicate
# (po-004/po-005 only) exist on so few documents that they can only ever appear in whichever
# split holds those specific doc_ids — see tests/evals/test_gold.py's explicit assertion that
# this is a known corpus limitation, not a design choice, and should be revisited once a
# second cross_page or near_duplicate document exists.
_SPLIT_ASSIGNMENTS: dict[str, Split] = {
    # DEV: the split tuned against during development, so it must carry every class,
    # including the ones that only exist on one or two documents.
    "po-001": Split.DEV,  # baseline case; every other split's results are compared to this
    "po-002": Split.DEV,  # true-null payment_terms/currency; DEV's absent-class coverage
    "po-004": Split.DEV,  # near_duplicate pair (part number ...56) — must stay with po-005
    "po-005": Split.DEV,  # near_duplicate pair (part number ...57) — must stay with po-004
    "po-006": Split.DEV,  # only cross_page document in the corpus
    "po-010": Split.DEV,  # chunk-boundary-crossing description; DEV's line_item edge case
    # TEST: header_field/line_item/absent coverage away from DEV.
    "po-003": Split.TEST,  # ambiguous po_date preserved verbatim
    "po-007": Split.TEST,  # blank page 2; page-numbering edge case
    "po-008": Split.TEST,  # printed total inconsistent with line items
    "po-012": Split.TEST,  # zero line items; TEST's absent-class coverage
    # HELD: header_field/line_item/absent coverage, held out from routine runs.
    "po-009": Split.HELD,  # trimmed supplier name with an ampersand
    "po-011": Split.HELD,  # bare currency symbol; HELD's absent-class coverage
}


def assign_split(doc_id: str) -> Split:
    """Look up doc_id's split in the explicit, committed assignment above.

    Raises for an unrecognized doc_id rather than silently defaulting it into a split —
    a new document belongs in _SPLIT_ASSIGNMENTS by deliberate choice, not by falling
    through.
    """
    try:
        return _SPLIT_ASSIGNMENTS[doc_id]
    except KeyError:
        raise ValueError(f"{doc_id} has no split assignment in _SPLIT_ASSIGNMENTS") from None


def assert_splits_are_disjoint(doc_ids: Sequence[str]) -> None:
    """Raise if any doc_id would resolve to more than one split.

    assign_split is a pure function of doc_id alone, so a single doc_id can never itself
    collide across splits; this guards the case of encountering the same doc_id twice with
    differing recorded splits (e.g. a caller that passes doc_id/split pairs from two
    inconsistent sources), and stays meaningful if assign_split's inputs ever grow.
    """
    seen: dict[str, Split] = {}
    for doc_id in doc_ids:
        split = assign_split(doc_id)
        if doc_id in seen and seen[doc_id] != split:
            raise ValueError(
                f"{doc_id} resolved to both {seen[doc_id]} and {split}"
            )
        seen[doc_id] = split


def load_gold(
    split: Split,
    *,
    allow_test: bool = False,
    allow_held: bool = False,
    run_label: str = "",
    goldset_path: Path = GOLDSET_PATH,
) -> GoldSet:
    """Load the gold set, filtered to one split, behind explicit access guards.

    DEV loads freely. TEST requires allow_test=True. HELD requires both allow_held=True and
    a non-empty run_label, so that a held-out run is always attributable.
    """
    if split == Split.TEST and not allow_test:
        raise ValueError("Loading the TEST split requires allow_test=True")
    if split == Split.HELD and not (allow_held and run_label):
        raise ValueError(
            "Loading the HELD split requires allow_held=True and a non-empty run_label"
        )

    goldset = GoldSet.model_validate_json(goldset_path.read_text(encoding="utf-8"))
    items = [item for item in goldset.items if assign_split(item.doc_id) == split]
    return goldset.model_copy(update={"items": items})

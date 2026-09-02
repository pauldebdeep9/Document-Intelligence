"""Deterministic DEV/TEST/HELD document split, with access guards on TEST and HELD."""

import hashlib
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from evals.gold.authoring import GOLDSET_PATH
from evals.gold.schema import GoldSet

_HASH_SPACE = 2**256
_TEST_CUTOFF = 0.5
_HELD_CUTOFF = 0.8


class Split(StrEnum):
    """Which document subset a gold item belongs to."""

    DEV = "dev"
    TEST = "test"
    HELD = "held"


def assign_split(doc_id: str) -> Split:
    """Deterministically bucket a doc_id via sha256, independent of any other doc_id.

    Normalizes sha256(doc_id) to [0, 1) and splits it 50/30/20: DEV below 0.5, TEST below
    0.8, HELD above. For the current 12-document corpus this lands on exactly 6/3/3.
    """
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    fraction = int(digest, 16) / _HASH_SPACE
    if fraction < _TEST_CUTOFF:
        return Split.DEV
    if fraction < _HELD_CUTOFF:
        return Split.TEST
    return Split.HELD


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

"""Supplier and part master-data resolution for Signal.MASTER_DATA.

Not a plain ID lookup. Two things the corpus deliberately makes hard:

  * 9/20 documents omit supplier_id, so an ID-only lookup misses on nearly
    half the corpus for correctly-extracted fields -- resolution falls back to
    exact name, then normalised name (casefold, collapsed whitespace, common
    legal suffixes stripped), never further than that.
  * The masters contain confusable pairs on purpose ("Fastenal Industrial
    Supply Pte Ltd" vs "...Services Pte Ltd"; two Bosch Rexroth entities).
    Resolution is exact-match-after-normalisation only, never fuzzy/edit-
    distance. An ambiguous near-match resolves to a miss with a conflict, not
    to whichever candidate scores highest -- picking a plausible-but-wrong
    supplier is the failure mode this refuses to create.

Parts: data/masters/unmastered_parts.json lists parts that appear on documents
but are absent from the master by design. A miss there is correct behaviour,
not an extraction error, and resolve_part() returns Confidence.unknown() for
it -- neither corroborating nor conflicting -- rather than flagging a
conflict.

Loaded once per process (lru_cache): the master lists are small, static within
a run, and extract() may be called once per document in a batch, so reloading
JSON from disk on every call would be wasted work for no benefit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from isc.common.confidence import Confidence, Signal

_LEGAL_SUFFIXES = (" pte ltd", " gmbh", " llc", " ag", " inc", " co", " corp")


def _normalise_name(name: str) -> str:
    s = " ".join(name.split()).casefold()
    for suffix in _LEGAL_SUFFIXES:
        if s.endswith(suffix):
            return s[: -len(suffix)].strip()
    return s.strip()


@lru_cache(maxsize=1)
def _suppliers(masters_dir: Path) -> tuple[dict[str, str], ...]:
    return tuple(json.loads((masters_dir / "suppliers.json").read_text()))


@lru_cache(maxsize=1)
def _parts(masters_dir: Path) -> frozenset[str]:
    raw = json.loads((masters_dir / "parts.json").read_text())
    return frozenset(p["part_number"] for p in raw)


@lru_cache(maxsize=1)
def _unmastered_parts(masters_dir: Path) -> frozenset[str]:
    raw = json.loads((masters_dir / "unmastered_parts.json").read_text())
    return frozenset(p["part_number"] for p in raw)


def resolve_supplier(
    supplier_id: str | None, supplier_name: str | None, masters_dir: Path
) -> Confidence:
    """ID first, then exact name, then normalised name. The same result is
    used to corroborate/conflict both the supplier_id and supplier_name
    fields -- resolving one is evidence about the supplier the record claims,
    which is what both fields are ultimately about."""
    if supplier_id is None and supplier_name is None:
        return Confidence.unknown()

    suppliers = _suppliers(masters_dir)

    if supplier_id is not None:
        hits = [s for s in suppliers if s["supplier_id"] == supplier_id]
        if len(hits) == 1:
            return Confidence.of(Signal.MASTER_DATA, 0.97, "resolved by id")
        if len(hits) > 1:
            return Confidence.of(Signal.MASTER_DATA, 0.15, "id ambiguous in master")

    if supplier_name is not None:
        exact = [s for s in suppliers if s["name"] == supplier_name]
        if len(exact) == 1:
            return Confidence.of(Signal.MASTER_DATA, 0.97, "resolved by exact name")
        if len(exact) > 1:
            return Confidence.of(Signal.MASTER_DATA, 0.15, "name ambiguous in master")

        normalised = _normalise_name(supplier_name)
        norm_hits = [s for s in suppliers if _normalise_name(s["name"]) == normalised]
        if len(norm_hits) == 1:
            return Confidence.of(Signal.MASTER_DATA, 0.9, "resolved by normalised name")
        if len(norm_hits) > 1:
            return Confidence.of(Signal.MASTER_DATA, 0.15, "normalised name ambiguous in master")

    label = "supplier_id" if supplier_id is not None else "supplier"
    return Confidence.of(Signal.MASTER_DATA, 0.15, f"{label} not found in master")


def resolve_part(part_number: str | None, masters_dir: Path) -> Confidence:
    if part_number is None:
        return Confidence.unknown()
    if part_number in _parts(masters_dir):
        return Confidence.of(Signal.MASTER_DATA, 0.97, "resolved in parts master")
    if part_number in _unmastered_parts(masters_dir):
        return Confidence.unknown()  # deliberately unmastered -- not a conflict
    return Confidence.of(Signal.MASTER_DATA, 0.15, "part not found in master or exception list")

"""Supplier, part, and site master-data resolution.

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
not an extraction error, and resolve_part() returns Check.not_applicable() for
it -- neither corroborating nor conflicting -- rather than flagging a
conflict.

check_description() cross-checks the extracted line description against the
resolved part's own canonical description -- resolve_part() alone only ever
looks at the part_number field, so a description that is textually short of
the truth (a wrapped table cell whose continuation never reached extract/;
see docs/LIMITATIONS.md) was invisible to MASTER_DATA entirely until this
existed. It cannot fix the truncation -- that is parse/'s job -- only make it
visible instead of silently auto-acceptable.

Sites: data/masters/sites.json maps a ship_to_site name to the shipping
site's own date convention. resolve_site_date_format() is exact-name lookup
only, same discipline as resolve_supplier() -- a near-miss site name must not
silently borrow another site's date convention. Feeds
extract/validators.py's parse_iso_date(), which is where the actual
disambiguation happens; this module only resolves *which* convention applies.

Loaded once per process (lru_cache): the master lists are small, static within
a run, and extract() may be called once per document in a batch, so reloading
JSON from disk on every call would be wasted work for no benefit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from isc.common.confidence import Check, CheckOutcome, Confidence, Signal

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
def _parts(masters_dir: Path) -> dict[str, str]:
    """part_number -> canonical description."""
    raw = json.loads((masters_dir / "parts.json").read_text())
    return {p["part_number"]: p["description"] for p in raw}


@lru_cache(maxsize=1)
def _unmastered_parts(masters_dir: Path) -> frozenset[str]:
    raw = json.loads((masters_dir / "unmastered_parts.json").read_text())
    return frozenset(p["part_number"] for p in raw)


@lru_cache(maxsize=1)
def _sites(masters_dir: Path) -> tuple[dict[str, str], ...]:
    return tuple(json.loads((masters_dir / "sites.json").read_text()))


def resolve_supplier(
    supplier_id: str | None, supplier_name: str | None, masters_dir: Path
) -> Check:
    """ID first, then exact name, then normalised name. The same result is
    used to corroborate/conflict both the supplier_id and supplier_name
    fields -- resolving one is evidence about the supplier the record claims,
    which is what both fields are ultimately about."""
    if supplier_id is None and supplier_name is None:
        return Check.not_applicable()

    suppliers = _suppliers(masters_dir)

    if supplier_id is not None:
        hits = [s for s in suppliers if s["supplier_id"] == supplier_id]
        if len(hits) == 1:
            return Check(CheckOutcome.PASS, Confidence.of(
                Signal.MASTER_DATA, 0.97, "resolved by id",
            ))
        if len(hits) > 1:
            return Check(CheckOutcome.FAIL, Confidence.of(
                Signal.MASTER_DATA, 0.15, "id ambiguous in master",
            ))

    if supplier_name is not None:
        exact = [s for s in suppliers if s["name"] == supplier_name]
        if len(exact) == 1:
            return Check(CheckOutcome.PASS, Confidence.of(
                Signal.MASTER_DATA, 0.97, "resolved by exact name",
            ))
        if len(exact) > 1:
            return Check(CheckOutcome.FAIL, Confidence.of(
                Signal.MASTER_DATA, 0.15, "name ambiguous in master",
            ))

        normalised = _normalise_name(supplier_name)
        norm_hits = [s for s in suppliers if _normalise_name(s["name"]) == normalised]
        if len(norm_hits) == 1:
            return Check(CheckOutcome.PASS, Confidence.of(
                Signal.MASTER_DATA, 0.9, "resolved by normalised name",
            ))
        if len(norm_hits) > 1:
            return Check(CheckOutcome.FAIL, Confidence.of(
                Signal.MASTER_DATA, 0.15, "normalised name ambiguous in master",
            ))

    label = "supplier_id" if supplier_id is not None else "supplier"
    return Check(CheckOutcome.FAIL, Confidence.of(
        Signal.MASTER_DATA, 0.15, f"{label} not found in master",
    ))


def resolve_part(part_number: str | None, masters_dir: Path) -> Check:
    if part_number is None:
        return Check.not_applicable()
    if part_number in _parts(masters_dir):
        return Check(CheckOutcome.PASS, Confidence.of(
            Signal.MASTER_DATA, 0.97, "resolved in parts master",
        ))
    if part_number in _unmastered_parts(masters_dir):
        return Check.not_applicable()  # deliberately unmastered -- not a conflict
    return Check(CheckOutcome.FAIL, Confidence.of(
        Signal.MASTER_DATA, 0.15, "part not found in master or exception list",
    ))


def check_description(
    part_number: str | None, description: str | None, masters_dir: Path
) -> Check:
    """Cross-check an extracted line description against the resolved
    part's canonical master description. Case/whitespace-insensitive, never
    fuzzy -- same discipline as resolve_supplier().

    Only runs when the part itself resolves in the master: with no
    part_number, no description, or a part that is unmastered or genuinely
    unknown, there is no canonical text to compare against, and that is not
    evidence the description is wrong -- Check.not_applicable(), same as
    resolve_part()'s own unmastered case.

    Three outcomes once a master description exists:
      exact match (casefold + strip)   -> PASS, corroborates.
      extracted is a strict substring
        of the master description      -> UNCERTAIN, discounts. Truncation is
        the expected failure mode here: the model did not invent anything, it
        received less than the full printed value (a parse-stage table cell
        that wraps onto a second physical line and never gets stitched back
        together -- see docs/LIMITATIONS.md). Doubt about completeness, not a
        disagreement, so this must not be logged as a conflict.
      anything else                    -> FAIL, conflicts. Neither the
        master's text nor a prefix of it is a real disagreement, not a gap.
    """
    if part_number is None or description is None:
        return Check.not_applicable()
    master_description = _parts(masters_dir).get(part_number)
    if master_description is None:
        return Check.not_applicable()

    extracted = description.strip().casefold()
    master = master_description.strip().casefold()
    if extracted == master:
        return Check(CheckOutcome.PASS, Confidence.of(
            Signal.MASTER_DATA, 0.97, "description matches master",
        ))
    if extracted and extracted in master:
        return Check(CheckOutcome.UNCERTAIN, Confidence.of(
            Signal.MASTER_DATA, 0.75,
            f"description is a partial match of master {master_description!r}",
        ))
    return Check(CheckOutcome.FAIL, Confidence.of(
        Signal.MASTER_DATA, 0.15, f"description does not match master {master_description!r}",
    ))


def resolve_site_date_format(ship_to_site: str | None, masters_dir: Path) -> str | None:
    """Exact-name lookup only -- same discipline as resolve_supplier(): a
    near-miss site name must not silently borrow another site's date
    convention. None means unresolved (ship_to_site absent, unknown, or
    ambiguous); callers fall back to a fixed-priority guess, never to a
    default convention."""
    if ship_to_site is None:
        return None
    hits = [s for s in _sites(masters_dir) if s["name"] == ship_to_site]
    return hits[0]["date_format"] if len(hits) == 1 else None

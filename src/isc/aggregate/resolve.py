"""Supplier named in a QUESTION -> every master supplier it could mean.

Deliberately a different policy from extract/masters.py. A name printed on a
document is a full legal name and wants exact-or-nothing
(supplier_id_for_printed_name()). A name typed in a question is often a
prefix -- "Kestrel Industrial", "Omron" -- and wants every candidate, so the
caller can show each one rather than silently choosing. Reusing the
extraction-side resolver here would resolve "Kestrel Industrial" to Kestrel
Industrial AG alone, because both normalise to "kestrel industrial": the
confidently-narrow answer retrieve/retriever.py's infer_filters() docstring
already refuses to produce.
"""

from __future__ import annotations

from pathlib import Path

from isc.extract.masters import normalise_supplier_name, supplier_ids_by_name


def _collapse(name: str) -> str:
    return " ".join(name.split()).casefold()


def resolve_mention(mention: str | None, masters_dir: Path) -> tuple[str, ...]:
    """Return every candidate supplier_id, in master order; () if none.

    1. The mention IS a full master name (case/whitespace-insensitive,
       legal suffix included): that entity only. "Kestrel Industrial AG"
       means AG, not the Pneumatics GmbH.
    2. Otherwise token-prefix match on normalised names: the mention's
       tokens must be the leading tokens of the master name. "Kestrel
       Industrial" -> both Kestrel entities; "Keyence" -> Keyence Singapore.
       Whole tokens only, never substrings or edit distance: "Kes" matches
       nothing.
    """
    if not mention or not mention.strip():
        return ()
    names = supplier_ids_by_name(masters_dir)

    wanted = _collapse(mention)
    exact = tuple(sid for name, sid in names.items() if _collapse(name) == wanted)
    if exact:
        return exact

    tokens = normalise_supplier_name(mention).split()
    if not tokens:
        return ()
    return tuple(
        sid for name, sid in names.items()
        if normalise_supplier_name(name).split()[: len(tokens)] == tokens
    )


def supplier_names_by_id(masters_dir: Path) -> dict[str, str]:
    return {sid: name for name, sid in supplier_ids_by_name(masters_dir).items()}

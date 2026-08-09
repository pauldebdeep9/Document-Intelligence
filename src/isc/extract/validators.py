"""Lexical and master-data checks. Each returns a Confidence factor to corroborate.

Cheap, deterministic, and independent of the model — which is what makes them
worth combining with noisy-OR rather than replacing the model output.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from isc.common.confidence import Confidence, Signal

PO_NUMBER = re.compile(r"^4[05]\d{8}$")          # SAP-style PO range
PART_NUMBER = re.compile(r"^[A-Z0-9]{2,}-[A-Z0-9\-]{2,}$")
CURRENCY = re.compile(r"^[A-Z]{3}$")
INCOTERM = {"EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"}


def check_pattern(value: str | None, pattern: re.Pattern[str], label: str) -> Confidence:
    if value is None:
        return Confidence.unknown()
    ok = bool(pattern.match(value))
    return Confidence.of(Signal.LEXICAL, 0.9 if ok else 0.1, f"{label}:{'ok' if ok else 'fail'}")


def check_enum(value: str | None, allowed: set[str], label: str) -> Confidence:
    if value is None:
        return Confidence.unknown()
    ok = value.upper() in allowed
    return Confidence.of(Signal.LEXICAL, 0.95 if ok else 0.05, f"{label}:{'ok' if ok else 'fail'}")


def parse_iso_date(value: str | None) -> tuple[date | None, Confidence]:
    if not value:
        return None, Confidence.unknown()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            d = datetime.strptime(value, fmt).date()
        except ValueError:
            continue
        # Ambiguous formats are a real error source in a global supply chain:
        # 03/04/2025 is two different dates depending on the shipping origin.
        ambiguous = fmt in {"%d/%m/%Y", "%m/%d/%Y"} and d.day <= 12
        return d, Confidence.of(
            Signal.LEXICAL, 0.6 if ambiguous else 0.95,
            f"date fmt {fmt}" + (" AMBIGUOUS" if ambiguous else ""),
        )
    return None, Confidence.of(Signal.LEXICAL, 0.05, f"unparseable date {value!r}")


def check_master(value: str | None, master: set[str], label: str) -> Confidence:
    """Resolution against supplier/part master is the strongest cheap signal."""
    if value is None:
        return Confidence.unknown()
    return Confidence.of(
        Signal.MASTER_DATA, 0.97 if value in master else 0.15,
        f"{label}:{'hit' if value in master else 'miss'}",
    )

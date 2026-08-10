"""Lexical and master-data checks. Each returns a Check: an explicit
CheckOutcome plus the Confidence evidence for it -- not a bare score for
_apply_check to reinterpret by threshold. See common.confidence.CheckOutcome
for why that distinction has to be explicit: a validator that can only say
"how sure" and not "sure of what" cannot tell a caller it is uncertain
rather than agreeing.

Cheap, deterministic, and independent of the model — which is what makes them
worth combining with noisy-OR rather than replacing the model output.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from isc.common.confidence import Check, CheckOutcome, Confidence, Signal

PO_NUMBER = re.compile(r"^4[05]\d{8}$")          # SAP-style PO range
PART_NUMBER = re.compile(r"^[A-Z0-9]{2,}-[A-Z0-9\-]{2,}$")
CURRENCY = re.compile(r"^[A-Z]{3}$")
INCOTERM = {"EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"}

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%b-%Y")
_SLASH_FORMATS = frozenset({"%d/%m/%Y", "%m/%d/%Y"})


def check_pattern(value: str | None, pattern: re.Pattern[str], label: str) -> Check:
    if value is None:
        return Check.not_applicable()
    ok = bool(pattern.match(value))
    return Check(
        CheckOutcome.PASS if ok else CheckOutcome.FAIL,
        Confidence.of(Signal.LEXICAL, 0.9 if ok else 0.1, f"{label}:{'ok' if ok else 'fail'}"),
    )


def check_enum(value: str | None, allowed: set[str], label: str) -> Check:
    if value is None:
        return Check.not_applicable()
    ok = value.upper() in allowed
    return Check(
        CheckOutcome.PASS if ok else CheckOutcome.FAIL,
        Confidence.of(Signal.LEXICAL, 0.95 if ok else 0.05, f"{label}:{'ok' if ok else 'fail'}"),
    )


def parse_iso_date(
    value: str | None, site_date_format: str | None = None
) -> tuple[date | None, Check]:
    """`site_date_format` is the shipping site's own date convention (see
    extract/masters.py's resolve_site_date_format()), when ship_to_site
    resolved to a known site -- else None. Tried first: 03/04/2025 is
    genuinely two different dates depending on origin, and no priority
    ordering over guessed formats can recover information that was never in
    the string -- only site context can. Falls back to the fixed priority
    order below when the site is unknown, or when the value doesn't parse
    under the site's own format at all (wrong field, OCR noise, etc.).

    A value that WOULD have been ambiguous without site context stays
    UNCERTAIN even once the site resolves it -- at a higher score than a
    blind guess, but below a value that was never ambiguous at all, because
    ship_to_site is itself an extracted field and could be wrong. See
    _date_check().
    """
    if not value:
        return None, Check.not_applicable()

    if site_date_format is not None:
        try:
            d = datetime.strptime(value, site_date_format).date()
        except ValueError:
            pass
        else:
            return d, _date_check(site_date_format, d, site_resolved=True)

    # %d.%m.%Y: the DE site's format (site_de07 in data/masters/sites.json).
    # Not ambiguous the way the two slash formats are -- there is no
    # competing %m.%d.%Y in this corpus, so the separator alone disambiguates.
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(value, fmt).date()
        except ValueError:
            continue
        return d, _date_check(fmt, d, site_resolved=False)
    return None, Check(
        CheckOutcome.FAIL, Confidence.of(Signal.LEXICAL, 0.05, f"unparseable date {value!r}"),
    )


def _date_check(fmt: str, d: date, *, site_resolved: bool) -> Check:
    """Ambiguous formats are a real error source in a global supply chain:
    03/04/2025 is two different dates depending on the shipping origin. The
    condition is symmetric in which slash format actually matched: whichever
    one succeeded already had its own %m (or %d) component in range by
    construction (strptime would have rejected it otherwise), so d.day <= 12
    alone is enough to tell whether the *other* reading would also have been
    valid.
    """
    if fmt not in _SLASH_FORMATS or d.day > 12:
        detail = f"date fmt {fmt}" + (" (site-resolved)" if site_resolved else "")
        return Check(CheckOutcome.PASS, Confidence.of(Signal.LEXICAL, 0.95, detail))
    if site_resolved:
        return Check(CheckOutcome.UNCERTAIN, Confidence.of(
            Signal.LEXICAL, 0.85, f"date fmt {fmt} (site-resolved, was ambiguous)",
        ))
    return Check(CheckOutcome.UNCERTAIN, Confidence.of(
        Signal.LEXICAL, 0.6, f"date fmt {fmt} AMBIGUOUS",
    ))

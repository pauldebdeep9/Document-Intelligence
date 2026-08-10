"""Type-aware normalisation for gold comparison.

Two independently normalised value spaces get compared against gold, on
purpose -- collapsing them into one is how a normalisation bug gets
misattributed to extraction, or the other way round:

  extraction axis      gold["raw"] (exactly as printed) vs the model's raw
                        structured output. Neither side has been through our
                        own normalisation code. Tests: did the model read the
                        right characters off the page?
  normalisation axis    gold["normalised"] (typed, ISO dates, 2dp decimals)
                        vs the wrapped record's own value. Tests: did OUR
                        code (parse_iso_date, Decimal conversion, stripping)
                        turn a correctly-read value into the right typed one?

A date read correctly but normalised wrongly must score correct on the
extraction axis and wrong (or missed) on the normalisation axis -- not the
same outcome smeared across both.

The normalise_* functions are deliberately format-tolerant on input (a date
arrives as a locale string, an ISO string, or a `date` object depending on
which side and which axis is calling) but always produce the same comparable
output shape, so the same function serves both axes.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

# Every date format this codebase produces or accepts anywhere: gold's raw
# site-local formats (validators.py), plus ISO for gold["normalised"] and any
# already-ISO input.
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%b-%Y")


def normalise_string(value: object) -> str | None:
    """casefold + strip. None and "" are the same "nothing here" for
    comparison purposes -- an empty string is not a value gold would ever
    assert, so treating it as absent rather than as a mismatch is correct,
    not lenient."""
    if value is None:
        return None
    s = str(value).strip().casefold()
    return s or None


def normalise_date(value: object) -> str | None:
    """Comparable ISO date string, regardless of which shape it arrives in:
    a `date` object (the wrapped record's own value), an ISO string
    (gold["normalised"]), or a locale-formatted string (gold["raw"] or the
    model's raw output, in whichever of the corpus's three site formats).
    None on anything unparseable -- the caller scores that as a miss, not a
    silent pass."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalise_decimal(value: object) -> Decimal | None:
    """2dp Decimal from a float, a comma-grouped string ("1,536.37"), a plain
    numeric string, or a Decimal already. None on anything unparseable or
    empty, same reasoning as normalise_date."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        try:
            return value.quantize(Decimal("0.01"))
        except InvalidOperation:
            return None
    s = str(value).replace(",", "").strip()
    if not s:
        return None
    try:
        return Decimal(s).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None

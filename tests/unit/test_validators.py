"""Unit tests for extract/validators.py: CheckOutcome dispatch and the
site-aware date parsing added to fix the corroborate/discount inversion --
see extractor.py's _apply_check() and validators.py's parse_iso_date().
"""

from __future__ import annotations

from datetime import date

from isc.common.confidence import CheckOutcome, Signal
from isc.extract import validators


def test_check_pattern_pass():
    check = validators.check_pattern("4500123456", validators.PO_NUMBER, "po_number")
    assert check.outcome is CheckOutcome.PASS
    assert check.confidence.score == 0.9


def test_check_pattern_fail():
    check = validators.check_pattern("PO-BAD", validators.PO_NUMBER, "po_number")
    assert check.outcome is CheckOutcome.FAIL
    assert check.confidence.score == 0.1


def test_check_pattern_none_is_not_applicable():
    check = validators.check_pattern(None, validators.PO_NUMBER, "po_number")
    assert check.confidence.factors == ()


def test_check_enum_pass_and_fail():
    pass_check = validators.check_enum("FOB", validators.INCOTERM, "incoterms")
    fail_check = validators.check_enum("XYZ", validators.INCOTERM, "incoterms")
    assert pass_check.outcome is CheckOutcome.PASS
    assert fail_check.outcome is CheckOutcome.FAIL


# --- parse_iso_date: no site context (fixed priority order) ----------------

def test_unambiguous_date_passes():
    d, check = validators.parse_iso_date("25/12/2025")  # day=25, can only be %d/%m/%Y
    assert d == date(2025, 12, 25)
    assert check.outcome is CheckOutcome.PASS
    assert check.confidence.score == 0.95


def test_ambiguous_date_without_site_context_is_uncertain_not_pass():
    """The regression this guards: a value the parser itself flags as a
    guess must not be treated the same as a confirmed read. Before the fix,
    _apply_check read this 0.6 score as a pass and corroborated confidence
    upward -- doubt read as support."""
    d, check = validators.parse_iso_date("03/04/2025")  # both readings valid
    assert d == date(2025, 4, 3)  # %d/%m/%Y tried first in the fallback order
    assert check.outcome is CheckOutcome.UNCERTAIN
    assert check.confidence.score == 0.6
    assert "AMBIGUOUS" in check.confidence.factors[0].detail


def test_unparseable_date_fails():
    d, check = validators.parse_iso_date("not-a-date")
    assert d is None
    assert check.outcome is CheckOutcome.FAIL


def test_none_or_empty_date_is_not_applicable():
    for value in (None, ""):
        d, check = validators.parse_iso_date(value)
        assert d is None
        assert check.confidence.factors == ()


# --- parse_iso_date: with site context --------------------------------

def test_site_format_resolves_ambiguous_date_correctly():
    """03/04/2025 is genuinely two different dates. Without site context the
    fixed order guesses %d/%m/%Y (2025-04-03). A US site's %m/%d/%Y
    convention means the correct read is 2025-03-04 -- the opposite date."""
    d, check = validators.parse_iso_date("03/04/2025", site_date_format="%m/%d/%Y")
    assert d == date(2025, 3, 4)
    assert check.outcome is CheckOutcome.UNCERTAIN  # still not a full pass -- see next test
    assert check.confidence.score == 0.85
    assert check.confidence.factors[0].signal == Signal.LEXICAL
    assert "site-resolved" in check.confidence.factors[0].detail


def test_site_resolved_ambiguous_date_scores_higher_than_blind_guess_but_below_pass():
    """Keep the ambiguity signal even when the site resolves it: ship_to_site
    is itself extracted and could be wrong, so this is more certain than a
    blind guess (0.6) but never as certain as a date that was never
    ambiguous at all (0.95)."""
    _, blind = validators.parse_iso_date("03/04/2025")
    _, site_resolved = validators.parse_iso_date("03/04/2025", site_date_format="%m/%d/%Y")
    _, unambiguous = validators.parse_iso_date("25/12/2025", site_date_format="%m/%d/%Y")
    assert blind.confidence.score < site_resolved.confidence.score < unambiguous.confidence.score


def test_site_format_unambiguous_date_still_passes():
    d, check = validators.parse_iso_date("01/31/2026", site_date_format="%m/%d/%Y")
    assert d == date(2026, 1, 31)
    assert check.outcome is CheckOutcome.PASS
    assert check.confidence.score == 0.95


def test_site_format_that_does_not_match_falls_back_to_fixed_order():
    """The value doesn't parse under the claimed site format at all (wrong
    field, OCR noise) -- falls back rather than failing outright."""
    d, check = validators.parse_iso_date("2025-08-16", site_date_format="%m/%d/%Y")
    assert d == date(2025, 8, 16)  # recovered via the %Y-%m-%d fallback
    assert check.outcome is CheckOutcome.PASS


def test_de_site_dot_format_is_never_ambiguous():
    """No competing %m.%d.%Y in this corpus -- the separator alone
    disambiguates, site-resolved or not."""
    d, check = validators.parse_iso_date("03.04.2025", site_date_format="%d.%m.%Y")
    assert d == date(2025, 4, 3)
    assert check.outcome is CheckOutcome.PASS
    assert check.confidence.score == 0.95

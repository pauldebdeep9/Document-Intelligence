from __future__ import annotations

from datetime import date
from decimal import Decimal

from isc.eval.normalise import normalise_date, normalise_decimal, normalise_string


def test_normalise_string_strips_and_casefolds():
    assert normalise_string("  Acme Supply Co  ") == "acme supply co"


def test_normalise_string_none_and_empty_both_none():
    assert normalise_string(None) is None
    assert normalise_string("   ") is None


def test_normalise_date_handles_a_date_object():
    assert normalise_date(date(2025, 8, 16)) == "2025-08-16"


def test_normalise_date_handles_every_corpus_site_format():
    assert normalise_date("16/08/2025") == "2025-08-16"       # SG: %d/%m/%Y
    assert normalise_date("08/16/2025") == "2025-08-16"       # US: %m/%d/%Y
    assert normalise_date("16.08.2025") == "2025-08-16"       # DE: %d.%m.%Y
    assert normalise_date("2025-08-16") == "2025-08-16"       # already ISO


def test_normalise_date_unparseable_is_none_not_a_guess():
    assert normalise_date("not-a-date") is None
    assert normalise_date(None) is None


def test_normalise_decimal_handles_comma_grouped_string():
    assert normalise_decimal("1,536.37") == Decimal("1536.37")


def test_normalise_decimal_handles_float_and_rounds_to_2dp():
    assert normalise_decimal(1536.375) == Decimal("1536.38")
    assert normalise_decimal(250.0) == Decimal("250.00")


def test_normalise_decimal_unparseable_is_none():
    assert normalise_decimal("not-a-number") is None
    assert normalise_decimal(None) is None

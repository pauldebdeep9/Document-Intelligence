"""Unit tests for extract/masters.py's check_description(): the description
cross-check added to catch a value that is textually short of the master's
truth (a wrapped table cell whose continuation never reached extract/ -- see
docs/LIMITATIONS.md) since resolve_part() alone only ever looks at
part_number.
"""

from __future__ import annotations

import json

import pytest

from isc.common.confidence import CheckOutcome
from isc.extract import masters


@pytest.fixture
def masters_dir(tmp_path):
    d = tmp_path / "masters"
    d.mkdir()
    (d / "parts.json").write_text(json.dumps([
        {"part_number": "PSU-24V-10A", "description": "Switched mode power supply 24V 10A",
         "uom": "EA"},
    ]))
    (d / "unmastered_parts.json").write_text("[]")
    return d


def test_exact_match_passes(masters_dir):
    check = masters.check_description(
        "PSU-24V-10A", "Switched mode power supply 24V 10A", masters_dir,
    )
    assert check.outcome is CheckOutcome.PASS
    assert check.confidence.score == 0.97


def test_case_and_whitespace_insensitive(masters_dir):
    check = masters.check_description(
        "PSU-24V-10A", "  switched MODE power supply 24v 10a  ", masters_dir,
    )
    assert check.outcome is CheckOutcome.PASS


def test_truncation_is_uncertain_not_fail(masters_dir):
    """The real P1-03 shape: the extracted text is a strict prefix of the
    master's -- the model received less than the full printed value, not a
    wrong one."""
    check = masters.check_description(
        "PSU-24V-10A", "Switched mode power supply 24V", masters_dir,
    )
    assert check.outcome is CheckOutcome.UNCERTAIN
    assert check.confidence.score == 0.75
    assert "partial match" in check.confidence.factors[0].detail


def test_genuinely_different_description_fails(masters_dir):
    check = masters.check_description("PSU-24V-10A", "Ball bearing 6205", masters_dir)
    assert check.outcome is CheckOutcome.FAIL
    assert check.confidence.score == 0.15


def test_unresolved_part_is_not_applicable(masters_dir):
    """No master description to compare against -- not evidence the
    description is wrong, the master simply has nothing to say about it,
    same reasoning as resolve_part()'s own unmastered case."""
    check = masters.check_description("ZZ-NOT-IN-MASTER", "Anything", masters_dir)
    assert check.confidence.factors == ()


def test_none_part_number_or_description_is_not_applicable(masters_dir):
    assert masters.check_description(None, "Anything", masters_dir).confidence.factors == ()
    assert masters.check_description("PSU-24V-10A", None, masters_dir).confidence.factors == ()


def test_empty_extracted_description_is_not_a_false_partial_match(masters_dir):
    """An empty string is a substring of everything in Python -- must not
    be misread as a (vacuous) partial match."""
    check = masters.check_description("PSU-24V-10A", "", masters_dir)
    assert check.outcome is CheckOutcome.FAIL


def test_resolve_site_date_format_exact_match_only(tmp_path):
    d = tmp_path / "masters_sites"
    d.mkdir()
    (d / "sites.json").write_text(json.dumps([
        {"site_id": "site_us42", "name": "Milwaukee Plant 42", "date_format": "%m/%d/%Y"},
    ]))
    assert masters.resolve_site_date_format("Milwaukee Plant 42", d) == "%m/%d/%Y"
    assert masters.resolve_site_date_format("Milwaukee Plant", d) is None  # near-miss, not fuzzy
    assert masters.resolve_site_date_format(None, d) is None

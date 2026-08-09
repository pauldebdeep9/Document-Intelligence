"""Unit tests for extract/spans.py.

The boundary-safety, normalisation, and row-shape tests are pure logic and
always run. Everything exercising `locate()`/`extract_rows()` end to end runs
against a real parsed corpus document rather than a synthetic fixture -- span
location is a string-search problem, and its real edge cases (column padding,
wrapped table cells, thousands separators, repeated part numbers) only show up
in what the renderer actually produces. Skipped when the corpus has not been
generated, same convention as test_corpus_fidelity.py / test_parse_fidelity.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isc.extract.spans import (
    SpanOutcome,
    _count_matches,
    _grouped_forms,
    _is_fragmented,
    _normalise_ws,
    extract_rows,
    locate,
)
from isc.models.acl import AclSet
from isc.models.document import Document
from isc.parse.chain import NativeTextParser

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
GOLD = ROOT / "data" / "gold" / "extraction"

_no_corpus = pytest.mark.skipif(
    not list(GOLD.glob("*.json")), reason="corpus not generated; run `make corpus`"
)


def _parse(name: str) -> Document:
    doc = Document(id="doc_spans_test", source_uri=name, content_sha256="x",
                    acl=AclSet(allow_terms=frozenset({"everyone:*"})))
    return NativeTextParser().parse(doc, (SYN / name).read_bytes())


def _gold(name: str) -> dict:
    return json.loads((GOLD / name).read_text())


# --- pure logic, no corpus needed ------------------------------------------

def test_whitespace_is_normalised_on_both_sides():
    assert _normalise_ws("PO Number             4522345741") == "PO Number 4522345741"


def test_grouped_form_adds_thousands_separators():
    assert _grouped_forms("1536.37") == ["1,536.37"]
    assert _grouped_forms("392589.57") == ["392,589.57"]


def test_grouped_form_empty_when_grouping_changes_nothing():
    assert _grouped_forms("250") == []


def test_short_numeric_value_does_not_match_inside_a_longer_number():
    """A quantity of '1' must not count as a fragment of '18' or a year like
    '2025' -- boundary-safe search, not a plain substring check."""
    assert _count_matches("1", "the year is 18 and PO 2025 was raised") == 0
    assert _count_matches("1", "quantity: 1 EA") == 1


def test_numeric_match_finds_grouped_form_in_haystack():
    assert _count_matches("1536.37", "unit price 1,536.37 per unit") == 1


def test_text_value_uses_plain_substring_match():
    assert _count_matches("EXW", "Incoterms EXW Payment Terms Net 30") == 1
    assert _count_matches("EXW", "no incoterm printed here") == 0


def test_comma_does_not_block_a_numeric_boundary_match():
    """A comma is not a word character, so a bare '1' still counts as a
    boundary-safe hit against the leading digit of '1,801.47' -- this is
    exactly why the multi-match guard exists, not something to prevent here."""
    row = "qty 1 EA unit price 1,801.47 extended 1,801.47"
    assert _count_matches("1", row) == 3


def test_is_fragmented_true_when_every_token_present_out_of_order():
    """Matching is case-sensitive (same as _count_matches), so the haystack
    here preserves the same capitalisation as the needle -- exactly what the
    real corpus does, since both sides come from the same printed text."""
    haystack = "24V Power Switched Mode 10A Supply"
    assert _is_fragmented("Switched Mode Power Supply 24V 10A", haystack) is True


def test_is_fragmented_false_when_a_token_is_missing():
    haystack = "Switched Mode Power 24V 10A"  # no "Supply"
    assert _is_fragmented("Switched Mode Power Supply 24V 10A", haystack) is False


def test_is_fragmented_requires_at_least_two_tokens():
    """A single-token value that fails a contiguous match has nothing left to
    call fragmented -- it either matched or it is simply not there."""
    assert _is_fragmented("Zephyrblorp", "Zephyrblorp is nowhere near this text") is False
    assert _is_fragmented("Zephyrblorp", "totally unrelated text") is False


def test_row_pattern_matches_a_line_item_row():
    line = "10       PLC-1756-L83          ControlLogix processor module           250   EA"
    rows = extract_rows(_row_doc([line]))
    assert rows == [(10, line)]


def test_row_pattern_excludes_the_table_header():
    """The header starts with a word ('Item'), not a digit."""
    rows = extract_rows(_row_doc([
        "Item     Part Number           Description                             Qty",
    ]))
    assert rows == []


def test_row_pattern_excludes_a_wrapped_description_continuation():
    """'10A' runs the digits straight into the letter with no space, so
    \\s+ after the ordinal group fails to match."""
    rows = extract_rows(_row_doc(["                               10A"]))
    assert rows == []


def _row_doc(lines: list[str]) -> Document:
    from isc.models.document import Block, BlockType, Page

    block = Block(id="blk_1", type=BlockType.PARAGRAPH, text="\n".join(lines),
                  page=1, reading_order=0)
    return Document(id="doc_row_test", source_uri="x.pdf", content_sha256="x",
                     acl=AclSet(allow_terms=frozenset({"everyone:*"})),
                     pages=[Page(number=1, blocks=[block])])


# --- against a real parsed corpus document ---------------------------------

@_no_corpus
def test_locates_a_simple_header_field():
    doc = _parse("po_000.pdf")
    located = locate(doc, "4522345741")
    assert located.outcome is SpanOutcome.FOUND
    assert located.span is not None
    assert located.span.page == 1
    assert located.span.document_id == doc.id


@_no_corpus
def test_locates_thousands_separated_amount_from_bare_model_number():
    """Page prints '1,536.37'; the extraction prompt requires the model to
    return amounts as '1536.37', digits and decimal point only."""
    doc = _parse("po_000.pdf")
    located = locate(doc, "1536.37")
    assert located.outcome is SpanOutcome.FOUND
    assert located.span is not None
    assert located.span.page == 1


@_no_corpus
def test_wrapped_description_is_fragmented_not_not_found():
    """'Switched mode power supply 24V' / '10A' wraps across two lines inside
    one table block. The rest of that row's columns sit between the two
    pieces in flattened text, so no contiguous match exists -- but every
    token IS present in the block, correctly extracted, so this must be
    FRAGMENTED (neutral), not NOT_FOUND (penalised). Scoring the parser's
    flattening as if it were the model's hallucination was the actual bug."""
    doc = _parse("po_000.pdf")
    located = locate(doc, "Switched mode power supply 24V 10A")
    assert located.outcome is SpanOutcome.FRAGMENTED
    assert located.span is None


@_no_corpus
def test_value_not_in_document_is_not_found():
    doc = _parse("po_000.pdf")
    located = locate(doc, "this string is not anywhere in the document")
    assert located.outcome is SpanOutcome.NOT_FOUND
    assert located.span is None


@_no_corpus
def test_none_value_is_not_found_without_raising():
    doc = _parse("po_000.pdf")
    located = locate(doc, None)
    assert located.outcome is SpanOutcome.NOT_FOUND
    assert located.span is None


@_no_corpus
def test_resolves_to_the_correct_page_on_a_multipage_document():
    """po_001 wraps to a second page; its total is only printed there."""
    g = _gold("po_001.json")
    assert g["meta"]["wraps_pages"]
    doc = _parse("po_001.pdf")
    located = locate(doc, g["normalised"]["total_amount"])
    assert located.outcome is SpanOutcome.FOUND
    assert located.span is not None
    assert located.span.page == 2


@_no_corpus
def test_document_wide_ambiguous_value_is_ambiguous_not_not_found():
    """po_000 has two line items priced/quantitied such that '1' is not
    unique document-wide (it also occurs elsewhere as a digit fragment) --
    this is the AMBIGUOUS outcome specifically, not NOT_FOUND: the value IS
    present, just not uniquely locatable, which is a different claim."""
    doc = _parse("po_000.pdf")
    located = locate(doc, "1")
    assert located.outcome is SpanOutcome.AMBIGUOUS
    assert located.span is None


@_no_corpus
def test_scoping_resolves_a_value_ambiguous_document_wide():
    """The same '1' that is AMBIGUOUS unscoped resolves to FOUND once scoped
    to the one row it actually belongs to."""
    doc = _parse("po_000.pdf")
    rows = extract_rows(doc)
    row_20 = next(text for ordinal, text in rows if ordinal == 20)
    located = locate(doc, "1", scope=row_20)
    assert located.outcome is SpanOutcome.FOUND
    assert located.span is not None


@_no_corpus
def test_extract_rows_aligns_with_gold_line_numbers_single_page():
    g = _gold("po_000.json")
    doc = _parse("po_000.pdf")
    rows = extract_rows(doc)
    assert [ordinal for ordinal, _ in rows] == [ln["line_number"] for ln in g["raw"]["lines"]]


@_no_corpus
def test_extract_rows_aligns_with_gold_line_numbers_42_line_document():
    """The multi-page, 42-line case that broke part-number-anchored scoping."""
    g = _gold("po_001.json")
    doc = _parse("po_001.pdf")
    rows = extract_rows(doc)
    assert len(rows) == len(g["raw"]["lines"]) == 42
    assert [ordinal for ordinal, _ in rows] == [ln["line_number"] for ln in g["raw"]["lines"]]

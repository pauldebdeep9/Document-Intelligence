"""Unit tests for extract/extractor.py: field mapping, the row-alignment
self-check, and confidence assembly (span-outcome, LEXICAL, MASTER_DATA,
AGREEMENT). Fixtures are synthetic (crafted Block text and a tiny masters
directory), not the real corpus -- the thing under test here is _wrap's own
branching, which does not depend on how the real renderer lays out a page.
spans.py's own matching behaviour against real documents is covered in
test_spans.py.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from isc.common.confidence import Confidence, Signal
from isc.extract.extractor import _date_field, _row_scopes, _wrap_purchase_order
from isc.models.acl import AclSet
from isc.models.document import Block, BlockType, Document, Page
from isc.models.records.purchase_order import POLineRaw, PurchaseOrderRaw

_ITEM_TABLE = (
    "Item  Part Number   Description         Qty  UoM  Unit Price  Extended  Promised\n"
    "10    PLC-1756-L83  ControlLogix module  3    EA   500.00      1500.00   16/08/2025\n"
    "20    PSU-24V-10A   Power supply         7    EA   100.00      700.00    17/08/2025"
)


@pytest.fixture
def masters_dir(tmp_path):
    """A tiny, controlled master list -- deliberately not the real corpus
    masters, so these tests do not silently depend on data/masters/*.json
    content that could change for unrelated reasons. PLC-1756-L83 is
    mastered, PSU-24V-10A is not (and is not on the unmastered exception
    list either), so the fixture exercises both resolve and genuine-miss."""
    d = tmp_path / "masters"
    d.mkdir()
    (d / "suppliers.json").write_text(json.dumps([
        {"supplier_id": "V1", "name": "Acme Supply Co", "country": "US"},
    ]))
    (d / "parts.json").write_text(json.dumps([
        {"part_number": "PLC-1756-L83", "description": "x", "uom": "EA"},
    ]))
    (d / "unmastered_parts.json").write_text(json.dumps([
        {"part_number": "ZZ-UNKNOWN", "description": "y", "uom": "EA"},
    ]))
    return d


def _doc(extra_text: str = "", items: str = _ITEM_TABLE) -> Document:
    blocks = [Block(id="b_title", type=BlockType.TITLE, text="PURCHASE ORDER",
                     page=1, reading_order=0)]
    if extra_text:
        blocks.append(Block(id="b_kv", type=BlockType.KEY_VALUE, text=extra_text,
                             page=1, reading_order=1))
    blocks.append(Block(id="b_items", type=BlockType.PARAGRAPH, text=items,
                         page=1, reading_order=2))
    return Document(id="doc_test", source_uri="x.pdf", content_sha256="x",
                     acl=AclSet(allow_terms=frozenset({"everyone:*"})),
                     pages=[Page(number=1, blocks=blocks)])


def _line(**overrides) -> POLineRaw:
    defaults = dict(line_number=10, part_number="PLC-1756-L83",
                     description="ControlLogix module", quantity=3,
                     unit_of_measure="EA", unit_price=500.0, extended_price=1500.0,
                     promised_date="16/08/2025")
    defaults.update(overrides)
    return POLineRaw(**defaults)


def _raw(**overrides) -> PurchaseOrderRaw:
    # Default total (2200.0) == 3*500 + 7*100 exactly, so AGREEMENT passes by
    # default; tests that want a mismatch override total_amount or a line.
    lines = overrides.pop("lines", [
        _line(),
        _line(line_number=20, part_number="PSU-24V-10A", description="Power supply",
              quantity=7, unit_price=100.0, extended_price=700.0, promised_date="17/08/2025"),
    ])
    defaults = dict(po_number="4500123456", po_date="16/08/2025",
                     supplier_name="Acme Supply Co", supplier_id=None,
                     ship_to_site="Plant 1", incoterms="FOB", payment_terms="Net 30",
                     currency="USD", total_amount=2200.0, buyer_contact=None)
    defaults.update(overrides)
    return PurchaseOrderRaw(lines=lines, **defaults)


def _wrap(raw, doc, masters_dir, base=None):
    return _wrap_purchase_order(raw, doc, base or Confidence.unknown(), masters_dir)


# --- field mapping (step 2 behaviour, unchanged by step 3) -----------------

def test_missing_field_uses_missing_with_certain_confidence(masters_dir):
    record = _wrap(_raw(), _doc(), masters_dir)
    assert record.buyer_contact.value is None
    assert record.buyer_contact.absent_reason is not None
    assert record.buyer_contact.confidence.score == 1.0


def test_present_but_unparseable_date_is_not_missing():
    field = _date_field(_doc(), "not-a-real-date", Confidence.unknown())
    assert field.value is None
    assert field.absent_reason is None


def test_string_fields_are_stripped(masters_dir):
    doc = _doc("Supplier   Acme Supply Co   Vendor Code   V1")
    record = _wrap(_raw(supplier_name="  Acme Supply Co  "), doc, masters_dir)
    assert record.supplier_name.value == "Acme Supply Co"


def test_amounts_convert_to_decimal(masters_dir):
    record = _wrap(_raw(), _doc(), masters_dir)
    assert record.total_amount.value == Decimal("2200.0")
    assert isinstance(record.total_amount.value, Decimal)


# --- row alignment (unchanged by step 3, re-verified with the new signature)

def test_row_alignment_succeeds_and_scopes_line_fields(masters_dir):
    record = _wrap(_raw(), _doc(), masters_dir)
    assert record.row_alignment_ok is True
    assert record.row_alignment_detail == ""
    assert record.lines[0].quantity.span is not None
    assert record.lines[1].quantity.span is not None


def test_row_count_mismatch_forces_every_line_field_to_span_none(masters_dir):
    record = _wrap(_raw(lines=[
        _line(),
        _line(line_number=20, part_number="PSU-24V-10A", quantity=7,
              unit_price=100.0, extended_price=700.0, promised_date="17/08/2025"),
        _line(line_number=30, part_number="X", quantity=1, unit_price=1.0,
              extended_price=1.0, promised_date="18/08/2025"),
    ]), _doc(), masters_dir)
    assert record.row_alignment_ok is False
    assert "row count" in record.row_alignment_detail
    for line in record.lines:
        assert line.line_number.span is None
        assert line.quantity.span is None
        assert line.part_number.span is None
        assert line.description.span is None
    assert record.lines[2].part_number.value == "X"  # values still populated


def test_ordinal_mismatch_forces_alignment_failure(masters_dir):
    record = _wrap(_raw(lines=[
        _line(line_number=99),  # wrong on purpose
        _line(line_number=20, part_number="PSU-24V-10A", quantity=7,
              unit_price=100.0, extended_price=700.0, promised_date="17/08/2025"),
    ]), _doc(), masters_dir)
    assert record.row_alignment_ok is False
    assert "ordinal" in record.row_alignment_detail
    assert all(line.quantity.span is None for line in record.lines)


def test_row_scopes_empty_lines_is_trivially_aligned():
    scopes, detail = _row_scopes(_doc(), [])
    assert scopes == []
    assert detail == ""


# --- span outcome feeding confidence (step 3) -------------------------------

def test_not_found_span_penalises_confidence(masters_dir):
    """A value asserted but absent from the document text is real negative
    evidence -- confidence must drop below the base."""
    base = Confidence.of(Signal.MODEL, 0.9)
    record = _wrap(_raw(ship_to_site="Nowhere On The Page"), _doc(), masters_dir, base)
    assert record.ship_to_site.confidence.score < base.score
    assert any("not found" in c for c in record.ship_to_site.conflicts)


def test_ambiguous_span_does_not_change_confidence(masters_dir):
    """A row engineered so quantity collides with itself: a quantity of 1
    printed alongside a unit_price AND extended_price of 1.00 gives three
    boundary-safe hits for '1' in the row (the standalone quantity, plus the
    leading digit of each '1.00' -- a comma/period is not a word character),
    so the scoped search is genuinely AMBIGUOUS, not FOUND or NOT_FOUND."""
    row = "10    PLC-1756-L83  ControlLogix module  1    EA   1.00      1.00   16/08/2025"
    doc = _doc(items=row)
    raw = _raw(lines=[_line(quantity=1, unit_price=1.0, extended_price=1.0)],
               total_amount=1.0)
    base = Confidence.of(Signal.MODEL, 0.9)
    record = _wrap(raw, doc, masters_dir, base)
    line = record.lines[0]
    assert line.quantity.span is None
    # score must equal base exactly -- AMBIGUOUS must not touch it at all,
    # not even reduce it slightly
    assert line.quantity.confidence.score == pytest.approx(base.score)
    assert line.quantity.conflicts == []


def test_found_span_alone_does_not_bonus_confidence(masters_dir):
    """FOUND is the absence of the NOT_FOUND penalty, not itself a
    corroborating bonus -- a field with nothing else applied (ship_to_site
    has no LEXICAL/MASTER_DATA/AGREEMENT check) stays exactly at
    base_confidence, provided its value is actually present in the text."""
    base = Confidence.of(Signal.MODEL, 0.9)
    doc = _doc("Ship To Plant 1 Currency USD")
    record = _wrap(_raw(), doc, masters_dir, base)
    assert record.ship_to_site.span is not None
    assert record.ship_to_site.confidence.score == pytest.approx(base.score)


# --- LEXICAL ------------------------------------------------------------

def test_lexical_pass_corroborates_po_number(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc("PO Number 4500123456 Order Date 16/08/2025")
    record = _wrap(_raw(po_number="4500123456"), doc, masters_dir, base)
    assert record.po_number.span is not None  # isolate LEXICAL, not NOT_FOUND
    assert record.po_number.confidence.score > base.score


def test_lexical_fail_conflicts_po_number(masters_dir):
    """'PO-BAD' does not match the SAP-style PO_NUMBER pattern."""
    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc("PO Number PO-BAD Order Date 16/08/2025")
    record = _wrap(_raw(po_number="PO-BAD"), doc, masters_dir, base)
    assert record.po_number.confidence.score < base.score
    assert any("po_number" in c for c in record.po_number.conflicts)


def test_ambiguous_date_gets_reduced_lexical_confidence(masters_dir):
    """d/m/Y and m/d/Y are indistinguishable for day <= 12 -- parse_iso_date's
    own ambiguity discount must show up as this field's LEXICAL factor."""
    doc = _doc("PO Date 03/04/2025")
    record = _wrap(_raw(po_date="03/04/2025"), doc, masters_dir)
    factor = next(f for f in record.po_date.confidence.factors if f.signal == Signal.LEXICAL)
    assert "AMBIGUOUS" in factor.detail


# --- MASTER_DATA ------------------------------------------------------------

def test_master_data_resolves_supplier_by_name(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc("Supplier Acme Supply Co Vendor Code V1")
    record = _wrap(_raw(supplier_name="Acme Supply Co"), doc, masters_dir, base)
    assert record.supplier_name.span is not None  # isolate MASTER_DATA, not NOT_FOUND
    assert record.supplier_name.confidence.score > base.score
    factors = record.supplier_name.confidence.factors
    factor = next(f for f in factors if f.signal == Signal.MASTER_DATA)
    assert "name" in factor.detail


def test_master_data_conflict_for_unresolvable_supplier(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc("Supplier Totally Unknown Co Vendor Code V9")
    record = _wrap(_raw(supplier_name="Totally Unknown Co"), doc, masters_dir, base)
    assert record.supplier_name.confidence.score < base.score
    assert any("not found" in c for c in record.supplier_name.conflicts)


def test_master_data_never_fuzzy_matches_confusable_names(masters_dir, tmp_path):
    """Two confusable suppliers in the master; a near-miss name must resolve
    to a miss, never to whichever one scores higher."""
    d = tmp_path / "masters2"
    d.mkdir()
    (d / "suppliers.json").write_text(json.dumps([
        {"supplier_id": "V1", "name": "Fastenal Industrial Supply Pte Ltd", "country": "SG"},
        {"supplier_id": "V2", "name": "Fastenal Industrial Services Pte Ltd", "country": "SG"},
    ]))
    (d / "parts.json").write_text("[]")
    (d / "unmastered_parts.json").write_text("[]")

    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc("Supplier Fastenal Industrial Pte Ltd Vendor Code V1")
    record = _wrap(_raw(supplier_name="Fastenal Industrial Pte Ltd"), doc, d, base)
    assert record.supplier_name.confidence.score < base.score
    assert any("not found" in c for c in record.supplier_name.conflicts)


def test_master_data_ignores_deliberately_unmastered_part(masters_dir):
    """A part on the unmastered exception list must not be flagged as a
    conflict -- it contributes nothing, same as an inapplicable check. Uses a
    part number ("ZZ-UNKNOWN") that also passes the PART_NUMBER LEXICAL
    pattern, so that signal legitimately corroborates and the only thing
    under test is that MASTER_DATA specifically never fires either way."""
    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc(items=_ITEM_TABLE.replace("PSU-24V-10A", "ZZ-UNKNOWN"))
    raw = _raw(lines=[
        _line(),
        _line(line_number=20, part_number="ZZ-UNKNOWN", description="Power supply",
              quantity=7, unit_price=100.0, extended_price=700.0, promised_date="17/08/2025"),
    ])
    record = _wrap(raw, doc, masters_dir, base)
    line = record.lines[1]
    assert not any(c.startswith("part not found") for c in line.part_number.conflicts)
    factors = line.part_number.confidence.factors
    assert not any(f.signal == Signal.MASTER_DATA for f in factors)


def test_master_data_conflicts_genuinely_unknown_part(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    record = _wrap(_raw(), _doc(), masters_dir, base)
    # PSU-24V-10A is neither mastered nor in the unmastered exception list
    line = record.lines[1]
    assert line.part_number.confidence.score < base.score
    assert any("not found" in c for c in line.part_number.conflicts)


# --- AGREEMENT ---------------------------------------------------------

def test_agreement_corroborates_when_total_matches_line_sum(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    doc = _doc("Order Total 2,200.00")
    record = _wrap(_raw(), doc, masters_dir, base)  # 3*500 + 7*100 == 2200
    assert record.total_amount.span is not None  # isolate AGREEMENT, not NOT_FOUND
    assert record.total_amount.confidence.score > base.score
    factor = next(f for f in record.total_amount.confidence.factors if f.signal == Signal.AGREEMENT)
    assert "agrees" in factor.detail


def test_agreement_conflicts_when_total_disagrees_with_line_sum(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    record = _wrap(_raw(total_amount=99999.0), _doc(), masters_dir, base)
    assert record.total_amount.confidence.score < base.score
    assert any("disagrees" in c for c in record.total_amount.conflicts)


def test_agreement_contributes_nothing_when_total_absent(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    record = _wrap(_raw(total_amount=None), _doc(), masters_dir, base)
    assert record.total_amount.absent_reason is not None
    assert record.total_amount.confidence.score == 1.0  # missing() certain-absent


def test_agreement_does_not_sum_printed_extended_prices(masters_dir):
    """The bug this guards: summing printed extended prices (rather than
    quantity x unit_price) would see two None-extended lines as a sum of
    zero and falsely conflict against a real total."""
    raw = _raw(lines=[
        _line(extended_price=None),
        _line(line_number=20, part_number="PSU-24V-10A", quantity=7,
              unit_price=100.0, extended_price=None, promised_date="17/08/2025"),
    ], total_amount=2200.0)  # still 3*500 + 7*100
    record = _wrap(raw, _doc(), masters_dir)
    factor = next(f for f in record.total_amount.confidence.factors if f.signal == Signal.AGREEMENT)
    assert "agrees" in factor.detail


def test_agreement_per_line_conflicts_on_wrong_extended_price(masters_dir):
    base = Confidence.of(Signal.MODEL, 0.8)
    record = _wrap(_raw(lines=[_line(extended_price=999999.0)],
                         total_amount=999999.0), _doc(), masters_dir, base)
    line = record.lines[0]
    assert line.extended_price.confidence.score < base.score
    assert any("disagrees" in c for c in line.extended_price.conflicts)


# --- rollup() / all_fields() / required_confidence() -----------------------
# Regression coverage for the exact bug found against the real corpus: a
# record with a bad total_amount and a bad line unit_price still reported
# rollup() == 1.0, because rollup() filtered to required_fields() (which
# never fails on this corpus) and fields() never traverses `lines` at all.

def test_all_fields_includes_line_items_keyed_by_position(masters_dir):
    record = _wrap(_raw(), _doc(), masters_dir)
    all_fields = record.all_fields()
    assert "lines[0].quantity" in all_fields
    assert "lines[1].part_number" in all_fields
    assert all_fields["lines[0].quantity"] is record.lines[0].quantity


def test_rollup_reflects_a_bad_line_field_not_just_required_fields(masters_dir):
    """po_number/supplier_name/po_date (required_fields()) are all sound here
    -- the only bad field is a line's unit_price. The old rollup() (weakest
    over required_fields() only) could not see this at all. Single-row item
    table so the row count still matches raw.lines (one line, not the
    module default two) and alignment succeeds -- otherwise every line span
    is suppressed for an unrelated reason and unit_price never even gets
    searched."""
    base = Confidence.of(Signal.MODEL, 0.95)
    row = "10    PLC-1756-L83  ControlLogix module  3    EA   500.00      1500.00   16/08/2025"
    doc = _doc("PO Number 4500123456 Order Date 16/08/2025 Supplier Acme Supply Co", items=row)
    raw = _raw(lines=[_line(unit_price=1.0)])  # 1.0 vs printed 500.00: NOT_FOUND
    record = _wrap(raw, doc, masters_dir, base)
    assert record.row_alignment_ok is True
    assert record.po_number.confidence.score > 0.9  # required fields still sound
    assert record.lines[0].unit_price.confidence.score < 0.5  # the bad field
    assert record.rollup().score < 0.5
    assert record.rollup().score <= record.lines[0].unit_price.confidence.score


def test_rollup_reflects_a_bad_nonrequired_header_field(masters_dir):
    """total_amount is not in required_fields() either -- same blind spot,
    header-level this time."""
    base = Confidence.of(Signal.MODEL, 0.95)
    record = _wrap(_raw(total_amount=99999.0), _doc(), masters_dir, base)  # disagrees
    assert record.total_amount.confidence.score < 0.5
    assert record.rollup().score < 0.5


def test_rollup_excludes_correctly_absent_fields(masters_dir):
    """missing() fields score certain() (1.0) and must not be allowed to
    (falsely) look like the best evidence in the record, and must not be
    allowed to be the reported minimum by coincidence either -- excluded
    explicitly regardless."""
    base = Confidence.of(Signal.MODEL, 0.95)
    record = _wrap(_raw(buyer_contact=None), _doc(), masters_dir, base)
    assert record.buyer_contact.absent_reason is not None
    populated_scores = [f.confidence.score for f in record.all_fields().values()
                         if f.absent_reason is None]
    assert record.rollup().score == pytest.approx(min(populated_scores))


def test_required_confidence_keeps_the_narrower_question(masters_dir):
    """required_confidence() is the old rollup() behaviour, preserved under
    its own name: unaffected by a bad line field."""
    base = Confidence.of(Signal.MODEL, 0.95)
    row = "10    PLC-1756-L83  ControlLogix module  3    EA   500.00      1500.00   16/08/2025"
    doc = _doc("PO Number 4500123456 Order Date 16/08/2025 Supplier Acme Supply Co", items=row)
    raw = _raw(lines=[_line(unit_price=1.0)])
    record = _wrap(raw, doc, masters_dir, base)
    assert record.row_alignment_ok is True
    assert record.required_confidence().score > 0.9
    assert record.required_confidence().score > record.rollup().score

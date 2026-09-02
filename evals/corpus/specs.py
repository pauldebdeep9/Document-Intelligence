"""Ground-truth specs for the synthetic Purchase Order gold corpus.

Each `DocumentSpec` carries the true `PurchaseOrder` field values (the answer key) alongside
the literal text that gets rendered onto the PDF's pages. Ground truth is therefore an input
to the generator, not something read back off a rendered document: where a spec's printed
text deliberately diverges from the true values (an omitted field, an inconsistent total,
trailing whitespace), that divergence is explicit in `pages`, right next to the true value it
diverges from.
"""

import textwrap
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from isc.models import LineItem, PurchaseOrder


class DocumentSpec(BaseModel):
    """A synthetic Purchase Order: true field values plus the literal text to render."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    purchase_order: PurchaseOrder
    pages: list[list[str]]
    notes: str


def _header_lines(
    po: PurchaseOrder,
    *,
    supplier_trailing_whitespace: bool = False,
) -> list[str]:
    """Format present header fields as printed lines, skipping true-null fields entirely."""
    lines = ["PURCHASE ORDER", ""]
    if po.po_number is not None:
        lines.append(f"PO Number: {po.po_number}")
    if po.po_date is not None:
        lines.append(f"PO Date: {po.po_date}")
    if po.supplier_name is not None:
        suffix = "   " if supplier_trailing_whitespace else ""
        lines.append(f"Supplier: {po.supplier_name}{suffix}")
    if po.ship_to_site is not None:
        lines.append(f"Ship-to Site: {po.ship_to_site}")
    if po.payment_terms is not None:
        lines.append(f"Payment Terms: {po.payment_terms}")
    if po.currency is not None:
        lines.append(f"Currency: {po.currency}")
    return lines


def _line_item_lines(number: int, item: LineItem, description_lines: list[str]) -> list[str]:
    """Format one line item as a numbered block; description is pre-wrapped by the caller."""
    return [
        f"{number}. Part Number: {item.part_number or '-'}",
        *[
            f"   Description: {line}" if index == 0 else f"      {line}"
            for index, line in enumerate(description_lines)
        ],
        f"   Quantity: {item.quantity if item.quantity is not None else '-'}",
        f"   Unit Price: {item.unit_price if item.unit_price is not None else '-'}",
    ]


def _total_line(po: PurchaseOrder) -> list[str]:
    """Format the total amount line, or nothing when it is genuinely absent."""
    if po.total_amount is None:
        return []
    return [f"Total Amount: {po.total_amount}"]


# --- po-001: baseline -------------------------------------------------------------------

_PO_001 = PurchaseOrder(
    po_number="PO-1001",
    po_date="2026-03-04",
    supplier_name="ACME Components",
    ship_to_site="Riverside Plant",
    payment_terms="Net 30",
    currency="USD",
    total_amount=Decimal("1985.00"),
    line_items=[
        LineItem(
            part_number="BRG-2050",
            description="Sealed ball bearing, 20mm bore",
            quantity=Decimal("10"),
            unit_price=Decimal("50.00"),
        ),
        LineItem(
            part_number="GSK-1140",
            description="Silicone gasket, 140mm diameter",
            quantity=Decimal("5"),
            unit_price=Decimal("97.00"),
        ),
        LineItem(
            part_number="SHF-3390",
            description="Stainless steel shaft, 390mm length",
            quantity=Decimal("100"),
            unit_price=Decimal("10.00"),
        ),
    ],
)

_PO_001_SPEC = DocumentSpec(
    doc_id="po-001",
    purchase_order=_PO_001,
    pages=[
        [
            *_header_lines(_PO_001),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_001.line_items[0], [_PO_001.line_items[0].description or ""]),
            *_line_item_lines(2, _PO_001.line_items[1], [_PO_001.line_items[1].description or ""]),
            *_line_item_lines(3, _PO_001.line_items[2], [_PO_001.line_items[2].description or ""]),
            "",
            *_total_line(_PO_001),
        ],
    ],
    notes=(
        "Baseline: single page, every header field present, three line items summing exactly "
        "to the printed total."
    ),
)

# --- po-002: payment_terms and currency are true-null, omitted from the page ------------

_PO_002 = PurchaseOrder(
    po_number="PO-1002",
    po_date="2026-02-10",
    supplier_name="Meridian Fasteners",
    ship_to_site="Dockside Warehouse 2",
    payment_terms=None,
    currency=None,
    total_amount=Decimal("640.00"),
    line_items=[
        LineItem(
            part_number="HXN-0812",
            description="Hex nut, M8",
            quantity=Decimal("400"),
            unit_price=Decimal("1.20"),
        ),
        LineItem(
            part_number="WSR-0820",
            description="Flat washer, M8",
            quantity=Decimal("800"),
            unit_price=Decimal("0.20"),
        ),
    ],
)

_PO_002_SPEC = DocumentSpec(
    doc_id="po-002",
    purchase_order=_PO_002,
    pages=[
        [
            *_header_lines(_PO_002),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_002.line_items[0], [_PO_002.line_items[0].description or ""]),
            *_line_item_lines(2, _PO_002.line_items[1], [_PO_002.line_items[1].description or ""]),
            "",
            *_total_line(_PO_002),
        ],
    ],
    notes=(
        "Payment Terms and Currency are true-null: their labels are omitted from the printed "
        "page entirely, not printed with a blank value."
    ),
)

# --- po-003: ambiguous date, preserved verbatim ------------------------------------------

_PO_003 = PurchaseOrder(
    po_number="PO-1003",
    po_date="03/04/2026",
    supplier_name="Coastal Hydraulics",
    ship_to_site="Plant 7",
    payment_terms="Net 45",
    currency="USD",
    total_amount=Decimal("212.50"),
    line_items=[
        LineItem(
            part_number="HYD-4471",
            description="Hydraulic hose fitting, 1/2in NPT",
            quantity=Decimal("25"),
            unit_price=Decimal("8.50"),
        ),
    ],
)

_PO_003_SPEC = DocumentSpec(
    doc_id="po-003",
    purchase_order=_PO_003,
    pages=[
        [
            *_header_lines(_PO_003),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_003.line_items[0], [_PO_003.line_items[0].description or ""]),
            "",
            *_total_line(_PO_003),
        ],
    ],
    notes=(
        "PO Date is printed as the ambiguous 03/04/2026; ground truth preserves it verbatim "
        "rather than resolving it to an unambiguous date."
    ),
)

# --- po-004 / po-005: near-duplicate part numbers, differing in the final digit ---------

_PO_004 = PurchaseOrder(
    po_number="PO-1004",
    po_date="2026-01-15",
    supplier_name="Titan Fabrication",
    ship_to_site="Bay 3",
    payment_terms="Net 30",
    currency="USD",
    total_amount=Decimal("330.00"),
    line_items=[
        LineItem(
            part_number="4500123456",
            description="Laser-cut steel bracket, 3mm gauge",
            quantity=Decimal("60"),
            unit_price=Decimal("5.50"),
        ),
    ],
)

_PO_004_SPEC = DocumentSpec(
    doc_id="po-004",
    purchase_order=_PO_004,
    pages=[
        [
            *_header_lines(_PO_004),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_004.line_items[0], [_PO_004.line_items[0].description or ""]),
            "",
            *_total_line(_PO_004),
        ],
    ],
    notes=(
        "Single line item with part number 4500123456; paired with po-005, which differs only "
        "in the final digit, to stress fine-grained identifier discrimination."
    ),
)

_PO_005 = _PO_004.model_copy(
    update={
        "po_number": "PO-1005",
        "po_date": "2026-01-16",
        "line_items": [
            LineItem(
                part_number="4500123457",
                description="Laser-cut steel bracket, 3mm gauge",
                quantity=Decimal("60"),
                unit_price=Decimal("5.50"),
            ),
        ],
    },
)

_PO_005_SPEC = DocumentSpec(
    doc_id="po-005",
    purchase_order=_PO_005,
    pages=[
        [
            *_header_lines(_PO_005),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_005.line_items[0], [_PO_005.line_items[0].description or ""]),
            "",
            *_total_line(_PO_005),
        ],
    ],
    notes=(
        "Identical to po-004 except the part number's final digit (...57 vs ...56); tests "
        "whether extraction/retrieval conflates near-duplicate identifiers."
    ),
)

# --- po-006: three pages, line item table spans the page 1 / page 2 boundary ------------

_PO_006 = PurchaseOrder(
    po_number="PO-1006",
    po_date="2026-04-02",
    supplier_name="Continental Bearings",
    ship_to_site="Distribution Center 5",
    payment_terms="Net 60",
    currency="USD",
    total_amount=Decimal("2960.00"),
    line_items=[
        LineItem(
            part_number="BRG-1001",
            description="Deep groove ball bearing, 30mm",
            quantity=Decimal("50"),
            unit_price=Decimal("12.00"),
        ),
        LineItem(
            part_number="BRG-1002",
            description="Deep groove ball bearing, 40mm",
            quantity=Decimal("40"),
            unit_price=Decimal("15.00"),
        ),
        LineItem(
            part_number="BRG-1003",
            description="Angular contact bearing, 25mm",
            quantity=Decimal("60"),
            unit_price=Decimal("18.00"),
        ),
        LineItem(
            part_number="BRG-1004",
            description="Angular contact bearing, 35mm",
            quantity=Decimal("34"),
            unit_price=Decimal("20.00"),
        ),
    ],
)

_PO_006_SPEC = DocumentSpec(
    doc_id="po-006",
    purchase_order=_PO_006,
    pages=[
        [
            *_header_lines(_PO_006),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_006.line_items[0], [_PO_006.line_items[0].description or ""]),
            *_line_item_lines(2, _PO_006.line_items[1], [_PO_006.line_items[1].description or ""]),
        ],
        [
            *_line_item_lines(3, _PO_006.line_items[2], [_PO_006.line_items[2].description or ""]),
            *_line_item_lines(4, _PO_006.line_items[3], [_PO_006.line_items[3].description or ""]),
        ],
        [
            *_total_line(_PO_006),
        ],
    ],
    notes=(
        "Three pages; the line item table's four entries are split 2/2 across the page 1 / "
        "page 2 boundary (page 2 does not repeat the header), with the total on page 3."
    ),
)

# --- po-007: three pages, page 2 entirely blank ------------------------------------------

_PO_007 = PurchaseOrder(
    po_number="PO-1007",
    po_date="2026-05-11",
    supplier_name="Northfield Industrial Supply",
    ship_to_site="Warehouse C",
    payment_terms="Net 30",
    currency="USD",
    total_amount=Decimal("945.00"),
    line_items=[
        LineItem(
            part_number="VLV-2201",
            description="Ball valve, 1in NPT",
            quantity=Decimal("15"),
            unit_price=Decimal("42.00"),
        ),
        LineItem(
            part_number="VLV-2202",
            description="Check valve, 1in NPT",
            quantity=Decimal("9"),
            unit_price=Decimal("35.00"),
        ),
    ],
)

_PO_007_SPEC = DocumentSpec(
    doc_id="po-007",
    purchase_order=_PO_007,
    pages=[
        [
            *_header_lines(_PO_007),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_007.line_items[0], [_PO_007.line_items[0].description or ""]),
            *_line_item_lines(2, _PO_007.line_items[1], [_PO_007.line_items[1].description or ""]),
        ],
        [],
        [
            *_total_line(_PO_007),
        ],
    ],
    notes=(
        "Three pages; page 2 is entirely blank (nothing drawn) to test that extract_pdf_pages "
        "preserves a genuinely blank page at its correct page_number rather than collapsing "
        "or renumbering it."
    ),
)

# --- po-008: total_amount is printed but does not sum from the line items ---------------

_PO_008 = PurchaseOrder(
    po_number="PO-1008",
    po_date="2026-06-20",
    supplier_name="Summit Electrical",
    ship_to_site="Substation 2",
    payment_terms="Net 30",
    currency="USD",
    total_amount=Decimal("750.00"),
    line_items=[
        LineItem(
            part_number="CBL-500",
            description="THHN copper wire, 500ft spool",
            quantity=Decimal("4"),
            unit_price=Decimal("85.00"),
        ),
        LineItem(
            part_number="CBL-501",
            description="THHN copper wire, 250ft spool",
            quantity=Decimal("3"),
            unit_price=Decimal("55.00"),
        ),
    ],
)

_PO_008_SPEC = DocumentSpec(
    doc_id="po-008",
    purchase_order=_PO_008,
    pages=[
        [
            *_header_lines(_PO_008),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_008.line_items[0], [_PO_008.line_items[0].description or ""]),
            *_line_item_lines(2, _PO_008.line_items[1], [_PO_008.line_items[1].description or ""]),
            "",
            *_total_line(_PO_008),
        ],
    ],
    notes=(
        "Total Amount is printed as 750.00 but the two line items sum to only 505.00; ground "
        "truth preserves the printed, inconsistent total rather than the computed sum, per the "
        "extraction contract's prohibition on recalculating totals."
    ),
)

# --- po-009: supplier name with an ampersand, printed with trailing whitespace ----------

_PO_009 = PurchaseOrder(
    po_number="PO-1009",
    po_date="2026-07-08",
    supplier_name="Smith & Sons Machining",
    ship_to_site="Unit 12",
    payment_terms="Net 15",
    currency="USD",
    total_amount=Decimal("180.00"),
    line_items=[
        LineItem(
            part_number="MCH-0090",
            description="CNC-machined spacer, brass",
            quantity=Decimal("20"),
            unit_price=Decimal("9.00"),
        ),
    ],
)

_PO_009_SPEC = DocumentSpec(
    doc_id="po-009",
    purchase_order=_PO_009,
    pages=[
        [
            *_header_lines(_PO_009, supplier_trailing_whitespace=True),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_009.line_items[0], [_PO_009.line_items[0].description or ""]),
            "",
            *_total_line(_PO_009),
        ],
    ],
    notes=(
        "Supplier name contains an ampersand (Smith & Sons Machining); tests that extraction "
        "and retrieval preserve the ampersand character verbatim rather than escaping, "
        "dropping, or otherwise mangling it."
    ),
)

# --- po-010: description long enough to cross a chunk_size=1200 chunk boundary ----------

# Longer than the default chunk_size (1200) by construction: no single 1200-character
# chunk_pages() window can hold it whole, regardless of its offset within the page text.
# Built from uniquely numbered segments (not a repeated sentence) so that any 50-character
# span anywhere in the string is provably unique to its position, not just probably unique —
# a repeated sentence would let the same substring recur at multiple chunk offsets and defeat
# any test that tries to locate a span by content.
_PO_010_DESCRIPTION = " ".join(
    f"titanium-bracket-segment-{index:04d}" for index in range(1, 200)
)[:1500]
assert len(_PO_010_DESCRIPTION) == 1500

_PO_010 = PurchaseOrder(
    po_number="PO-1010",
    po_date="2026-08-01",
    supplier_name="Vantage Aerospace Components",
    ship_to_site="Hangar 4",
    payment_terms="Net 30",
    currency="USD",
    total_amount=Decimal("4400.00"),
    line_items=[
        LineItem(
            part_number="BRK-7701",
            description=_PO_010_DESCRIPTION,
            quantity=Decimal("40"),
            unit_price=Decimal("110.00"),
        ),
    ],
)


def _wrap(text: str, width: int = 90) -> list[str]:
    """Word-wrap long text into fixed-width lines for rendering only."""
    return textwrap.wrap(text, width=width)


_PO_010_SPEC = DocumentSpec(
    doc_id="po-010",
    purchase_order=_PO_010,
    pages=[
        [
            *_header_lines(_PO_010),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_010.line_items[0], _wrap(_PO_010_DESCRIPTION)),
            "",
            *_total_line(_PO_010),
        ],
    ],
    notes=(
        "Line item description is a deterministic 1500-character string, longer than the "
        "default 1200-character chunk_size, guaranteeing by construction that no single "
        "chunk_pages() window can contain it whole."
    ),
)

# --- po-011: currency shown as a bare symbol, no ISO code anywhere -----------------------

_PO_011 = PurchaseOrder(
    po_number="PO-1011",
    po_date="2026-08-15",
    supplier_name="Alpine Precision Tools",
    ship_to_site="Site B",
    payment_terms="Net 30",
    currency="$",
    total_amount=Decimal("612.00"),
    line_items=[
        LineItem(
            part_number="TL-3305",
            description="Carbide end mill, 1/4in, 4-flute",
            quantity=Decimal("24"),
            unit_price=Decimal("25.50"),
        ),
    ],
)

_PO_011_SPEC = DocumentSpec(
    doc_id="po-011",
    purchase_order=_PO_011,
    pages=[
        [
            *_header_lines(_PO_011),
            "",
            "Line Items:",
            *_line_item_lines(1, _PO_011.line_items[0], [_PO_011.line_items[0].description or ""]),
            "",
            *_total_line(_PO_011),
        ],
    ],
    notes=(
        "Currency is printed only as a bare '$' symbol; no ISO 4217 code (e.g. USD) appears "
        "anywhere in the document, and ground truth preserves the bare symbol rather than "
        "inferring an ISO code from geography or convention."
    ),
)

# --- po-012: zero line items --------------------------------------------------------------

_PO_012 = PurchaseOrder(
    po_number="PO-1012",
    po_date="2026-08-20",
    supplier_name="Bramwell Logistics",
    ship_to_site="Yard 9",
    payment_terms="Net 30",
    currency="USD",
    total_amount=None,
    line_items=[],
)

_PO_012_SPEC = DocumentSpec(
    doc_id="po-012",
    purchase_order=_PO_012,
    pages=[
        [
            *_header_lines(_PO_012),
            "",
            "Line Items:",
            "  None",
        ],
    ],
    notes=(
        "Zero line items; the Line Items section is present but literally empty, and "
        "total_amount is genuinely absent rather than printed as zero."
    ),
)

DOCUMENT_SPECS: list[DocumentSpec] = [
    _PO_001_SPEC,
    _PO_002_SPEC,
    _PO_003_SPEC,
    _PO_004_SPEC,
    _PO_005_SPEC,
    _PO_006_SPEC,
    _PO_007_SPEC,
    _PO_008_SPEC,
    _PO_009_SPEC,
    _PO_010_SPEC,
    _PO_011_SPEC,
    _PO_012_SPEC,
]

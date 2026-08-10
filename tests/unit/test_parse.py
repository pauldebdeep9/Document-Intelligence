from __future__ import annotations

import io

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

from isc.models.acl import AclSet
from isc.models.document import BlockType, DocType, Document
from isc.parse.chain import (
    LayoutParser,
    NativeTextParser,
    OcrParser,
    _column_starts,
    _reconstruct_table,
    _table_text,
    parse_with_fallback,
)

_STYLES = getSampleStyleSheet()


def _build_pdf(pages: list[list[str]]) -> bytes:
    """One page per list of paragraph strings; a Spacer between each so pypdf
    sees a blank line and splits them into separate blocks."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    for i, paragraphs in enumerate(pages):
        for text in paragraphs:
            story.append(Paragraph(text, _STYLES["Normal"]))
            story.append(Spacer(1, 24))
        if i < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)
    return buf.getvalue()


def _make_doc(doc_id: str = "doc_test") -> Document:
    return Document(id=doc_id, source_uri="test.pdf", content_sha256="x",
                     acl=AclSet(allow_terms=frozenset({"everyone:*"})))


@pytest.fixture(scope="module")
def two_page_pdf() -> bytes:
    return _build_pdf([
        ["PURCHASE ORDER", "First page paragraph one.", "First page paragraph two."],
        ["Second page paragraph."],
    ])


def test_block_ids_are_deterministic_across_two_parses(two_page_pdf: bytes) -> None:
    doc = _make_doc()
    parser = NativeTextParser()
    ids1 = [b.id for b in parser.parse(doc, two_page_pdf).blocks()]
    ids2 = [b.id for b in parser.parse(doc, two_page_pdf).blocks()]
    assert ids1 == ids2


def test_reading_order_strictly_increasing_across_pages(two_page_pdf: bytes) -> None:
    parsed = NativeTextParser().parse(_make_doc(), two_page_pdf)
    orders = [b.reading_order for b in parsed.blocks()]
    assert orders == list(range(len(orders)))


def test_multi_page_document_yields_right_page_count(two_page_pdf: bytes) -> None:
    parsed = NativeTextParser().parse(_make_doc(), two_page_pdf)
    assert len(parsed.pages) == 2


def test_doc_type_detection_returns_purchase_order_with_confidence(two_page_pdf: bytes) -> None:
    parsed = NativeTextParser().parse(_make_doc(), two_page_pdf)
    assert parsed.doc_type == DocType.PURCHASE_ORDER
    assert parsed.doc_type_confidence is not None


def test_fallback_chain_records_native_parser_undegraded(two_page_pdf: bytes) -> None:
    parsed, conf = parse_with_fallback(
        _make_doc(), two_page_pdf, ".pdf",
        [NativeTextParser(), LayoutParser(), OcrParser()],
    )
    assert parsed.parser == "native"
    assert parsed.parser_degraded is False
    assert conf.score == pytest.approx(1.0)


def test_bbox_is_none_everywhere(two_page_pdf: bytes) -> None:
    parsed = NativeTextParser().parse(_make_doc(), two_page_pdf)
    assert parsed.blocks(), "fixture produced no blocks to check"
    assert all(b.bbox is None for b in parsed.blocks())


# --- table reconstruction: pure logic, real corpus text ---------------
# Fixture text taken verbatim from a real parsed po_000.pdf block (P1-04),
# not invented -- the exact case that produced 14 line-description false
# negatives in the P1-03 eval run: 'Switched mode power supply 24V' wraps
# onto its own physical line as '10A' inside the table block.

_REAL_TABLE_BLOCK = (
    "Item     Part Number           Description                             "
    "Qty        UoM      Unit Price         Extended            Promised\n"
    "10       PLC-1756-L83          ControlLogix processor module           "
    "250        EA       1,536.37                               18/11/2025\n"
    "20       PSU-24V-10A           Switched mode power supply 24V          "
    "1          EA       198.57                                 26/09/2025\n"
    "                               10A\n"
    "30       PSU-24V-10A           Switched mode power supply 24V          "
    "50         EA       165.97                                 05/11/2025\n"
    "                               10A"
)


def test_column_starts_finds_gaps_after_leading_whitespace():
    starts = _column_starts("   Item     Part Number")
    assert starts[0] == 3  # skips the 3-space lead, not itself a column boundary


def test_reconstruct_table_returns_none_for_non_table_text():
    assert _reconstruct_table("PO Number   4500123456   Order Date   16/08/2025") is None
    assert _reconstruct_table("A single line of prose.") is None


def test_reconstruct_table_folds_continuation_into_the_right_column():
    table = _reconstruct_table(_REAL_TABLE_BLOCK)
    assert table is not None
    assert table.header_rows == 1
    assert table.rows[0][2] == "Description"  # header preserved as row 0
    # exactly 4 rows total: header + 3 line items -- the two continuation
    # lines must NOT appear as rows of their own
    assert len(table.rows) == 4
    assert table.rows[2] == [
        "20", "PSU-24V-10A", "Switched mode power supply 24V 10A",
        "1", "EA", "198.57", "", "26/09/2025",
    ]
    assert table.rows[3][2] == "Switched mode power supply 24V 10A"


def test_table_text_keeps_row_pattern_matching():
    """The reconstructed text must still look like rows to extract_rows()/
    ROW_PATTERN -- ordinal at line start, followed by whitespace -- with the
    continuation gone as a separate physical line entirely."""
    from isc.parse.chain import ROW_PATTERN

    table = _reconstruct_table(_REAL_TABLE_BLOCK)
    text = _table_text(table)
    lines = text.split("\n")
    assert len(lines) == 4  # header + 3 rows, no stray continuation line
    matches = [ROW_PATTERN.match(ln) for ln in lines[1:]]
    assert all(matches)
    assert [int(m.group(1)) for m in matches] == [10, 20, 30]
    assert "10A" not in lines[0]  # sanity: continuation text landed on row 20/30, not the header
    assert "24V 10A" in lines[2]
    assert "24V 10A" in lines[3]


def test_reconstruct_table_with_no_continuations_is_unchanged():
    table = _reconstruct_table(
        "Item  Part  Description  Qty\n"
        "10    P1    Widget       3\n"
        "20    P2    Gadget       7"
    )
    assert table is not None
    assert len(table.rows) == 3
    assert table.rows[1] == ["10", "P1", "Widget", "3"]
    assert table.rows[2] == ["20", "P2", "Gadget", "7"]


def test_reconstruct_table_survives_a_row_with_an_empty_cell():
    """This corpus omits the Extended-price column on some documents
    entirely (see ADR 0005) -- a real row can have a blank middle cell,
    which merges two columns' worth of whitespace into one gap. Column
    boundaries must still come from the header (always fully populated),
    not get silently lost because the first row happened to have a gap."""
    table = _reconstruct_table(
        "Item  Part  Description  Qty  Extended  Promised\n"
        "10    P1    Widget       3              16/08/2025"
    )
    assert table is not None
    assert table.rows[1] == ["10", "P1", "Widget", "3", "", "16/08/2025"]


def test_reconstruct_table_corrects_for_a_header_stripped_shorter_than_its_rows():
    """Regression for the exact mechanism _split_blocks() creates: str.strip()
    on the whole block text removes leading whitespace from only the first
    line, so a block whose renderer indented every line uniformly can end up
    with a header reporting a smaller left margin than the rows that follow
    it, even though they were aligned in the original rendering. Simulated
    directly here rather than through a rendered PDF, since the PDF fixture
    elsewhere in this file avoids the renderer-side margin that would
    trigger it."""
    text = (
        "Item  Part Number   Description                      Qty  Promised\n"
        "         10    PLC-1756-L83  ControlLogix module               3    16/08/2025\n"
        "         20    PSU-24V-10A   Switched mode power supply 24V    1    26/09/2025\n"
        "                             10A"
    )
    table = _reconstruct_table(text)
    assert table is not None
    assert table.rows[0] == ["Item", "Part Number", "Description", "Qty", "Promised"]
    assert table.rows[2] == [
        "20", "PSU-24V-10A", "Switched mode power supply 24V 10A", "1", "26/09/2025",
    ]


def test_reconstruct_table_handles_a_row_with_no_header():
    """Defensive case: if the first line is itself row-shaped, there is no
    header, and header_rows must reflect that rather than mislabelling the
    first line item as a header."""
    table = _reconstruct_table("10    P1    Widget       3\n20    P2    Gadget       7")
    assert table is not None
    assert table.header_rows == 0
    assert len(table.rows) == 2


# --- table reconstruction: end to end through NativeTextParser --------

@pytest.fixture(scope="module")
def table_pdf() -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    table_text = (
        "Item  Part Number   Description                      Qty  UoM  Promised\n"
        "10    PLC-1756-L83  ControlLogix module               3    EA   16/08/2025\n"
        "20    PSU-24V-10A   Switched mode power supply 24V    1    EA   26/09/2025\n"
        "                    10A"
    )
    # leftIndent=0: the real corpus's item table renders flush with the page
    # margin (verified against a real parsed document). _STYLES["Code"]'s
    # default 36pt indent is a styling artifact of this fixture's font
    # choice, not a property of how gen_corpus.py's renderer lays out a
    # table -- using it here would test an indentation _split_blocks()
    # doesn't actually strip consistently (only the block's very first
    # line loses a leading run to str.strip(), not the rows after it),
    # which no real document exhibits.
    flush = ParagraphStyle("flush", parent=_STYLES["Code"], leftIndent=0)
    story = [
        Paragraph("PURCHASE ORDER", _STYLES["Normal"]), Spacer(1, 24),
        Preformatted(table_text, flush), Spacer(1, 24),
    ]
    doc.build(story)
    return buf.getvalue()


def test_parser_emits_a_table_block_with_reconstructed_text(table_pdf: bytes) -> None:
    parsed = NativeTextParser().parse(_make_doc(), table_pdf)
    tables = [b for b in parsed.blocks() if b.type == BlockType.TABLE]
    assert len(tables) == 1
    block = tables[0]
    assert block.table is not None
    assert "10A" not in block.text.split("\n")[0]  # not stuck on the header
    assert any("24V 10A" in line for line in block.text.split("\n"))
    assert not any(line.strip() == "10A" for line in block.text.split("\n")), (
        "continuation must not survive as its own line"
    )


def test_parser_does_not_reclassify_non_table_blocks(two_page_pdf: bytes) -> None:
    parsed = NativeTextParser().parse(_make_doc(), two_page_pdf)
    assert not any(b.type == BlockType.TABLE for b in parsed.blocks())
    assert all(b.table is None for b in parsed.blocks() if b.type != BlockType.TABLE)

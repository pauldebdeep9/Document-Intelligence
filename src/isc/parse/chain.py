"""Parser fallback chain with confidence penalties.

Order: native text layer -> layout model -> OCR. Each fallback is *recorded* and
discounts downstream confidence, so a field read off a bad OCR pass can never
present as clean. That linkage is the whole reason the chain is explicit.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Protocol

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from isc.common.confidence import Confidence, Signal
from isc.common.errors import ParseError
from isc.common.ids import block_id
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.models.document import Block, BlockType, DocType, Document, Page, Table

log = get_logger("parse")

# Discount applied when this parser had to be used. Native text is trusted.
PENALTY = {"native": 1.0, "layout": 0.95, "ocr": 0.80}

# The corpus prints header fields as table cells, not "Label: value" — e.g.
# "PO Number        4522345741        Order Date        16/08/2025", two
# label/value pairs per line, columns separated by runs of 2+ spaces. A block
# counts as KEY_VALUE when most of its non-empty lines split into an even
# number of cells whose label positions look like short capitalised labels.
# Anything else heuristic-detected as a block is PARAGRAPH.
_COLUMN_GAP = re.compile(r" {2,}")
_LABEL_CELL = re.compile(r"^[A-Z][A-Za-z0-9]*(?:[ /&-][A-Z][A-Za-z0-9]*)*$")
_KV_LINE_RATIO = 0.6

# Structural shape of a printed line-item row: a physical line beginning with
# an ordinal (the printed "Item" number) followed by a separate token. Shared
# with extract/spans.py's extract_rows() -- the same rule has to find the
# same rows whether it is scoping a span search after this module has already
# wrapped a document (P1-02) or reconstructing a table block before wrapping
# it (P1-04, below). Two independently maintained copies of this pattern
# would drift the moment either changes and nothing would notice.
ROW_PATTERN = re.compile(r"^\s*(\d+)\s+(\S+)\s")


def _kv_pairs_in_line(line: str) -> int:
    cells = [c for c in _COLUMN_GAP.split(line.strip()) if c]
    if len(cells) < 2 or len(cells) % 2 != 0:
        return 0
    labels = cells[0::2]
    if all(len(lb) < 40 and _LABEL_CELL.match(lb) for lb in labels):
        return len(cells) // 2
    return 0


# --- table reconstruction ---------------------------------------------
# A wrapped cell -- a description too long for its column, continuing on the
# next physical line -- used to be silently dropped: _split_blocks() only
# splits on blank lines, so the continuation stayed part of the same block's
# text but was never attached to its row, and Document.text() (what the LLM
# actually reads) carried the truncated first line only. Measured against
# the P1-03 eval run: 14 line-description false negatives, all the same
# recurring part, confirmed against the source PDF -- not a model misread,
# not a gold error, a parser gap. See docs/LIMITATIONS.md.


def _column_starts(row: str) -> list[int]:
    """Character offset where each column begins, from runs of 2+ spaces in
    a row line -- the same column-gap convention _is_key_value() relies on.
    Leading whitespace (e.g. from a renderer's own block margin) is not
    itself a column boundary, so the search for gaps starts after it."""
    lead = len(row) - len(row.lstrip(" "))
    starts = [lead]
    for m in _COLUMN_GAP.finditer(row, lead):
        starts.append(m.end())
    return starts


def _slice_row(line: str, starts: list[int]) -> list[str]:
    cells = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else None
        cells.append(line[start:end].strip() if start < len(line) else "")
    return cells


def _reconstruct_table(text: str) -> Table | None:
    """Detect and reconstruct a printed line-item table from a block's raw
    text: an optional header line naming columns, one ROW_PATTERN-shaped
    line per item, and zero or more continuation lines -- a physical line
    that does not start with an ordinal and follows a line that does --
    folded into the row above by column position (not assumed to always be
    the description column; whichever column the continuation's text lines
    up under is the one it extends).

    Column boundaries come from the header, not a row: a row can have an
    empty cell (this corpus omits the Extended-price column on some
    documents entirely -- see ADR 0005), which merges two columns' worth of
    whitespace into one gap and silently loses a boundary; the header
    always has every label populated, so it is the only reliable source for
    the *shape*. But the header cannot be sliced at its own offsets and
    reused for the rows unmodified: _split_blocks() strips leading
    whitespace off only the very first line of a block (str.strip() on the
    whole block text), never the lines after it, so the header can end up
    with a smaller left margin than the rows that follow even though the
    original rendering aligned them. The header's own boundaries are
    shifted by (row's margin - header's margin) before being used to slice
    the rows, so the *shape* comes from the header and the *position* comes
    from the rows -- both must hold for a continuation's lone cell to land
    in the right column.

    Returns None when the block has no row-shaped line at all: not every
    block is a table, and misreading a KEY_VALUE or PARAGRAPH block as one
    would be worse than leaving it unclassified.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2 or not any(ROW_PATTERN.match(ln) for ln in lines[1:]):
        return None

    has_header = not ROW_PATTERN.match(lines[0])
    row_lines = lines[1:] if has_header else lines

    rows: list[list[str]] = []
    if has_header:
        header_starts = _column_starts(lines[0])
        row_lead = len(row_lines[0]) - len(row_lines[0].lstrip(" "))
        shift = row_lead - header_starts[0]
        starts = [s + shift for s in header_starts]
        rows.append([c for c in _COLUMN_GAP.split(lines[0].strip()) if c])
    else:
        starts = _column_starts(row_lines[0])

    for line in row_lines:
        cells = _slice_row(line, starts)
        if ROW_PATTERN.match(line):
            rows.append(cells)
        elif rows:
            prev = rows[-1]
            for i, cell in enumerate(cells):
                if cell:
                    prev[i] = f"{prev[i]} {cell}".strip() if prev[i] else cell
    return Table(rows=rows, header_rows=1 if has_header else 0)


def _table_text(table: Table) -> str:
    """Plain-text rendering that preserves ROW_PATTERN matching (ordinal at
    line start, followed by whitespace) so extract_rows() and
    Document.text() see one complete line per row -- continuations already
    folded in -- without either needing to know reconstruction happened.

    Columns are padded to their own widest value, not joined with a flat
    2-space gap: measured against the real corpus, a flat join loses the
    original table's vertical alignment the moment one row's Description
    differs in length from another's (routine -- "Spare parts kit 4420" vs
    "ControlLogix processor module"), so Unit Price and Extended end up
    sitting at different horizontal offsets on every other row with no
    consistent gap between them. Confirmed as the cause of a real
    regression: po_018.pdf's extraction went from 8/8 extended_price values
    correct (before this reconstruction existed) to 8/8 missed (with the
    flat-join rendering) despite the values being genuinely present and
    correctly reconstructed -- the model could no longer tell where one
    column ended and the next began. See docs/LIMITATIONS.md.
    """
    if not table.rows:
        return ""
    n_cols = max(len(row) for row in table.rows)
    widths = [
        max((len(row[i]) for row in table.rows if i < len(row)), default=0)
        for i in range(n_cols)
    ]
    lines = []
    for row in table.rows:
        padded = [
            cell.ljust(widths[i] + 2) if i < n_cols - 1 else cell
            for i, cell in enumerate(row)
        ]
        lines.append("".join(padded).rstrip())
    return "\n".join(lines)


# Content markers used to detect doc_type. Extend deliberately, one entry per
# doc type the corpus actually carries — a guess here is worse than UNKNOWN.
_DOC_TYPE_MARKERS: list[tuple[str, DocType]] = [
    ("PURCHASE ORDER", DocType.PURCHASE_ORDER),
]


class Parser(Protocol):
    name: str
    def can_parse(self, data: bytes, suffix: str) -> bool: ...
    def parse(self, doc: Document, data: bytes) -> Document: ...


class NativeTextParser:
    """pypdf text layer. Fast, exact when the PDF is digitally generated."""
    name = "native"

    def can_parse(self, data: bytes, suffix: str) -> bool:
        return suffix.lower() == ".pdf"

    def parse(self, doc: Document, data: bytes) -> Document:
        reader = PdfReader(BytesIO(data))
        pages: list[Page] = []
        reading_order = 0
        title_set = False
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            blocks: list[Block] = []
            for raw_block_text in _split_blocks(_extract_page_text(pdf_page)):
                table = None
                block_text = raw_block_text
                if not title_set and page_number == 1:
                    btype = BlockType.TITLE
                    title_set = True
                elif (table := _reconstruct_table(raw_block_text)) is not None:
                    btype = BlockType.TABLE
                    block_text = _table_text(table)
                elif _is_key_value(raw_block_text):
                    btype = BlockType.KEY_VALUE
                else:
                    btype = BlockType.PARAGRAPH
                blocks.append(Block(
                    id=block_id(doc.id, page_number, reading_order, block_text),
                    type=btype,
                    text=block_text,
                    table=table,
                    page=page_number,
                    layout_confidence=1.0,
                    reading_order=reading_order,
                ))
                reading_order += 1
            pages.append(Page(number=page_number, blocks=blocks))

        doc_type, doc_type_confidence = _detect_doc_type(pages)
        return doc.model_copy(update={
            "pages": pages,
            "doc_type": doc_type,
            "doc_type_confidence": doc_type_confidence,
        })


class LayoutParser:
    """Block/table detection. Populates Block.bbox and Table.rows."""
    name = "layout"

    def can_parse(self, data: bytes, suffix: str) -> bool:
        return suffix.lower() in {".pdf", ".png", ".jpg", ".tiff"}

    def parse(self, doc: Document, data: bytes) -> Document:
        raise NotImplementedError("week 2: layout model + table reconstruction")


class OcrParser:
    """Last resort. Sets Block.ocr_confidence per block."""
    name = "ocr"

    def can_parse(self, data: bytes, suffix: str) -> bool:
        return True

    def parse(self, doc: Document, data: bytes) -> Document:
        raise NotImplementedError("week 2: tesseract / Document Intelligence")


def _extract_page_text(pdf_page) -> str:  # noqa: ANN001 - pypdf.PageObject, not worth importing
    """Layout mode preserves column spacing pypdf's default mode collapses. Some
    malformed content trips the layout-mode code path in pypdf itself, so fall
    back to default extraction rather than losing the page."""
    try:
        text = pdf_page.extract_text(extraction_mode="layout")
    except Exception:  # noqa: BLE001 - genuinely any pypdf internal failure, by design
        text = pdf_page.extract_text()
    return text or ""


def _split_blocks(text: str) -> list[str]:
    """Blank-line-separated blocks. No table detection — that's P1-04's job, and a
    wrong guess here would propagate into every downstream stage."""
    return [b.strip() for b in re.split(r"\n\s*\n+", text.strip()) if b.strip()]


def _is_key_value(block_text: str) -> bool:
    lines = [ln for ln in block_text.splitlines() if ln.strip()]
    if not lines:
        return False
    hits = sum(1 for ln in lines if _kv_pairs_in_line(ln) > 0)
    return hits / len(lines) >= _KV_LINE_RATIO


def _detect_doc_type(pages: list[Page]) -> tuple[DocType, float]:
    text = " ".join(b.text for p in pages for b in p.blocks).upper()
    for marker, doc_type in _DOC_TYPE_MARKERS:
        if marker in text:
            return doc_type, 0.95
    return DocType.UNKNOWN, 0.0


def parse_with_fallback(doc: Document, data: bytes, suffix: str,
                        parsers: list[Parser]) -> tuple[Document, Confidence]:
    errors: list[str] = []
    for parser in parsers:
        if not parser.can_parse(data, suffix):
            continue
        with span("parse.attempt", parser=parser.name, doc=doc.id):
            try:
                parsed = parser.parse(doc, data)
            except (NotImplementedError, PyPdfError) as exc:
                errors.append(f"{parser.name}: {exc}")
                log.warning("parser %s failed for %s: %s", parser.name, doc.id, exc)
                continue
        parsed.parser = parser.name
        parsed.parser_degraded = parser.name != "native"
        conf = Confidence.of(Signal.LAYOUT, PENALTY[parser.name], f"parser={parser.name}")
        parsed.parse_confidence = conf
        return parsed, conf
    raise ParseError(f"all parsers failed for {doc.id}: {'; '.join(errors)}")

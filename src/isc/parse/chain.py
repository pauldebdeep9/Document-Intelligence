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
from isc.models.document import Block, BlockType, DocType, Document, Page

log = get_logger("parse")

# Discount applied when this parser had to be used. Native text is trusted.
PENALTY = {"native": 1.0, "layout": 0.95, "ocr": 0.80}

# The corpus prints header fields as table cells, not "Label: value" — e.g.
# "PO Number        4522345741        Order Date        16/08/2025", two
# label/value pairs per line, columns separated by runs of 2+ spaces. A block
# counts as KEY_VALUE when most of its non-empty lines split into an even
# number of cells whose label positions look like short capitalised labels.
# Anything else heuristic-detected as a block is PARAGRAPH; table detection is
# P1-04's job, and misreading the item table here would propagate downstream.
_COLUMN_GAP = re.compile(r" {2,}")
_LABEL_CELL = re.compile(r"^[A-Z][A-Za-z0-9]*(?:[ /&-][A-Z][A-Za-z0-9]*)*$")
_KV_LINE_RATIO = 0.6


def _kv_pairs_in_line(line: str) -> int:
    cells = [c for c in _COLUMN_GAP.split(line.strip()) if c]
    if len(cells) < 2 or len(cells) % 2 != 0:
        return 0
    labels = cells[0::2]
    if all(len(lb) < 40 and _LABEL_CELL.match(lb) for lb in labels):
        return len(cells) // 2
    return 0

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
            for block_text in _split_blocks(_extract_page_text(pdf_page)):
                if not title_set and page_number == 1:
                    btype = BlockType.TITLE
                    title_set = True
                elif _is_key_value(block_text):
                    btype = BlockType.KEY_VALUE
                else:
                    btype = BlockType.PARAGRAPH
                blocks.append(Block(
                    id=block_id(doc.id, page_number, reading_order, block_text),
                    type=btype,
                    text=block_text,
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

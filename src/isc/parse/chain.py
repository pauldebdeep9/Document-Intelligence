"""Parser fallback chain with confidence penalties.

Order: native text layer -> layout model -> OCR. Each fallback is *recorded* and
discounts downstream confidence, so a field read off a bad OCR pass can never
present as clean. That linkage is the whole reason the chain is explicit.
"""

from __future__ import annotations

from typing import Protocol

from isc.common.confidence import Confidence, Signal
from isc.common.errors import ParseError
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.models.document import Document

log = get_logger("parse")

# Discount applied when this parser had to be used. Native text is trusted.
PENALTY = {"native": 1.0, "layout": 0.95, "ocr": 0.80}


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
        raise NotImplementedError("first slice: pypdf pages -> Block(PARAGRAPH)")


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


def parse_with_fallback(doc: Document, data: bytes, suffix: str,
                        parsers: list[Parser]) -> tuple[Document, Confidence]:
    errors: list[str] = []
    for parser in parsers:
        if not parser.can_parse(data, suffix):
            continue
        with span("parse.attempt", parser=parser.name, doc=doc.id):
            try:
                parsed = parser.parse(doc, data)
            except (NotImplementedError, Exception) as exc:  # noqa: B014
                errors.append(f"{parser.name}: {exc}")
                log.warning("parser %s failed for %s: %s", parser.name, doc.id, exc)
                continue
        parsed.parser = parser.name
        parsed.parser_degraded = parser.name != "native"
        conf = Confidence.of(Signal.LAYOUT, PENALTY[parser.name], f"parser={parser.name}")
        return parsed, conf
    raise ParseError(f"all parsers failed for {doc.id}: {'; '.join(errors)}")

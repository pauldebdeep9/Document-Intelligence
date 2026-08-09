from __future__ import annotations

import io

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from isc.models.acl import AclSet
from isc.models.document import DocType, Document
from isc.parse.chain import LayoutParser, NativeTextParser, OcrParser, parse_with_fallback

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

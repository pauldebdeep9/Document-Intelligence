from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.errors import PdfReadError
from reportlab.pdfgen import canvas

import isc.pdf as pdf_module
from isc.models import PDFPage
from isc.pdf import extract_pdf_pages


def make_pdf(path: Path, pages: list[str]) -> None:
    pdf = canvas.Canvas(str(path))
    for text in pages:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()


def test_extract_pdf_pages_accepts_string_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "one-page.pdf"
    make_pdf(pdf_path, ["Purchase Order PO-1001"])

    pages = extract_pdf_pages(str(pdf_path))

    assert len(pages) == 1
    assert isinstance(pages[0], PDFPage)
    assert pages[0].page_number == 1
    assert "Purchase Order PO-1001" in pages[0].text


def test_extract_pdf_pages_accepts_path_object(tmp_path: Path) -> None:
    pdf_path = tmp_path / "path-object.pdf"
    make_pdf(pdf_path, ["Supplier ABC Components"])

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Supplier ABC Components" in pages[0].text


def test_extract_pdf_pages_preserves_page_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "three-pages.pdf"
    expected_text = [
        "Purchase Order PO-1001",
        "Supplier ABC Components",
        "Payment Terms Net 30",
    ]
    make_pdf(pdf_path, expected_text)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 3
    assert all(isinstance(page, PDFPage) for page in pages)
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert all(expected in page.text for expected, page in zip(expected_text, pages, strict=True))


def test_extract_pdf_pages_preserves_blank_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank-middle-page.pdf"
    make_pdf(pdf_path, ["Purchase Order PO-1001", "", "Payment Terms Net 30"])

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 3
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert "Purchase Order PO-1001" in pages[0].text
    assert pages[1].text.strip() == ""
    assert "Payment Terms Net 30" in pages[2].text


def test_extract_pdf_pages_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path / "missing.pdf")


def test_extract_pdf_pages_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PDF path is not a regular file"):
        extract_pdf_pages(tmp_path)


def test_extract_pdf_pages_wraps_malformed_pdf_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "malformed.pdf"
    pdf_path.write_bytes(b"this is not a PDF")

    with pytest.raises(ValueError, match="PDF could not be read") as exc_info:
        extract_pdf_pages(pdf_path)

    assert isinstance(exc_info.value.__cause__, PdfReadError)


def test_extract_pdf_pages_rejects_encrypted_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    with pdf_path.open("wb") as output:
        writer.write(output)

    with pytest.raises(ValueError, match="Encrypted PDFs are not supported"):
        extract_pdf_pages(pdf_path)


def test_extract_pdf_pages_rejects_all_blank_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    make_pdf(pdf_path, ["", ""])

    with pytest.raises(
        ValueError,
        match="No native PDF text found; OCR is not supported by this PoC",
    ):
        extract_pdf_pages(pdf_path)


def test_extract_pdf_pages_rejects_zero_page_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "zero-pages.pdf"
    pdf_path.write_bytes(b"fake PDF used with a fake reader")
    reader = SimpleNamespace(is_encrypted=False, pages=[])
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _path: reader)

    with pytest.raises(ValueError, match="No native PDF text found"):
        extract_pdf_pages(pdf_path)


def test_extract_pdf_pages_falls_back_without_stripping_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "layout-fallback.pdf"
    pdf_path.write_bytes(b"fake PDF used with a fake reader")

    class FakePage:
        def __init__(self) -> None:
            self.extraction_modes: list[str | None] = []

        def extract_text(self, extraction_mode: str | None = None) -> str:
            self.extraction_modes.append(extraction_mode)
            if extraction_mode == "layout":
                raise ValueError("layout extraction failed")
            return "  Payment Terms Net 30  \n"

    page = FakePage()
    reader = SimpleNamespace(is_encrypted=False, pages=[page])
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _path: reader)

    pages = extract_pdf_pages(pdf_path)

    assert page.extraction_modes == ["layout", None]
    assert pages == [PDFPage(page_number=1, text="  Payment Terms Net 30  \n")]

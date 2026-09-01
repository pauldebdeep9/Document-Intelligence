"""Native-text PDF extraction for the proof of concept."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from isc.models import PDFPage


def extract_pdf_pages(path: str | Path) -> list[PDFPage]:
    """Extract native text from every PDF page in document order."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a regular file: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as exc:
        raise ValueError(f"PDF could not be read: {pdf_path}") from exc

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")

    pages: list[PDFPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout")
        except (TypeError, ValueError, NotImplementedError, PdfReadError):
            text = page.extract_text()

        pages.append(PDFPage(page_number=page_number, text=text or ""))

    if not any(page.text.strip() for page in pages):
        raise ValueError("No native PDF text found; OCR is not supported by this PoC")

    return pages

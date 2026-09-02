"""Deterministic reportlab renderer and CLI for the synthetic Purchase Order gold corpus."""

import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from evals.corpus.specs import DOCUMENT_SPECS, DocumentSpec

PDF_DIR = Path("data/gold/pdfs")
CORPUS_PATH = Path("data/gold/corpus.json")

_LEFT_MARGIN = 72
_TOP_Y = 740
_LINE_HEIGHT = 14


def render_pdf(spec: DocumentSpec, output_path: Path) -> None:
    """Draw spec.pages verbatim onto a byte-reproducible PDF at output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=letter, invariant=1)
    for page_lines in spec.pages:
        y = _TOP_Y
        for line in page_lines:
            pdf.drawString(_LEFT_MARGIN, y, line)
            y -= _LINE_HEIGHT
        pdf.showPage()
    pdf.save()


def write_corpus_json(specs: list[DocumentSpec], output_path: Path) -> None:
    """Write specs as a deterministic JSON array, in stable declared field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [spec.model_dump(mode="json") for spec in specs]
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_corpus(pdf_dir: Path = PDF_DIR, corpus_path: Path = CORPUS_PATH) -> None:
    """Render every spec's PDF and write the corpus manifest."""
    for spec in DOCUMENT_SPECS:
        render_pdf(spec, pdf_dir / f"{spec.doc_id}.pdf")
    write_corpus_json(DOCUMENT_SPECS, corpus_path)


def main() -> None:
    """Generate the full gold corpus at its default locations."""
    generate_corpus(pdf_dir=PDF_DIR, corpus_path=CORPUS_PATH)
    print(f"Wrote {len(DOCUMENT_SPECS)} PDFs to {PDF_DIR} and corpus to {CORPUS_PATH}")


if __name__ == "__main__":
    main()

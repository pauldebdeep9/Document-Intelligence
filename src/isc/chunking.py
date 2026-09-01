"""Simple page-local text chunking for the proof of concept."""

from isc.models import Chunk, PDFPage


def chunk_pages(
    pages: list[PDFPage],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Split each page into deterministic overlapping character windows."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size")

    page_numbers = [page.page_number for page in pages]
    if len(page_numbers) != len(set(page_numbers)):
        raise ValueError("Page numbers must be unique")

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for page in pages:
        if not page.text.strip():
            continue

        chunk_number = 1
        for start in range(0, len(page.text), step):
            text = page.text[start : start + chunk_size]
            if text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"page-{page.page_number:03d}-chunk-{chunk_number:03d}",
                        page_number=page.page_number,
                        text=text,
                    )
                )
                chunk_number += 1
            if start + chunk_size >= len(page.text):
                break

    return chunks

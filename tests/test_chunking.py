import pytest

from isc.chunking import chunk_pages
from isc.models import Chunk, PDFPage


def test_short_page_produces_one_complete_chunk() -> None:
    page = PDFPage(page_number=1, text="Purchase Order PO-1001")

    chunks = chunk_pages([page], doc_id="doc-1")

    assert chunks == [
        Chunk(
            doc_id="doc-1",
            chunk_id="doc-1:page-001-chunk-001",
            page_number=1,
            text="Purchase Order PO-1001",
        )
    ]


def test_long_page_produces_expected_character_windows() -> None:
    text = "A" * 1000 + "B" * 1000 + "C" * 500

    chunks = chunk_pages([PDFPage(page_number=1, text=text)], doc_id="doc-1")

    assert len(chunks) == 3
    assert [chunk.text for chunk in chunks] == [
        text[0:1200],
        text[1000:2200],
        text[2000:3200],
    ]


def test_default_windows_have_two_hundred_character_overlap() -> None:
    text = "A" * 1000 + "B" * 1000 + "C" * 500

    chunks = chunk_pages([PDFPage(page_number=1, text=text)], doc_id="doc-1")

    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert chunks[1].text[-200:] == chunks[2].text[:200]


def test_every_chunk_is_an_exact_source_substring() -> None:
    text = "Header\n" + "line item\n" * 300

    chunks = chunk_pages([PDFPage(page_number=4, text=text)], doc_id="doc-1")

    assert all(chunk.text in text for chunk in chunks)


def test_multiple_pages_preserve_input_order_and_restart_chunk_numbers() -> None:
    pages = [
        PDFPage(page_number=5, text="Page five source"),
        PDFPage(page_number=2, text="Page two source"),
    ]

    chunks = chunk_pages(pages, doc_id="doc-1")

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1:page-005-chunk-001",
        "doc-1:page-002-chunk-001",
    ]
    assert [chunk.page_number for chunk in chunks] == [5, 2]
    assert chunks[0].text == pages[0].text
    assert chunks[1].text == pages[1].text


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_blank_page_produces_no_chunks(text: str) -> None:
    assert chunk_pages([PDFPage(page_number=1, text=text)], doc_id="doc-1") == []


def test_blank_page_between_valid_pages_preserves_original_page_numbers() -> None:
    pages = [
        PDFPage(page_number=1, text="First source"),
        PDFPage(page_number=2, text=""),
        PDFPage(page_number=3, text="Third source"),
    ]

    chunks = chunk_pages(pages, doc_id="doc-1")

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1:page-001-chunk-001",
        "doc-1:page-003-chunk-001",
    ]


def test_meaningful_page_preserves_leading_and_trailing_whitespace() -> None:
    text = "  Purchase Order PO-1001  \n"

    chunks = chunk_pages([PDFPage(page_number=1, text=text)], doc_id="doc-1")

    assert chunks[0].text == text


def test_empty_page_list_returns_empty_chunk_list() -> None:
    assert chunk_pages([], doc_id="doc-1") == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_raises(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="Chunk size must be greater than zero"):
        chunk_pages([], doc_id="doc-1", chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 1200, 1201])
def test_invalid_overlap_raises(overlap: int) -> None:
    with pytest.raises(ValueError, match="Overlap must be non-negative"):
        chunk_pages([], doc_id="doc-1", overlap=overlap)


def test_duplicate_page_numbers_raise() -> None:
    pages = [
        PDFPage(page_number=2, text="First source"),
        PDFPage(page_number=2, text="Second source"),
    ]

    with pytest.raises(ValueError, match="Page numbers must be unique"):
        chunk_pages(pages, doc_id="doc-1")


def test_chunking_is_deterministic() -> None:
    pages = [PDFPage(page_number=1, text="A" * 2500)]

    assert chunk_pages(pages, doc_id="doc-1") == chunk_pages(pages, doc_id="doc-1")


def test_custom_size_and_overlap_control_exact_windows() -> None:
    text = "0123456789ABCDEFGHIJ"

    chunks = chunk_pages(
        [PDFPage(page_number=1, text=text)],
        doc_id="doc-1",
        chunk_size=10,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == [
        text[0:10],
        text[7:17],
        text[14:24],
    ]
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1:page-001-chunk-001",
        "doc-1:page-001-chunk-002",
        "doc-1:page-001-chunk-003",
    ]
    assert len(chunks[-1].text) == 6


@pytest.mark.parametrize(
    ("length", "expected_starts"),
    [
        (10, [0]),
        (11, [0, 7]),
        (7, [0]),
        (18, [0, 7, 14]),
    ],
)
def test_exact_boundary_lengths_follow_ordinary_slicing(
    length: int,
    expected_starts: list[int],
) -> None:
    text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:length]

    chunks = chunk_pages(
        [PDFPage(page_number=1, text=text)],
        doc_id="doc-1",
        chunk_size=10,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == [
        text[start : start + 10] for start in expected_starts
    ]


def test_whitespace_candidate_windows_are_skipped_without_id_gaps() -> None:
    text = "ABCDE" + "     " + "FGHIJ"

    chunks = chunk_pages(
        [PDFPage(page_number=1, text=text)],
        doc_id="doc-1",
        chunk_size=5,
        overlap=0,
    )

    assert [chunk.text for chunk in chunks] == ["ABCDE", "FGHIJ"]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1:page-001-chunk-001",
        "doc-1:page-001-chunk-002",
    ]


def test_newline_structure_is_preserved_exactly() -> None:
    text = "PO NUMBER: PO-1001\n\nSupplier: ABC\nPayment Terms: Net 30\n"

    chunks = chunk_pages([PDFPage(page_number=1, text=text)], doc_id="doc-1")

    assert chunks[0].text == text


def test_unicode_text_is_sliced_as_python_characters() -> None:
    text = "Supplier: Müller Components\nCurrency: €\n"

    chunks = chunk_pages(
        [PDFPage(page_number=1, text=text)],
        doc_id="doc-1",
        chunk_size=20,
        overlap=4,
    )

    assert [chunk.text for chunk in chunks] == [
        text[0:20],
        text[16:36],
        text[32:52],
    ]


def test_large_page_uses_expected_starts_and_unique_ids() -> None:
    text = "".join(chr(33 + index % 90) for index in range(5000))
    chunk_size = 257
    overlap = 31
    step = chunk_size - overlap
    expected_count = 1 + (len(text) - chunk_size + step - 1) // step
    expected_starts = [index * step for index in range(expected_count)]

    chunks = chunk_pages(
        [PDFPage(page_number=8, text=text)],
        doc_id="doc-1",
        chunk_size=chunk_size,
        overlap=overlap,
    )

    assert len(chunks) == expected_count
    assert len({chunk.chunk_id for chunk in chunks}) == expected_count
    assert all(chunk.page_number == 8 for chunk in chunks)
    assert [chunk.text for chunk in chunks] == [
        text[start : start + chunk_size] for start in expected_starts
    ]


def test_non_sequential_pages_preserve_order_and_unique_ids() -> None:
    pages = [
        PDFPage(page_number=10, text="A" * 15),
        PDFPage(page_number=3, text="B" * 15),
    ]

    chunks = chunk_pages(pages, doc_id="doc-1", chunk_size=10, overlap=2)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert chunk_ids == [
        "doc-1:page-010-chunk-001",
        "doc-1:page-010-chunk-002",
        "doc-1:page-003-chunk-001",
        "doc-1:page-003-chunk-002",
    ]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_large_page_number_uses_at_least_three_digits() -> None:
    chunks = chunk_pages([PDFPage(page_number=1234, text="Purchase Order")], doc_id="doc-1")

    assert chunks[0].chunk_id == "doc-1:page-1234-chunk-001"


@pytest.mark.parametrize(
    ("length", "chunk_size", "overlap"),
    [
        (1, 1, 0),
        (10, 4, 0),
        (11, 4, 1),
        (37, 10, 3),
        (100, 17, 5),
    ],
)
def test_chunk_invariants_across_deterministic_sizes(
    length: int,
    chunk_size: int,
    overlap: int,
) -> None:
    text = "".join(chr(65 + index % 26) for index in range(length))

    chunks = chunk_pages(
        [PDFPage(page_number=7, text=text)],
        doc_id="doc-1",
        chunk_size=chunk_size,
        overlap=overlap,
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert all(chunk.text for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.text in text for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert len(chunk_ids) == len(set(chunk_ids))


def test_two_documents_chunked_independently_produce_disjoint_chunk_id_sets() -> None:
    # Same page content, same page number, same chunk_size/overlap -> under the old
    # doc-unqualified chunk_id format this produced identical IDs for both documents.
    page = PDFPage(page_number=1, text="Purchase Order PO-1001")

    chunks_a = chunk_pages([page], doc_id="po-004")
    chunks_b = chunk_pages([page], doc_id="po-005")

    ids_a = {chunk.chunk_id for chunk in chunks_a}
    ids_b = {chunk.chunk_id for chunk in chunks_b}

    assert ids_a.isdisjoint(ids_b)
    assert all(chunk.doc_id == "po-004" for chunk in chunks_a)
    assert all(chunk.doc_id == "po-005" for chunk in chunks_b)

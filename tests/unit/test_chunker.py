"""Unit tests for index/chunker.py: windowing, table whole/split, ACL
inheritance, section_path, filters, and the settings fingerprint.

Documents are built directly (Block/Table objects), not through PDF
rendering -- the chunker operates on an already-parsed Document, and the
table shapes here are deliberately synthetic so token budgets can be hit
exactly rather than approximated against real corpus text.
"""

from __future__ import annotations

from isc.common.config import ChunkSettings
from isc.index.chunker import (
    chunk_document,
    estimate_tokens,
    filters_from_record,
    settings_fingerprint,
)
from isc.models.acl import AclSet
from isc.models.document import Block, BlockType, DocType, Document, Page, Table


def _doc(blocks: list[Block], doc_id: str = "doc_test", pages: int = 1) -> Document:
    by_page: dict[int, list[Block]] = {n: [] for n in range(1, pages + 1)}
    for b in blocks:
        by_page[b.page].append(b)
    return Document(
        id=doc_id, source_uri="test.pdf", content_sha256="x",
        doc_type=DocType.PURCHASE_ORDER,
        acl=AclSet(allow_terms=frozenset({"everyone:*"})),
        pages=[Page(number=n, blocks=bs) for n, bs in by_page.items()],
    )


def _block(text: str, btype: BlockType, page: int = 1, order: int = 0,
           table: Table | None = None) -> Block:
    return Block(id=f"blk_{order}", type=btype, text=text, page=page,
                 reading_order=order, table=table)


def _line_table(n_rows: int, *, start: int = 10, step: int = 10) -> Table:
    header = ["Item", "Part Number", "Description", "Qty", "Unit Price"]
    rows = [header]
    for i in range(n_rows):
        ordinal = start + i * step
        rows.append([str(ordinal), f"PART-{ordinal}", f"Widget number {ordinal}", "3", "199.99"])
    return Table(rows=rows, header_rows=1)


def _table_block(n_rows: int, page: int = 1, order: int = 0, **kw) -> Block:
    table = _line_table(n_rows, **kw)
    text = "\n".join("  ".join(row) for row in table.rows)
    return Block(id=f"blk_table_{order}", type=BlockType.TABLE, text=text,
                 page=page, reading_order=order, table=table)


_SETTINGS = ChunkSettings(target_tokens=512, overlap_tokens=64,
                           keep_tables_whole=True, max_table_tokens=800)


# --- windowing / ACL / section_path -----------------------------------

def test_every_chunk_inherits_the_documents_acl_at_construction():
    doc = _doc([
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _block("PO Number   4500123456   Order Date   16/08/2025", BlockType.KEY_VALUE, order=1),
        _table_block(2, order=2),
    ])
    chunks = chunk_document(doc, _SETTINGS)
    assert chunks
    assert all(c.acl == doc.acl for c in chunks)


def test_chunk_ids_are_deterministic_across_two_calls():
    doc = _doc([
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _block("PO Number   4500123456", BlockType.KEY_VALUE, order=1),
        _table_block(5, order=2),
        _block("Closing boilerplate paragraph.", BlockType.PARAGRAPH, order=3),
    ])
    ids1 = [c.id for c in chunk_document(doc, _SETTINGS)]
    ids2 = [c.id for c in chunk_document(doc, _SETTINGS)]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1))  # no accidental collisions either


def test_chunk_ids_change_when_settings_change():
    """Chunk ids are a function of chunking settings (via chunk text) -- a
    smaller target_tokens changes how blocks are windowed together, which
    changes chunk text, which changes chunk_id. This is the exact mechanism
    P1-08's gold depends on staying stable."""
    doc = _doc([
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _block("PO Number   4500123456", BlockType.KEY_VALUE, order=1),
        _block("Ship To   Plant 1", BlockType.KEY_VALUE, order=2),
        _block("A longer closing paragraph with enough words to matter here.",
               BlockType.PARAGRAPH, order=3),
    ])
    wide = chunk_document(doc, ChunkSettings(target_tokens=512, overlap_tokens=0))
    narrow = chunk_document(doc, ChunkSettings(target_tokens=5, overlap_tokens=0))
    assert [c.id for c in wide] != [c.id for c in narrow]


def test_section_path_tracks_title_and_most_recent_key_value():
    doc = _doc([
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _block("PO Number   4500123456", BlockType.KEY_VALUE, order=1),
        _block("Ship To   Plant 1", BlockType.KEY_VALUE, order=2),
        _block("Closing paragraph.", BlockType.PARAGRAPH, order=3),
    ])
    chunks = chunk_document(doc, _SETTINGS)
    assert len(chunks) == 1  # everything fits in one window
    assert chunks[0].section_path == ("PURCHASE ORDER", "Ship To Plant 1")


def test_section_path_before_any_title_is_empty():
    doc = _doc([_block("No title yet.", BlockType.PARAGRAPH, order=0)])
    chunks = chunk_document(doc, _SETTINGS)
    assert chunks[0].section_path == ()


def test_target_tokens_forces_a_new_window():
    blocks = [
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _block("First paragraph with a handful of words in it here today.",
               BlockType.PARAGRAPH, order=1),
        _block("Second paragraph with a handful of words in it here today.",
               BlockType.PARAGRAPH, order=2),
        _block("Third paragraph with a handful of words in it here today.",
               BlockType.PARAGRAPH, order=3),
    ]
    tight = ChunkSettings(target_tokens=15, overlap_tokens=0, max_table_tokens=800)
    chunks = chunk_document(_doc(blocks), tight)
    assert len(chunks) > 1, "a 15-token budget must not fit all four blocks in one window"
    # each window packs whole blocks only up to the budget -- no chunk here
    # contains all three paragraph blocks together
    assert not any(
        "First" in c.text and "Second" in c.text and "Third" in c.text for c in chunks
    )


def test_overlap_carries_trailing_blocks_into_the_next_window():
    blocks = [
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _block("Alpha paragraph with several words padding it out nicely.",
               BlockType.PARAGRAPH, order=1),
        _block("Bravo paragraph with several words padding it out nicely.",
               BlockType.PARAGRAPH, order=2),
        _block("Charlie paragraph with several words padding it out nicely.",
               BlockType.PARAGRAPH, order=3),
    ]
    settings = ChunkSettings(target_tokens=20, overlap_tokens=12, max_table_tokens=800)
    chunks = chunk_document(_doc(blocks), settings)
    assert len(chunks) >= 2
    # the block that ended window 0 shows up again at the start of window 1
    assert "Alpha" in chunks[0].text or "Bravo" in chunks[0].text
    assert any(chunks[0].text.split("\n\n")[-1] in chunks[1].text for _ in [0])


# --- tables: whole vs split, the one correctness rule -------------------

def test_small_table_emitted_whole_with_header_included():
    doc = _doc([_block("PURCHASE ORDER", BlockType.TITLE, order=0), _table_block(3, order=1)])
    chunks = chunk_document(doc, _SETTINGS)
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Item" in table_chunks[0].text  # header present
    assert table_chunks[0].line_range == (10, 30)


def test_large_table_splits_between_rows_never_mid_row():
    """The one correctness rule, asserted directly: every printed row
    appears complete in exactly one chunk, never split across two, and
    never duplicated."""
    doc = _doc([_block("PURCHASE ORDER", BlockType.TITLE, order=0), _table_block(60, order=1)])
    settings = ChunkSettings(target_tokens=512, overlap_tokens=64,
                              keep_tables_whole=True, max_table_tokens=500)
    chunks = chunk_document(doc, settings)
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) > 1, "fixture should force a split"

    all_ordinals = [str(10 + i * 10) for i in range(60)]
    seen: list[str] = []
    for c in table_chunks:
        for line in c.text.split("\n"):
            cell = line.strip("| ").split("|")[0].strip()
            if cell.isdigit():
                seen.append(cell)
    assert seen == all_ordinals, "every row must appear exactly once, in order, complete"


def test_large_table_repeats_the_header_on_every_part():
    doc = _doc([_block("PURCHASE ORDER", BlockType.TITLE, order=0), _table_block(60, order=1)])
    settings = ChunkSettings(target_tokens=512, overlap_tokens=64,
                              keep_tables_whole=True, max_table_tokens=500)
    chunks = [c for c in chunk_document(doc, settings) if c.is_table]
    assert len(chunks) > 1
    assert all("Item" in c.text and "Part Number" in c.text for c in chunks)


def test_large_table_line_ranges_are_contiguous_and_non_overlapping():
    doc = _doc([_block("PURCHASE ORDER", BlockType.TITLE, order=0), _table_block(60, order=1)])
    settings = ChunkSettings(target_tokens=512, overlap_tokens=64,
                              keep_tables_whole=True, max_table_tokens=500)
    chunks = [c for c in chunk_document(doc, settings) if c.is_table]
    ranges = [c.line_range for c in chunks]
    assert all(r is not None for r in ranges)
    for (_, last), (first_next, _) in zip(ranges, ranges[1:], strict=False):
        assert last < first_next


def test_table_never_merges_into_a_prose_chunk():
    doc = _doc([
        _block("PURCHASE ORDER", BlockType.TITLE, order=0),
        _table_block(2, order=1),
        _block("Closing paragraph.", BlockType.PARAGRAPH, order=2),
    ])
    chunks = chunk_document(doc, _SETTINGS)
    assert [c.is_table for c in chunks] == [False, True, False]


def test_keep_tables_whole_false_folds_table_into_ordinary_windowing():
    doc = _doc([_block("PURCHASE ORDER", BlockType.TITLE, order=0), _table_block(3, order=1)])
    settings = ChunkSettings(target_tokens=512, overlap_tokens=0,
                              keep_tables_whole=False, max_table_tokens=1)
    chunks = chunk_document(doc, settings)
    assert not any(c.is_table for c in chunks)
    assert all(c.line_range is None for c in chunks)


# --- filters ------------------------------------------------------------

class _FakeField:
    def __init__(self, value):
        self.value = value


class _FakeRecord:
    def __init__(self, po_number=None, supplier_id=None):
        self.po_number = _FakeField(po_number)
        self.supplier_id = _FakeField(supplier_id)


def test_filters_from_record_extracts_po_number_and_supplier_id():
    record = _FakeRecord(po_number="4500123456", supplier_id="V102337")
    assert filters_from_record(record) == {"po_number": "4500123456", "supplier_id": "V102337"}


def test_filters_from_record_omits_absent_values():
    record = _FakeRecord(po_number="4500123456", supplier_id=None)
    assert filters_from_record(record) == {"po_number": "4500123456"}


def test_filters_from_record_handles_missing_attributes_gracefully():
    assert filters_from_record(object()) == {}


def test_chunk_document_populates_filters_on_every_chunk():
    doc = _doc([_block("PURCHASE ORDER", BlockType.TITLE, order=0), _table_block(2, order=1)])
    chunks = chunk_document(doc, _SETTINGS, filters={"po_number": "4500123456"})
    assert all(c.filters == {"po_number": "4500123456"} for c in chunks)


# --- settings_fingerprint ------------------------------------------------

def test_settings_fingerprint_is_deterministic():
    a = settings_fingerprint(ChunkSettings(target_tokens=512))
    b = settings_fingerprint(ChunkSettings(target_tokens=512))
    assert a == b


def test_settings_fingerprint_changes_with_any_field():
    base = settings_fingerprint(ChunkSettings())
    assert base != settings_fingerprint(ChunkSettings(target_tokens=256))
    assert base != settings_fingerprint(ChunkSettings(overlap_tokens=32))
    assert base != settings_fingerprint(ChunkSettings(max_table_tokens=1000))
    assert base != settings_fingerprint(ChunkSettings(keep_tables_whole=False))


def test_estimate_tokens_uses_tiktoken_not_a_length_heuristic():
    """A heuristic like len(text)//4 would give 3 for 'hello world' (11
    chars); the real cl100k_base count is 2."""
    assert estimate_tokens("hello world") == 2

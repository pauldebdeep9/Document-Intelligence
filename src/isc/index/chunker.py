"""Layout-aware and table-aware chunking, plus ACL projection.

Two rules that matter more than the token budget:
  * A table is never split mid-row. A half table is worse than no table: it
    retrieves confidently and answers wrongly.
  * Every chunk inherits its document's AclSet at construction. There is no
    'attach permissions afterwards' step, because that step gets skipped.

Design follows what parse/ actually produces for this corpus (measured, not
assumed against the enum): TITLE once, KEY_VALUE and PARAGRAPH repeatedly,
TABLE for the reconstructed line-item table (see parse/chain.py, P1-04 step
1). No HEADING is ever emitted, so section_path breadcrumbs are built from
TITLE and KEY_VALUE block text -- the only two BlockTypes that read as
orientation rather than content in this parser's output.

Chunk ids are a function of chunking settings (via chunk text, transitively
-- see common/ids.py's chunk_id()). P1-08's retrieval gold will be built
against a specific set of them; settings_fingerprint() exists so a later
config change can be detected and made to fail loudly instead of silently
scoring against the wrong boundaries. See docs/adr/0007.
"""

from __future__ import annotations

from typing import Any

import tiktoken

from isc.common.config import ChunkSettings
from isc.common.ids import cache_key, chunk_id
from isc.models.chunk import Chunk
from isc.models.document import Block, BlockType, Document, Table

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Breadcrumb entries are truncated to keep citation_label() readable -- a raw
# KEY_VALUE block's text ("PO Number   4522345741   Order Date   16/08/2025")
# is meant to be read as data, not as a heading, so only enough of it to
# orient a reader survives into section_path.
_BREADCRUMB_MAX_CHARS = 60


def estimate_tokens(text: str) -> int:
    """tiktoken, not len(text)//4: chunk boundaries feed retrieval quality
    and P1-08's gold chunk ids directly, and a heuristic that drifts from
    the real count by even 20% is not a foundation to build recall numbers
    on. cl100k_base is the encoding OpenAI's current embedding models use;
    a fixed encoding also keeps token counts (and therefore chunk_ids)
    stable across a provider/model swap that does not change the encoding,
    which is the whole point of measuring rather than guessing."""
    return len(_ENCODING.encode(text))


def settings_fingerprint(settings: ChunkSettings) -> str:
    """Deterministic id for one chunking configuration -- a pure function of
    its fields, per common/ids.py's own discipline. Recorded into the index
    artifact at chunk time (see storage/local_vector.py's settings_fingerprint
    parameter); P1-08's gold is expected to record the same value when it is
    built, and a mismatch between the two at eval time must be checked and
    must fail loudly -- changing target_tokens/overlap_tokens/max_table_tokens
    after that gold exists silently invalidates every recall number without
    raising anything, because chunk_id() already embeds ordinal and text, not
    settings, so two different settings can legally produce the same set of
    ids for a small document and the same ids can legally mean different
    boundaries for a large one."""
    return cache_key("chunk_settings", settings.model_dump())


def filters_from_record(record: Any) -> dict[str, str]:
    """po_number and supplier_id, when the record has them and they are
    present -- the two fields retrieve/ needs for exact-match metadata
    filtering. Duck-typed rather than importing ExtractionRecord: chunker.py
    stays a pure function of Document + settings + already-resolved filter
    values, and does not need to know which doc types extract/ supports."""
    filters: dict[str, str] = {}
    for name in ("po_number", "supplier_id"):
        field = getattr(record, name, None)
        value = getattr(field, "value", None)
        if value:
            filters[name] = str(value)
    return filters


def chunk_document(
    doc: Document, settings: ChunkSettings, filters: dict[str, str] | None = None,
) -> list[Chunk]:
    """Fixed-size windows over doc.blocks(), tables handled separately.

    A TABLE block always flushes whatever prose window was accumulating --
    a table never merges into a prose chunk in either direction, and prose
    overlap never carries across a table boundary (the header-repeat on a
    split table is that boundary's own continuity mechanism, not overlap).
    """
    filters = filters or {}
    chunks: list[Chunk] = []
    ordinal = 0
    window: list[Block] = []
    window_tokens = 0
    title: str | None = None
    kv: str | None = None

    def section_path() -> tuple[str, ...]:
        return tuple(_truncate(s) for s in (title, kv) if s)

    def flush() -> None:
        nonlocal ordinal, window, window_tokens
        if not window:
            return
        chunks.append(_prose_chunk(doc, window, ordinal, section_path(), filters))
        ordinal += 1
        window = _overlap_tail(window, settings.overlap_tokens)
        window_tokens = sum(estimate_tokens(b.text) for b in window)

    for block in doc.blocks():
        if block.type == BlockType.TABLE:
            flush()
            window, window_tokens = [], 0
            if block.table is not None and settings.keep_tables_whole:
                for part in _table_parts(block.table, settings.max_table_tokens):
                    chunks.append(
                        _table_chunk(doc, block, part, ordinal, section_path(), filters)
                    )
                    ordinal += 1
                continue
            # keep_tables_whole=False: fall through and treat the table
            # block like any other block of text, subject to ordinary
            # windowing. Still never splits mid-row -- windowing never
            # splits within a single block's own text -- it just stops
            # treating the table as a structurally distinct unit.

        block_tokens = estimate_tokens(block.text)
        if window and window_tokens + block_tokens > settings.target_tokens:
            flush()
        window.append(block)
        window_tokens += block_tokens
        if block.type == BlockType.TITLE:
            title = block.text
        elif block.type == BlockType.KEY_VALUE:
            kv = block.text

    flush()
    return chunks


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _BREADCRUMB_MAX_CHARS else text[: _BREADCRUMB_MAX_CHARS - 1] + "…"


def _overlap_tail(blocks: list[Block], overlap_tokens: int) -> list[Block]:
    """Trailing whole blocks totalling ~overlap_tokens, carried into the next
    window. Never splits a block's own text -- overlap is block-granular,
    same as windowing itself."""
    if overlap_tokens <= 0:
        return []
    tail: list[Block] = []
    total = 0
    for b in reversed(blocks):
        t = estimate_tokens(b.text)
        if tail and total + t > overlap_tokens:
            break
        tail.insert(0, b)
        total += t
    return tail


def _prose_chunk(
    doc: Document, blocks: list[Block], ordinal: int,
    section: tuple[str, ...], filters: dict[str, str],
) -> Chunk:
    text = "\n\n".join(b.text for b in blocks if b.text)
    cid = chunk_id(doc.id, ordinal, text)
    return Chunk(
        id=cid, document_id=doc.id, ordinal=ordinal, text=text, acl=doc.acl,
        doc_type=doc.doc_type,
        page_start=min(b.page for b in blocks), page_end=max(b.page for b in blocks),
        section_path=section, is_table=False,
        token_count=estimate_tokens(text), filters=dict(filters),
    )


def _table_parts(table: Table, max_table_tokens: int) -> list[Table]:
    """Whole under max_table_tokens; otherwise split between whole rows,
    header repeated on every part. Never splits a row -- the header and each
    data row are the only units this ever moves as a whole."""
    header = table.rows[: table.header_rows]
    data_rows = table.rows[table.header_rows :]
    header_tokens = estimate_tokens(_rows_text(header))

    if estimate_tokens(_rows_text(table.rows)) <= max_table_tokens:
        return [table]

    parts: list[Table] = []
    current = list(header)
    current_tokens = header_tokens
    for row in data_rows:
        row_tokens = estimate_tokens(_rows_text([row]))
        if len(current) > len(header) and current_tokens + row_tokens > max_table_tokens:
            parts.append(Table(rows=current, header_rows=table.header_rows,
                                caption=table.caption, ocr_confidence=table.ocr_confidence))
            current = list(header)
            current_tokens = header_tokens
        current.append(row)
        current_tokens += row_tokens
    if len(current) > len(header):
        parts.append(Table(rows=current, header_rows=table.header_rows,
                            caption=table.caption, ocr_confidence=table.ocr_confidence))
    return parts


def _rows_text(rows: list[list[str]]) -> str:
    return "\n".join("  ".join(row) for row in rows)


def _table_chunk(
    doc: Document, block: Block, table: Table, ordinal: int,
    section: tuple[str, ...], filters: dict[str, str],
) -> Chunk:
    text = table.to_markdown()
    line_range = _line_range(table)
    cid = chunk_id(doc.id, ordinal, text)
    return Chunk(
        id=cid, document_id=doc.id, ordinal=ordinal, text=text, acl=doc.acl,
        doc_type=doc.doc_type, page_start=block.page, page_end=block.page,
        section_path=section, is_table=True, line_range=line_range,
        token_count=estimate_tokens(text), filters=dict(filters),
    )


def _line_range(table: Table) -> tuple[int, int] | None:
    """First and last data row's leading column, when it parses as an
    integer -- the reconstructed table's ordinal column always lands there
    (see parse/chain.py's _reconstruct_table()), but this stays defensive
    for any Table that did not come from that path."""
    data_rows = table.rows[table.header_rows :]
    if not data_rows:
        return None
    try:
        first = int(data_rows[0][0])
        last = int(data_rows[-1][0])
    except (ValueError, IndexError):
        return None
    return (first, last)

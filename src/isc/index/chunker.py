"""Layout-aware and table-aware chunking, plus ACL projection.

Two rules that matter more than the token budget:
  * A table is never split mid-row. A half table is worse than no table: it
    retrieves confidently and answers wrongly.
  * Every chunk inherits its document's AclSet at construction. There is no
    'attach permissions afterwards' step, because that step gets skipped.
"""

from __future__ import annotations

from isc.common.config import ChunkSettings
from isc.common.ids import chunk_id
from isc.models.chunk import Chunk
from isc.models.document import Document


def chunk_document(doc: Document, settings: ChunkSettings) -> list[Chunk]:
    raise NotImplementedError(
        "first slice: fixed-size windows over doc.blocks() with heading breadcrumbs; "
        "week 2: emit table blocks whole and prepend the header row to each"
    )


def estimate_tokens(text: str) -> int:
    """Cheap heuristic; swap for tiktoken when cost accounting needs precision."""
    return max(1, len(text) // 4)

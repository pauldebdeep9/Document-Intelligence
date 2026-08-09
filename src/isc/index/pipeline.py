"""Stage 4: Document -> Chunks -> embeddings -> vector store."""

from __future__ import annotations

from isc.common.config import Settings
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.index.chunker import chunk_document
from isc.llm.ports import EmbeddingModel
from isc.models.document import Document
from isc.storage.local_vector import LocalVectorStore

log = get_logger("index")


def run(docs: list[Document], embedder: EmbeddingModel,
        store: LocalVectorStore, settings: Settings) -> int:
    total = 0
    for doc in docs:
        with span("index.document", doc=doc.id):
            chunks = chunk_document(doc, settings.chunk)
            if not chunks:
                continue
            vectors = embedder.embed([c.text for c in chunks])
            store.add(chunks, vectors)
            total += len(chunks)
    store.save()
    log.info("indexed %d chunks", total)
    return total

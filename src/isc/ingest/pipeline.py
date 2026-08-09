"""Stage 1: source -> blob + document stub. Dedupe by content hash."""

from __future__ import annotations

from pathlib import Path

from isc.common.ids import content_hash, doc_id
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.ingest.local_source import LocalDirectorySource
from isc.models.document import Document
from isc.storage.local_blob import LocalBlobStore
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("ingest")


def run(source_dir: Path, blobs: LocalBlobStore, docs: SqliteDocStore) -> list[str]:
    ingested: list[str] = []
    source = LocalDirectorySource(source_dir)
    for item in source.items():
        with span("ingest.item", uri=item.uri):
            sha = content_hash(item.data)
            if docs.seen_hash(sha):
                log.info("skip duplicate %s", item.uri)
                continue
            did = doc_id(item.uri, sha)
            blobs.put(f"{did}/original{Path(item.uri).suffix}", item.data)
            docs.upsert_document(
                Document(id=did, source_uri=item.uri, content_sha256=sha,
                         acl=item.acl, metadata=item.metadata)
            )
            ingested.append(did)
    log.info("ingested %d documents", len(ingested))
    return ingested

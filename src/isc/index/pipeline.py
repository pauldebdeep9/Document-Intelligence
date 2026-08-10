"""Stage 4: Document -> Chunks -> embeddings -> vector store.

Runs per document, isolated -- same reasoning as parse/pipeline.py and
extract/pipeline.py: a batch of twenty documents must not abort because one
fails to chunk or embed. Each failure is caught, logged, and counted; the
caller (cli.py) decides what a non-empty failure list means for the exit code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from isc.common.config import Settings
from isc.common.errors import ChunkSettingsMismatch
from isc.common.logging import get_logger
from isc.common.tracing import Run, span
from isc.index.chunker import chunk_document, filters_from_record, settings_fingerprint
from isc.llm.ports import EmbeddingModel

# registry.get(doc_type) only finds a record class once that class's module
# has been imported -- @register's side effect runs at import time. This
# stage runs standalone (a separate `isc index` process, not guaranteed to
# run after anything that already imported these) -- see eval/pipeline.py
# for the identical fix, needed for the identical reason.
from isc.models.records import invoice, purchase_order  # noqa: F401
from isc.models.records.base import registry
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("index")


@dataclass
class IndexResult:
    indexed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)
    chunks: int = 0


def run(
    run_obj: Run,
    docs: SqliteDocStore,
    embedder: EmbeddingModel,
    store: LocalVectorStore,
    settings: Settings,
) -> IndexResult:
    # Computed once per run, not per document: it is a function of settings
    # alone. Threaded to store.add() (persisted in the store itself, read
    # back via store.settings_fingerprint() after load) AND written into
    # this run's index/manifest.json below -- both derived from this one
    # value, so the two can never disagree within a run. See this stage's
    # docstring in the P1-05 conversation for why both copies exist: the
    # store is what P1-09 actually queries at eval time; the manifest is
    # what P1-08's gold generation reads without loading vectors into memory.
    fingerprint = settings_fingerprint(settings.chunk)

    # Whole-run guard, in addition to store.add()'s own per-call guard below.
    # Every document in this run shares one fingerprint, so if it mismatches
    # the store's, every single document's add() is guaranteed to fail
    # identically. Checking once, up front, turns that into one clear
    # run-level error instead of twenty identical per-document failures --
    # and skips paying for twenty documents' worth of chunking and embedding
    # on a run that was never going to succeed.
    existing_fingerprint = store.settings_fingerprint()
    if existing_fingerprint is not None and existing_fingerprint != fingerprint:
        raise ChunkSettingsMismatch(
            f"store at {store.path} was built with chunk settings "
            f"{existing_fingerprint!r}; this run computed {fingerprint!r} for "
            "the current settings -- refusing to run. Start a fresh store or "
            "match the settings."
        )

    out_dir = run_obj.artifact_dir("index")
    result = IndexResult()

    for doc_id in docs.list_documents():
        try:
            with span("index.document", doc=doc_id):
                doc = docs.get_document(doc_id)
                if doc is None:
                    raise RuntimeError(f"{doc_id} vanished from the docstore mid-run")

                # A document with no extraction record yet still indexes --
                # just without po_number/supplier_id filters. Only look the
                # record up if one exists; do not require extract/ to have run.
                filters: dict[str, str] = {}
                record_payload = docs.get_record(doc_id)
                if record_payload is not None:
                    record_cls = registry.get(doc.doc_type)
                    record = record_cls.model_validate(record_payload)
                    filters = filters_from_record(record)

                chunks = chunk_document(doc, settings.chunk, filters=filters)
                if chunks:
                    # One batched embed() call per document, not one per
                    # chunk -- the embedder already batches internally
                    # (embed_batch_size); calling it per chunk would only
                    # multiply round trips and cache lookups for no benefit,
                    # since a document's chunk count here is always well
                    # under that batch size.
                    vectors = embedder.embed([c.text for c in chunks])
                    # add() first, remove_document() after -- not the other
                    # way around. add()'s replace-vs-insert bookkeeping needs
                    # a document's previous chunk ids still present when it
                    # runs, or an unchanged re-chunk (identical ids) would
                    # look like 100% fresh inserts instead of 100% replaces.
                    # Scoping remove_document() to keep_ids=the ids just
                    # written also means a failure inside add() (fingerprint
                    # mismatch, a bad chunk) never runs remove_document() at
                    # all, so this document's prior chunks are left exactly
                    # as they were -- matching this loop's per-document
                    # isolation guarantee, the same as embed() failing above.
                    upsert = store.add(chunks, vectors, settings_fingerprint=fingerprint)
                    removed = store.remove_document(
                        doc.id, keep_ids=frozenset(c.id for c in chunks)
                    )
                    result.chunks += len(chunks)
                    if upsert.replaced or removed:
                        log.info(
                            "doc %s: %d inserted, %d replaced, %d orphaned chunks removed",
                            doc_id, upsert.inserted, upsert.replaced, removed,
                        )
                else:
                    # Legitimately re-chunked to nothing (e.g. now-empty
                    # pages) -- still purge whatever this document had from a
                    # prior run, since chunk_document() succeeding with an
                    # empty result is not a failure.
                    removed = store.remove_document(doc.id)
                    if removed:
                        log.info("doc %s: re-chunked to zero chunks, %d orphans removed",
                                  doc_id, removed)
        except Exception as exc:  # noqa: BLE001 - batch isolation boundary, by design
            log.warning("index failed for %s: %s", doc_id, exc)
            result.failed.append((doc_id, str(exc)))
            continue
        result.indexed.append(doc_id)

    store.save()

    manifest = {
        "run_id": run_obj.run_id,
        "documents_indexed": len(result.indexed),
        "documents_failed": len(result.failed),
        "chunks": result.chunks,
        "settings_fingerprint": fingerprint,
        "embed_model": getattr(embedder, "model", None),
        "embed_dimensions": embedder.dimensions,
        "store_path": str(store.path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info("indexed %d documents (%d chunks), %d failed",
              len(result.indexed), result.chunks, len(result.failed))
    return result

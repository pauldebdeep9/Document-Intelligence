"""Stage 2: blob -> parsed Document via the parser fallback chain.

Runs per document, isolated: a batch of twenty documents must not abort because
one is malformed. Each failure is caught, logged, and counted; the caller
(cli.py) decides what a non-empty failure list means for the exit code.
parse_with_fallback's own exception handling stays narrow (NotImplementedError,
PyPdfError) -- that isolation is a different concern, one parser stepping aside
for the next. This is the outer boundary: one document stepping aside for the
next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from isc.common.logging import get_logger
from isc.common.tracing import Run, span
from isc.parse.chain import LayoutParser, NativeTextParser, OcrParser, Parser, parse_with_fallback
from isc.storage.local_blob import LocalBlobStore
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("parse")

DEFAULT_PARSERS: list[Parser] = [NativeTextParser(), LayoutParser(), OcrParser()]


@dataclass
class ParseResult:
    parsed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)


def run(run_obj: Run, blobs: LocalBlobStore, docs: SqliteDocStore,
        parsers: list[Parser] | None = None) -> ParseResult:
    parsers = parsers if parsers is not None else DEFAULT_PARSERS
    out_dir = run_obj.artifact_dir("parse")
    result = ParseResult()

    for doc_id in docs.list_documents():
        try:
            with span("parse.document", doc=doc_id):
                doc = docs.get_document(doc_id)
                if doc is None:
                    raise RuntimeError(f"{doc_id} vanished from the docstore mid-run")
                suffix = Path(doc.source_uri).suffix
                data = blobs.get(f"{doc_id}/original{suffix}")
                parsed, _conf = parse_with_fallback(doc, data, suffix, parsers)
                (out_dir / f"{doc_id}.json").write_text(parsed.model_dump_json(indent=2))
                docs.upsert_document(parsed)
        except Exception as exc:  # noqa: BLE001 - batch isolation boundary, by design
            log.warning("parse failed for %s: %s", doc_id, exc)
            result.failed.append((doc_id, str(exc)))
            continue
        result.parsed.append(doc_id)

    log.info("parsed %d documents, %d failed", len(result.parsed), len(result.failed))
    return result

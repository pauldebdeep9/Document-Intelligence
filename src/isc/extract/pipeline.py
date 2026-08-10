"""Stage 3: Document -> ExtractionRecord, one doc_type per run.

Runs per document, isolated -- same reasoning as parse/pipeline.py: a batch of
twenty documents must not abort because one fails structured output or exhausts
its schema repair budget. Each failure is caught, logged, and counted; the
caller (cli.py) decides what a non-empty failure list means for the exit code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from isc.common.confidence import Thresholds
from isc.common.logging import get_logger
from isc.common.tracing import Run, span
from isc.extract.extractor import extract as extract_one
from isc.llm.ports import ChatModel
from isc.models.document import DocType
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("extract")


@dataclass
class ExtractResult:
    extracted: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)
    mean_logprobs: dict[str, float | None] = field(default_factory=dict)


def run(
    run_obj: Run,
    docs: SqliteDocStore,
    model: ChatModel,
    doc_type: DocType,
    thresholds: Thresholds,
    masters_dir: Path,
) -> ExtractResult:
    out_dir = run_obj.artifact_dir("extract")
    result = ExtractResult()

    for doc_id in docs.list_documents(doc_type.value):
        try:
            with span("extract.document", doc=doc_id):
                doc = docs.get_document(doc_id)
                if doc is None:
                    raise RuntimeError(f"{doc_id} vanished from the docstore mid-run")
                if doc.parse_confidence is None:
                    raise RuntimeError(f"{doc_id} has no parse_confidence; run `isc parse` first")
                record, raw, llm_result = extract_one(
                    doc, model, docs, thresholds, doc.parse_confidence, masters_dir
                )
                # `raw` (the model's structured output, pre-wrap) is persisted
                # next to `record` so eval/ scores the extraction axis against
                # the exact bytes this run produced -- see
                # eval/extraction.py's load_artifact(). Re-running extraction
                # to get raw output for eval would make the eval depend on a
                # second, unreproducible inference pass.
                artifact = {
                    "doc_type": doc.doc_type.value,
                    "record": record.model_dump(mode="json"),
                    "raw": raw.model_dump(mode="json"),
                    "llm": {
                        "model": llm_result.model,
                        "finish_reason": llm_result.finish_reason,
                        "mean_logprob": llm_result.mean_logprob,
                        "usage": asdict(llm_result.usage),
                    },
                }
                (out_dir / f"{doc_id}.json").write_text(json.dumps(artifact, indent=2))
                docs.upsert_record(doc_id, doc.doc_type.value, record.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 - batch isolation boundary, by design
            log.warning("extract failed for %s: %s", doc_id, exc)
            result.failed.append((doc_id, str(exc)))
            continue
        result.extracted.append(doc_id)
        result.mean_logprobs[doc_id] = llm_result.mean_logprob

    log.info("extracted %d documents, %d failed", len(result.extracted), len(result.failed))
    return result

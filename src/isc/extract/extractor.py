"""Stage 3: Document -> ExtractionRecord.

The mapping Raw -> wrapped is where confidence is *assembled*, and it is the most
important 200 lines in the repo. Signals folded in, in order:

  1. Signal.MODEL      logprob from the completion
  2. Signal.SCHEMA     discounted per repair round (from structured.py)
  3. Signal.OCR/LAYOUT from the source block the value was located in
  4. Signal.LEXICAL    format check (PO number pattern, ISO date, currency code)
  5. Signal.MASTER_DATA value resolves in supplier/part master
  6. Signal.AGREEMENT  arithmetic cross-checks (line totals vs header total)

Anything routing to 'review' is enqueued in the HITL table with its weakest
factor attached, so the reviewer is told *why*, not just *that*.
"""

from __future__ import annotations

from isc.common.confidence import Confidence, Signal, Thresholds
from isc.common.config import load_prompt
from isc.llm.ports import ChatModel, Message
from isc.llm.structured import parse_structured
from isc.models.document import Document
from isc.models.records.base import ExtractionRecord, registry
from isc.storage.sqlite_docstore import SqliteDocStore


def extract(
    doc: Document,
    model: ChatModel,
    docs: SqliteDocStore,
    thresholds: Thresholds,
    parse_confidence: Confidence,
) -> ExtractionRecord:
    record_cls = registry.get(doc.doc_type)
    prompt = load_prompt(f"extract/{doc.doc_type}.v1.md")
    messages = [Message.system(prompt), Message.user(doc.text())]

    raw, conf, _ = parse_structured(model, messages, record_cls.raw_model)
    record = _wrap(raw, record_cls, doc, Confidence.independent(conf, parse_confidence))

    for name, field in record.fields().items():
        if thresholds.route(field.confidence) in {"review", "low_confidence"}:
            weakest = field.confidence.weakest()
            docs.enqueue_review(
                doc.id, name, field.confidence.score,
                f"{weakest.signal}={weakest.value:.2f}" if weakest else "",
            )
    return record


def _wrap(raw, record_cls, doc, base_confidence):
    """Raw -> ExtractedField per attribute, attaching spans and running checks."""
    raise NotImplementedError(
        "first slice: map 8 PurchaseOrderRaw fields, locate spans by string search, "
        "apply lexical validators from isc.extract.validators"
    )

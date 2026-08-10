"""Unit tests for extract/pipeline.py's artifact persistence and
eval/extraction.py's load_artifact() round trip.

Settles the raw-output dependency: eval must score the exact bytes a run
produced, not a fresh completion, so extract/pipeline.py persists the
model's raw structured output and LLMResult metadata (finish_reason,
mean_logprob, usage) next to the wrapped record in runs/<id>/extract/, and
eval/ reads both back with no model calls at all. ScriptedModel stands in
for a live provider -- same reasoning as test_structured.py's fake -- so
these tests make no network calls either.
"""

from __future__ import annotations

import json

from isc.common.confidence import Confidence, Thresholds
from isc.common.tracing import Run
from isc.eval.extraction import compare, load_artifact
from isc.extract.pipeline import run as run_extract
from isc.llm.ports import LLMResult, Usage
from isc.models.acl import AclSet
from isc.models.document import Block, BlockType, DocType, Document, Page
from isc.models.records.purchase_order import PurchaseOrderRaw
from isc.storage.sqlite_docstore import SqliteDocStore


class ScriptedModel:
    """Fake ChatModel returning one fixed structured response. No network."""

    def __init__(self, raw: PurchaseOrderRaw, **llm_kwargs) -> None:
        self._text = raw.model_dump_json()
        self.calls = 0
        self._kwargs = llm_kwargs

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        self.calls += 1
        return LLMResult(text=self._text, model="fake", **self._kwargs)


def _masters_dir(tmp_path):
    d = tmp_path / "masters"
    d.mkdir()
    (d / "suppliers.json").write_text("[]")
    (d / "parts.json").write_text("[]")
    (d / "unmastered_parts.json").write_text("[]")
    (d / "sites.json").write_text("[]")
    return d


def _doc() -> Document:
    blocks = [Block(id="b1", type=BlockType.PARAGRAPH,
                     text="PO Number 4500123456 Order Date 16/08/2025 Supplier Acme Supply Co",
                     page=1, reading_order=0)]
    return Document(
        id="doc_1", source_uri="x.pdf", content_sha256="x",
        doc_type=DocType.PURCHASE_ORDER,
        acl=AclSet(allow_terms=frozenset({"everyone:*"})),
        pages=[Page(number=1, blocks=blocks)],
        parse_confidence=Confidence(0.9),
    )


def test_pipeline_persists_raw_and_llm_metadata_next_to_record(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc())
    raw = PurchaseOrderRaw(po_number="4500123456", po_date="16/08/2025",
                            supplier_name="Acme Supply Co", total_amount=100.0)
    model = ScriptedModel(raw, usage=Usage(120, 40), mean_logprob=-0.05, finish_reason="stop")
    run_obj = Run("t1", tmp_path / "runs")

    result = run_extract(run_obj, docs, model, DocType.PURCHASE_ORDER,
                          Thresholds(), _masters_dir(tmp_path))

    assert result.extracted == ["doc_1"]
    assert not result.failed
    assert model.calls == 1  # exactly one completion for the whole run -- no re-run anywhere

    artifact = json.loads((run_obj.dir / "extract" / "doc_1.json").read_text())
    assert artifact["doc_type"] == "purchase_order"
    assert artifact["record"]["po_number"]["value"] == "4500123456"
    assert artifact["raw"] == raw.model_dump(mode="json")
    assert artifact["llm"] == {
        "model": "fake",
        "finish_reason": "stop",
        "mean_logprob": -0.05,
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "cached_prompt_tokens": 0},
    }


def test_load_artifact_round_trips_record_and_raw_with_no_model_calls(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc())
    raw = PurchaseOrderRaw(po_number="4500123456", po_date="16/08/2025",
                            supplier_name="Acme Supply Co", total_amount=100.0)
    model = ScriptedModel(raw)
    run_obj = Run("t2", tmp_path / "runs")
    run_extract(run_obj, docs, model, DocType.PURCHASE_ORDER, Thresholds(), _masters_dir(tmp_path))

    record, raw_predicted = load_artifact(run_obj.dir / "extract" / "doc_1.json")

    assert model.calls == 1  # the one call made during the run itself
    assert record.document_id == "doc_1"
    assert record.po_number.value == "4500123456"
    assert raw_predicted == raw.model_dump(mode="json")

    # eval reads both halves straight from the artifact -- compare() takes
    # them as-is, with no further calls to `model` at all.
    gold = {
        "raw": {"po_number": "4500123456", "po_date": "16/08/2025",
                "supplier_name": "Acme Supply Co", "total_amount": "100.00"},
        "normalised": {"po_number": "4500123456", "po_date": "2025-08-16",
                        "supplier_name": "Acme Supply Co", "total_amount": "100.00"},
    }
    outcomes = compare("doc_1", gold, raw_predicted, record)
    assert model.calls == 1  # still just the one -- compare() made no new calls
    po_number_outcomes = {o.axis: o.outcome for o in outcomes if o.field_name == "po_number"}
    assert po_number_outcomes == {"extraction": "correct", "normalisation": "correct"}

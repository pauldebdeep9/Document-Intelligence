"""Unit tests for index/pipeline.py: chunk -> embed -> store wiring, filters
from an extraction record, per-document isolation, one batched embed() call
per document, the manifest artifact, and the settings-fingerprint mechanism.

FakeEmbedder stands in for a live provider so these tests make no network
calls -- same reasoning as test_extract_pipeline.py's ScriptedModel.
"""

from __future__ import annotations

import json

import pytest

from isc.common.confidence import Confidence
from isc.common.config import ChunkSettings, Settings
from isc.common.errors import ChunkSettingsMismatch
from isc.common.tracing import Run
from isc.index.chunker import settings_fingerprint
from isc.index.pipeline import run as run_index
from isc.models.acl import AclSet
from isc.models.document import Block, BlockType, DocType, Document, Page
from isc.models.records.base import ExtractedField
from isc.models.records.purchase_order import PurchaseOrder
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore


class FakeEmbedder:
    """No network. Tracks call/batch shape so tests can assert one call per
    document, not one per chunk."""

    def __init__(self, dims: int = 4) -> None:
        self._dims = dims
        self.model = "fake-embed"
        self.calls = 0
        self.batch_sizes: list[int] = []

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return [[0.1 * (i + 1)] * self._dims for i in range(len(texts))]


def _doc(doc_id: str, with_table: bool = True) -> Document:
    blocks = [
        Block(id="b0", type=BlockType.TITLE, text="PURCHASE ORDER", page=1, reading_order=0),
        Block(id="b1", type=BlockType.KEY_VALUE,
              text="PO Number   4500123456   Order Date   16/08/2025",
              page=1, reading_order=1),
    ]
    if with_table:
        from isc.models.document import Table
        table = Table(rows=[
            ["Item", "Part Number", "Description", "Qty"],
            ["10", "PLC-1756-L83", "ControlLogix module", "3"],
        ], header_rows=1)
        blocks.append(Block(
            id="b2", type=BlockType.TABLE,
            text="Item  Part Number  Description  Qty\n10  PLC-1756-L83  ControlLogix module  3",
            page=1, reading_order=2, table=table,
        ))
    return Document(id=doc_id, source_uri=f"{doc_id}.pdf", content_sha256="x",
                     doc_type=DocType.PURCHASE_ORDER,
                     acl=AclSet(allow_terms=frozenset({"everyone:*"})),
                     pages=[Page(number=1, blocks=blocks)])


def _record(po_number: str = "4500123456", supplier_id: str | None = "V102337") -> PurchaseOrder:
    def f(value):
        return ExtractedField(value=value, confidence=Confidence(0.9))
    return PurchaseOrder(
        document_id="doc_x", po_number=f(po_number), supplier_id=f(supplier_id),
    )


def _settings(tmp_path) -> Settings:
    s = Settings()
    s.paths.data = tmp_path / "data"
    s.paths.runs = tmp_path / "runs"
    return s


def test_indexes_a_document_and_writes_the_manifest(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    store = LocalVectorStore(tmp_path / "store.pkl")
    embedder = FakeEmbedder()
    run_obj = Run("t1", tmp_path / "runs")
    settings = _settings(tmp_path)

    result = run_index(run_obj, docs, embedder, store, settings)

    assert result.indexed == ["doc_1"]
    assert not result.failed
    assert result.chunks > 0
    assert store.count() == result.chunks

    manifest = json.loads((run_obj.dir / "index" / "manifest.json").read_text())
    assert manifest["documents_indexed"] == 1
    assert manifest["documents_failed"] == 0
    assert manifest["chunks"] == result.chunks
    assert manifest["settings_fingerprint"] == settings_fingerprint(settings.chunk)
    assert manifest["embed_model"] == "fake-embed"
    assert manifest["embed_dimensions"] == 4


def test_embeds_once_per_document_not_once_per_chunk(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))  # title + kv + table -> multiple chunks
    store = LocalVectorStore(tmp_path / "store.pkl")
    embedder = FakeEmbedder()
    run_obj = Run("t1", tmp_path / "runs")

    result = run_index(run_obj, docs, embedder, store, _settings(tmp_path))

    assert result.chunks > 1, "fixture should produce more than one chunk"
    assert embedder.calls == 1
    assert embedder.batch_sizes == [result.chunks]


def test_populates_filters_from_extraction_record(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    record = _record(po_number="4500123456", supplier_id="V102337")
    docs.upsert_record("doc_1", "purchase_order", record.model_dump(mode="json"))
    store = LocalVectorStore(tmp_path / "store.pkl")

    run_index(Run("t1", tmp_path / "runs"), docs, FakeEmbedder(), store, _settings(tmp_path))

    assert store.count() > 0
    assert all(c.filters == {"po_number": "4500123456", "supplier_id": "V102337"}
               for c in store._chunks)


def test_indexes_without_an_extraction_record(tmp_path):
    """A document with no extraction record yet still indexes -- just
    without filters."""
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    store = LocalVectorStore(tmp_path / "store.pkl")
    run_obj = Run("t1", tmp_path / "runs")

    result = run_index(run_obj, docs, FakeEmbedder(), store, _settings(tmp_path))

    assert result.indexed == ["doc_1"]
    assert all(c.filters == {} for c in store._chunks)


def test_one_bad_document_does_not_abort_the_others(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_good"))
    docs.upsert_document(_doc("doc_bad"))
    store = LocalVectorStore(tmp_path / "store.pkl")
    real_get = docs.get_document

    def flaky_get(doc_id):
        if doc_id == "doc_bad":
            return None  # triggers "vanished from the docstore mid-run"
        return real_get(doc_id)

    docs.get_document = flaky_get  # type: ignore[method-assign]
    run_obj = Run("t1", tmp_path / "runs")

    result = run_index(run_obj, docs, FakeEmbedder(), store, _settings(tmp_path))

    assert result.indexed == ["doc_good"]
    assert len(result.failed) == 1
    assert result.failed[0][0] == "doc_bad"
    assert "vanished" in result.failed[0][1]


def test_store_save_round_trips_in_a_fresh_instance(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    path = tmp_path / "store.pkl"
    store = LocalVectorStore(path)
    run_obj = Run("t1", tmp_path / "runs")

    result = run_index(run_obj, docs, FakeEmbedder(), store, _settings(tmp_path))

    reloaded = LocalVectorStore(path)
    assert reloaded.count() == result.chunks
    assert reloaded.settings_fingerprint() == store.settings_fingerprint()


def test_settings_fingerprint_mismatch_is_recorded_in_manifest_and_store(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    store = LocalVectorStore(tmp_path / "store.pkl")
    settings = _settings(tmp_path)
    settings.chunk = ChunkSettings(target_tokens=256)

    run_index(Run("t1", tmp_path / "runs"), docs, FakeEmbedder(), store, settings)

    expected = settings_fingerprint(settings.chunk)
    assert store.settings_fingerprint() == expected
    manifest = json.loads((tmp_path / "runs" / "t1" / "index" / "manifest.json").read_text())
    assert manifest["settings_fingerprint"] == expected


def test_running_index_twice_end_to_end_does_not_duplicate(tmp_path):
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    docs.upsert_document(_doc("doc_2"))
    store = LocalVectorStore(tmp_path / "store.pkl")
    settings = _settings(tmp_path)

    first = run_index(Run("t1", tmp_path / "runs"), docs, FakeEmbedder(), store, settings)
    count_after_first = store.count()
    assert count_after_first > 0

    second = run_index(Run("t2", tmp_path / "runs"), docs, FakeEmbedder(), store, settings)

    assert store.count() == count_after_first
    assert len({c.id for c in store._chunks}) == store.count()
    assert first.chunks == second.chunks


def test_settings_mismatch_against_an_existing_store_aborts_before_removing_anything(tmp_path):
    """add()'s upsert removes a document's old chunks before re-adding its new
    ones. Without a whole-run preflight, a fingerprint mismatch would only
    surface on the first document's add() call -- AFTER that document's old
    chunks were already removed -- and since every document in the run shares
    one fingerprint, every document would fail identically: the store would be
    silently wiped to empty one document at a time rather than left untouched.
    """
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    docs.upsert_document(_doc("doc_1"))
    docs.upsert_document(_doc("doc_2"))
    store = LocalVectorStore(tmp_path / "store.pkl")
    settings = _settings(tmp_path)

    run_index(Run("t1", tmp_path / "runs"), docs, FakeEmbedder(), store, settings)
    count_before = store.count()
    ids_before = {c.id for c in store._chunks}
    assert count_before > 0

    settings.chunk = ChunkSettings(target_tokens=256)  # different fingerprint
    with pytest.raises(ChunkSettingsMismatch):
        run_index(Run("t2", tmp_path / "runs"), docs, FakeEmbedder(), store, settings)

    assert store.count() == count_before
    assert {c.id for c in store._chunks} == ids_before

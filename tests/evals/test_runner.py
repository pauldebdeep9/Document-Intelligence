import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from reportlab.pdfgen import canvas

from evals.gold.schema import (
    Anchor,
    ChunkingConfig,
    ExtractionGold,
    GoldItem,
    GoldSet,
    RetrievalGold,
)
from evals.gold.split import Split
from evals.runner import (
    DocumentRecord,
    ItemRecord,
    RunConfig,
    RunRecord,
    preflight,
    run_eval,
    save_run_record,
)
from isc.models import GroundedAnswer, PurchaseOrder

_INSUFFICIENT_ANSWER = "I don't have enough information in the provided sources."


def _write_pdf(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), invariant=1)
    y = 720
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 18
    pdf.showPage()
    pdf.save()


def _deterministic_vector(text: str, dim: int = 8) -> list[float]:
    """A fixed-dimension vector derived from a hash of the text — deterministic across
    processes, unlike Python's built-in hash() which is randomized per-process for strings.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(dim)]


def _absent_question(doc_id: str, index: int) -> RetrievalGold:
    return RetrievalGold(
        question_id=f"{doc_id}-q{index}",
        doc_id=doc_id,
        question=f"Absent question {index} about {doc_id}?",
        question_class="absent",
        anchors=[],
    )


class _StubResponses:
    def __init__(self, fail_on_nth_answer_call: int | None, exception: Exception) -> None:
        self.calls: list[dict[str, object]] = []
        self._answer_call_count = 0
        self._fail_on_nth_answer_call = fail_on_nth_answer_call
        self._exception = exception

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        text_format = kwargs["text_format"]
        if text_format is PurchaseOrder:
            return SimpleNamespace(
                output_parsed=PurchaseOrder(po_number="STUB-PO"), output=[]
            )
        if text_format is GroundedAnswer:
            self._answer_call_count += 1
            if self._fail_on_nth_answer_call == self._answer_call_count:
                raise self._exception
            return SimpleNamespace(
                output_parsed=GroundedAnswer(answer=_INSUFFICIENT_ANSWER, source_chunk_ids=[]),
                output=[],
            )
        raise AssertionError(f"unexpected text_format: {text_format!r}")


class _StubEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        data = [
            SimpleNamespace(index=index, embedding=_deterministic_vector(text))
            for index, text in enumerate(texts)
        ]
        return SimpleNamespace(data=data)


class StubClient:
    """Satisfies the call surface run_eval uses: client.responses.parse(...) and
    client.embeddings.create(...). Exists to test the harness's plumbing (call counts,
    pooling, error propagation), not to measure extraction/retrieval quality — its default
    answers are fixed, valid stand-ins, not realistic ones.
    """

    def __init__(
        self,
        *,
        fail_on_nth_answer_call: int | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.responses = _StubResponses(
            fail_on_nth_answer_call, exception or RuntimeError("stub failure")
        )
        self.embeddings = _StubEmbeddings()


def _small_goldset(doc_ids: list[str], questions_per_doc: int = 1) -> GoldSet:
    return GoldSet(
        version="test",
        chunking=ChunkingConfig(chunk_size=1200, overlap=200),
        items=[
            GoldItem(
                doc_id=doc_id,
                extraction=ExtractionGold(doc_id=doc_id, expected=PurchaseOrder()),
                retrieval=[
                    _absent_question(doc_id, index) for index in range(1, questions_per_doc + 1)
                ],
            )
            for doc_id in doc_ids
        ],
    )


# --- preflight ---------------------------------------------------------------------------------


def test_preflight_raises_for_blank_chat_model(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])
    client = StubClient()

    with pytest.raises(ValueError, match="chat_model must not be blank"):
        run_eval(goldset, Split.DEV, client, "", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir)

    assert client.responses.calls == []
    assert client.embeddings.calls == []


def test_preflight_raises_for_blank_embedding_model() -> None:
    goldset = _small_goldset(["doc-a"])

    with pytest.raises(ValueError, match="embedding_model must not be blank"):
        preflight(goldset, "chat-model", "  ", 1200, 200, 3)


def test_preflight_raises_for_k_less_than_one() -> None:
    goldset = _small_goldset(["doc-a"])

    with pytest.raises(ValueError, match="k must be at least 1"):
        preflight(goldset, "chat-model", "embed-model", 1200, 200, 0)


def test_preflight_raises_for_invalid_chunk_size() -> None:
    goldset = _small_goldset(["doc-a"])

    with pytest.raises(ValueError, match="Chunk size must be greater than zero"):
        preflight(goldset, "chat-model", "embed-model", 0, 200, 3)


def test_preflight_raises_for_empty_goldset() -> None:
    goldset = GoldSet(
        version="test", chunking=ChunkingConfig(chunk_size=1200, overlap=200), items=[]
    )

    with pytest.raises(ValueError, match="goldset has no items"):
        preflight(goldset, "chat-model", "embed-model", 1200, 200, 3)


def test_preflight_raises_for_unresolvable_anchor(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = GoldSet(
        version="test",
        chunking=ChunkingConfig(chunk_size=1200, overlap=200),
        items=[
            GoldItem(
                doc_id="doc-a",
                extraction=ExtractionGold(doc_id="doc-a", expected=PurchaseOrder()),
                retrieval=[
                    RetrievalGold(
                        question_id="doc-a-q1",
                        doc_id="doc-a",
                        question="What is the PO number?",
                        question_class="header_field",
                        anchors=[Anchor(page_number=1, text="this text is not on the page")],
                    ),
                ],
            ),
        ],
    )
    client = StubClient()

    with pytest.raises(ValueError, match="Unresolvable anchor"):
        run_eval(
            goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
        )

    assert client.responses.calls == []
    assert client.embeddings.calls == []


def test_preflight_passes_for_a_valid_configuration(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])

    preflight(goldset, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir)


# --- run_eval ------------------------------------------------------------------------------


def test_run_eval_ingests_each_document_exactly_once_regardless_of_question_count(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001", "Supplier: Alpha Corp"])
    goldset = _small_goldset(["doc-a"], questions_per_doc=3)
    client = StubClient()

    record = run_eval(
        goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
    )

    extraction_calls = [c for c in client.responses.calls if c["text_format"] is PurchaseOrder]
    assert len(extraction_calls) == 1

    chunk_count = record.documents[0].chunk_count
    assert len(client.embeddings.calls[0]["input"]) == chunk_count  # type: ignore[arg-type]
    # 1 chunk-embedding call (ingest) + 3 query-embedding calls (one per question), not 3+3.
    assert len(client.embeddings.calls) == 1 + 3


def test_run_eval_pools_chunks_across_every_document_in_split(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    _write_pdf(pdf_dir / "doc-b.pdf", ["PO Number: PO-B001"])
    goldset = GoldSet(
        version="test",
        chunking=ChunkingConfig(chunk_size=1200, overlap=200),
        items=[
            GoldItem(
                doc_id="doc-a",
                extraction=ExtractionGold(doc_id="doc-a", expected=PurchaseOrder()),
                retrieval=[
                    RetrievalGold(
                        question_id="doc-a-q1",
                        doc_id="doc-a",
                        question="What is the PO number?",
                        question_class="header_field",
                        anchors=[Anchor(page_number=1, text="PO Number: PO-A001")],
                    ),
                ],
            ),
            GoldItem(
                doc_id="doc-b",
                extraction=ExtractionGold(doc_id="doc-b", expected=PurchaseOrder()),
                retrieval=[_absent_question("doc-b", 1)],
            ),
        ],
    )
    client = StubClient()

    record = run_eval(
        goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
    )

    item = next(i for i in record.items if i.question_id == "doc-a-q1")
    # k=3 but only 2 chunks total exist (one per doc) -> top_k_chunks returns both; the pool
    # must include doc-b's chunk even though the question is about doc-a.
    assert set(item.retrieved_doc_ids) == {"doc-a", "doc-b"}
    assert record.config.pooled_chunk_count == 2


def test_run_eval_records_per_item_provider_exception_and_continues(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"], questions_per_doc=3)
    provider_error = RuntimeError("provider unavailable")
    client = StubClient(fail_on_nth_answer_call=2, exception=provider_error)

    record = run_eval(
        goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
    )

    assert len(record.items) == 3
    errored = [item for item in record.items if item.error is not None]
    assert len(errored) == 1
    assert errored[0].error == "RuntimeError: provider unavailable"
    succeeded = [item for item in record.items if item.error is None]
    assert len(succeeded) == 2


def test_run_eval_returns_config_matching_the_run(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])
    client = StubClient()

    record = run_eval(
        goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
    )

    assert record.config.chat_model == "chat-model"
    assert record.config.embedding_model == "embed-model"
    assert record.config.chunk_size == 1200
    assert record.config.overlap == 200
    assert record.config.k == 3
    assert record.config.goldset_version == "test"
    assert record.config.split == Split.DEV
    assert record.config.document_count == 1


def test_run_eval_document_records_hold_extracted_purchase_order_and_chunk_count(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])
    client = StubClient()

    record = run_eval(
        goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
    )

    assert len(record.documents) == 1
    assert record.documents[0].doc_id == "doc-a"
    assert record.documents[0].purchase_order == PurchaseOrder(po_number="STUB-PO")
    assert record.documents[0].chunk_count == 1


# --- save_run_record -----------------------------------------------------------------------


def _sample_record() -> RunRecord:
    return RunRecord(
        run_id="sample-run-id",
        created_utc="2026-01-01T00:00:00+00:00",
        git_commit_sha="deadbeef",
        config=RunConfig(
            chat_model="chat-model",
            embedding_model="embed-model",
            chunk_size=1200,
            overlap=200,
            k=3,
            goldset_version="test",
            split=Split.DEV,
            document_count=1,
            pooled_chunk_count=1,
        ),
        documents=[
            DocumentRecord(
                doc_id="doc-a", purchase_order=PurchaseOrder(po_number="X"), chunk_count=1
            ),
        ],
        items=[
            ItemRecord(
                question_id="doc-a-q1",
                doc_id="doc-a",
                question_class="absent",
                retrieved_chunk_ids=[],
                retrieved_doc_ids=[],
                retrieved_page_numbers=[],
                retrieved_texts=[],
                retrieved_scores=[],
                answer=_INSUFFICIENT_ANSWER,
                source_chunk_ids=[],
                error=None,
            ),
        ],
    )


def test_save_run_record_writes_json_named_by_run_id(tmp_path: Path) -> None:
    record = _sample_record()

    output_path = save_run_record(record, runs_dir=tmp_path)

    assert output_path == tmp_path / "sample-run-id.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text())["run_id"] == "sample-run-id"


def test_save_run_record_creates_runs_dir_if_missing(tmp_path: Path) -> None:
    record = _sample_record()
    runs_dir = tmp_path / "nested" / "runs"

    output_path = save_run_record(record, runs_dir=runs_dir)

    assert output_path.exists()


def test_save_run_record_round_trips(tmp_path: Path) -> None:
    record = _sample_record()

    output_path = save_run_record(record, runs_dir=tmp_path)
    reloaded = RunRecord.model_validate_json(output_path.read_text())

    assert reloaded == record


# --- RunRecord JSON round-trip ---------------------------------------------------------------


def test_run_record_round_trips_through_json_without_loss() -> None:
    record = _sample_record()

    reloaded = RunRecord.model_validate_json(record.model_dump_json())

    assert reloaded == record

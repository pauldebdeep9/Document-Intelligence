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
    _scope_candidates,
    preflight,
    run_eval,
    save_run_record,
)
from isc.models import Chunk, GroundedAnswer, PurchaseOrder

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


def _chunk(doc_id: str, chunk_number: int) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}:page-001-chunk-{chunk_number:03d}",
        page_number=1,
        text=f"text for {doc_id} chunk {chunk_number}",
    )


# --- _scope_candidates ---------------------------------------------------------------------


def test_scope_candidates_filters_to_only_the_given_doc_id() -> None:
    chunks = [_chunk("po-004", 1), _chunk("po-005", 1), _chunk("po-010", 1)]
    embeddings = [[1.0], [2.0], [3.0]]

    scoped_chunks, scoped_embeddings = _scope_candidates(chunks, embeddings, "po-004")

    assert [c.doc_id for c in scoped_chunks] == ["po-004"]
    assert scoped_embeddings == [[1.0]]


def test_scope_candidates_with_scope_none_returns_input_unchanged() -> None:
    chunks = [_chunk("po-004", 1), _chunk("po-005", 1)]
    embeddings = [[1.0], [2.0]]

    scoped_chunks, scoped_embeddings = _scope_candidates(chunks, embeddings, None)

    assert scoped_chunks == chunks
    assert scoped_embeddings == embeddings


def test_scope_candidates_with_no_matching_doc_id_returns_empty_not_an_error() -> None:
    chunks = [_chunk("po-004", 1), _chunk("po-005", 1)]
    embeddings = [[1.0], [2.0]]

    scoped_chunks, scoped_embeddings = _scope_candidates(chunks, embeddings, "po-999")

    assert scoped_chunks == []
    assert scoped_embeddings == []


def test_scope_candidates_keeps_multiple_chunks_from_the_same_scoped_document() -> None:
    chunks = [_chunk("po-006", 1), _chunk("po-006", 2), _chunk("po-001", 1)]
    embeddings = [[1.0], [2.0], [3.0]]

    scoped_chunks, scoped_embeddings = _scope_candidates(chunks, embeddings, "po-006")

    assert [c.chunk_id for c in scoped_chunks] == [
        "po-006:page-001-chunk-001",
        "po-006:page-001-chunk-002",
    ]
    assert scoped_embeddings == [[1.0], [2.0]]


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


# chunk_pages([], doc_id="x", chunk_size=0, overlap=200) and
# chunk_pages([], doc_id="x", chunk_size=1200, overlap=1200) were verified directly (not
# through preflight) to raise ValueError before touching any page — chunk_size/overlap
# validation runs before chunk_pages's page-number-uniqueness check and its chunking loop, so
# preflight's indirect check (calling chunk_pages against an empty page list) is not a silent
# pass-through. These four go through run_eval specifically, not preflight directly, so a
# real client is in scope to assert zero calls were ever recorded against it.
@pytest.mark.parametrize(
    ("chunk_size", "overlap", "match"),
    [
        (0, 200, "Chunk size must be greater than zero"),
        (-100, 200, "Chunk size must be greater than zero"),
        (1200, -1, "Overlap must be non-negative and smaller than chunk size"),
        (1200, 1200, "Overlap must be non-negative and smaller than chunk size"),
    ],
)
def test_run_eval_aborts_via_preflight_for_invalid_chunking_params_before_any_call(
    tmp_path: Path,
    chunk_size: int,
    overlap: int,
    match: str,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])
    client = StubClient()

    with pytest.raises(ValueError, match=match):
        run_eval(
            goldset,
            Split.DEV,
            client,
            "chat-model",
            "embed-model",
            chunk_size,
            overlap,
            3,
            pdf_dir=pdf_dir,
        )

    assert client.responses.calls == []
    assert client.embeddings.calls == []


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


def test_preflight_raises_for_goldset_version_mismatch(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])  # goldset.version == "test"

    with pytest.raises(ValueError, match="goldset version mismatch"):
        preflight(
            goldset,
            "chat-model",
            "embed-model",
            1200,
            200,
            3,
            pdf_dir=pdf_dir,
            expected_goldset_version="1.1.0",
        )


def test_preflight_error_message_names_both_versions(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])  # goldset.version == "test"

    with pytest.raises(ValueError, match="loaded 'test'.*expected '1.1.0'"):
        preflight(
            goldset,
            "chat-model",
            "embed-model",
            1200,
            200,
            3,
            pdf_dir=pdf_dir,
            expected_goldset_version="1.1.0",
        )


def test_preflight_passes_when_goldset_version_matches_expected(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])  # goldset.version == "test"

    preflight(
        goldset,
        "chat-model",
        "embed-model",
        1200,
        200,
        3,
        pdf_dir=pdf_dir,
        expected_goldset_version="test",
    )


def test_preflight_skips_version_check_when_expected_goldset_version_not_given(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])  # goldset.version == "test"

    # No expected_goldset_version passed -> no comparison happens, regardless of what
    # goldset.version actually is.
    preflight(goldset, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir)


def test_run_eval_raises_via_preflight_for_goldset_version_mismatch_before_any_call(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    goldset = _small_goldset(["doc-a"])  # goldset.version == "test"
    client = StubClient()

    with pytest.raises(ValueError, match="goldset version mismatch"):
        run_eval(
            goldset,
            Split.DEV,
            client,
            "chat-model",
            "embed-model",
            1200,
            200,
            3,
            pdf_dir=pdf_dir,
            expected_goldset_version="1.1.0",
        )

    assert client.responses.calls == []
    assert client.embeddings.calls == []


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


def test_run_eval_pools_chunks_across_every_document_in_split_when_scoping_pooled(
    tmp_path: Path,
) -> None:
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
        goldset,
        Split.DEV,
        client,
        "chat-model",
        "embed-model",
        1200,
        200,
        3,
        pdf_dir=pdf_dir,
        scoping="pooled",
    )

    item = next(i for i in record.items if i.question_id == "doc-a-q1")
    # k=3 but only 2 chunks total exist (one per doc) -> top_k_chunks returns both; the pool
    # must include doc-b's chunk even though the question is about doc-a. Only true under
    # explicit scoping="pooled" now that "scoped" is the default.
    assert set(item.retrieved_doc_ids) == {"doc-a", "doc-b"}
    assert record.config.pooled_chunk_count == 2
    assert record.config.scoping == "pooled"
    assert item.scope is None


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


# --- run_eval scoping ------------------------------------------------------------------------


def _two_doc_goldset() -> GoldSet:
    return GoldSet(
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


def test_run_eval_defaults_to_scoped_retrieval(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    _write_pdf(pdf_dir / "doc-b.pdf", ["PO Number: PO-B001"])
    goldset = _two_doc_goldset()
    client = StubClient()

    # No scoping argument passed -> must default to "scoped".
    record = run_eval(
        goldset, Split.DEV, client, "chat-model", "embed-model", 1200, 200, 3, pdf_dir=pdf_dir
    )

    item = next(i for i in record.items if i.question_id == "doc-a-q1")
    assert set(item.retrieved_doc_ids) == {"doc-a"}
    assert record.config.scoping == "scoped"


def test_run_eval_scoped_with_k_larger_than_document_chunk_count_has_no_filler(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    _write_pdf(pdf_dir / "doc-b.pdf", ["PO Number: PO-B001"])
    goldset = _two_doc_goldset()
    client = StubClient()

    # doc-a chunks to exactly 1 chunk; k=5 is far larger than that.
    record = run_eval(
        goldset,
        Split.DEV,
        client,
        "chat-model",
        "embed-model",
        1200,
        200,
        5,
        pdf_dir=pdf_dir,
        scoping="scoped",
    )

    item = next(i for i in record.items if i.question_id == "doc-a-q1")
    assert item.retrieved_doc_ids == ["doc-a"]
    assert len(item.retrieved_chunk_ids) == 1


def test_run_eval_records_scope_used_on_each_item_for_both_modes(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    _write_pdf(pdf_dir / "doc-a.pdf", ["PO Number: PO-A001"])
    _write_pdf(pdf_dir / "doc-b.pdf", ["PO Number: PO-B001"])
    goldset = _two_doc_goldset()

    scoped_record = run_eval(
        goldset,
        Split.DEV,
        StubClient(),
        "chat-model",
        "embed-model",
        1200,
        200,
        3,
        pdf_dir=pdf_dir,
        scoping="scoped",
    )
    scoped_item = next(i for i in scoped_record.items if i.question_id == "doc-a-q1")
    assert scoped_item.scope == "doc-a"

    pooled_record = run_eval(
        goldset,
        Split.DEV,
        StubClient(),
        "chat-model",
        "embed-model",
        1200,
        200,
        3,
        pdf_dir=pdf_dir,
        scoping="pooled",
    )
    pooled_item = next(i for i in pooled_record.items if i.question_id == "doc-a-q1")
    assert pooled_item.scope is None


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
            scoping="scoped",
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
                scope="doc-a",
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


def test_save_run_record_round_trips_with_an_errored_item(tmp_path: Path) -> None:
    record = _sample_record().model_copy(
        update={
            "items": [
                ItemRecord(
                    question_id="doc-a-q1",
                    doc_id="doc-a",
                    question_class="header_field",
                    scope="doc-a",
                    retrieved_chunk_ids=["doc-a:page-001-chunk-001"],
                    retrieved_doc_ids=["doc-a"],
                    retrieved_page_numbers=[1],
                    retrieved_texts=["PO Number: PO-A001"],
                    retrieved_scores=[0.9],
                    answer=None,
                    source_chunk_ids=[],
                    error="RuntimeError: provider unavailable",
                ),
            ],
        }
    )

    output_path = save_run_record(record, runs_dir=tmp_path)
    reloaded = RunRecord.model_validate_json(output_path.read_text())

    assert reloaded == record
    assert reloaded.items[0].error == "RuntimeError: provider unavailable"


def test_save_run_record_writes_valid_utf8_json(tmp_path: Path) -> None:
    record = _sample_record().model_copy(
        update={
            "documents": [
                DocumentRecord(
                    doc_id="doc-a",
                    purchase_order=PurchaseOrder(supplier_name="Müller & Söhne"),
                    chunk_count=1,
                ),
            ],
        }
    )

    output_path = save_run_record(record, runs_dir=tmp_path)
    with output_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["documents"][0]["purchase_order"]["supplier_name"] == "Müller & Söhne"


# --- RunRecord JSON round-trip ---------------------------------------------------------------


def test_run_record_round_trips_through_json_without_loss() -> None:
    record = _sample_record()

    reloaded = RunRecord.model_validate_json(record.model_dump_json())

    assert reloaded == record

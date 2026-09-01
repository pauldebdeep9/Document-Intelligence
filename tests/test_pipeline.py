from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from reportlab.pdfgen import canvas

from isc import pipeline
from isc.chunking import chunk_pages
from isc.models import (
    Chunk,
    GroundedAnswer,
    PDFPage,
    PipelineResult,
    PurchaseOrder,
    SourceEvidence,
)
from isc.pdf import extract_pdf_pages


def test_process_document_orchestrates_primitives_and_resolves_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    received: dict[str, object] = {}
    client = object()
    pages = [PDFPage(page_number=1, text="Purchase Order PO-1001")]
    purchase_order = PurchaseOrder(po_number="PO-1001")
    chunks = [
        Chunk(chunk_id="a", page_number=1, text="Chunk A"),
        Chunk(chunk_id="b", page_number=1, text="Chunk B"),
        Chunk(chunk_id="c", page_number=2, text="Chunk C"),
    ]
    vectors = [
        [9.0, 9.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    retrieved_sources = [
        SourceEvidence(chunk_id="a", page_number=1, text="Chunk A", score=0.9),
        SourceEvidence(chunk_id="b", page_number=1, text="Chunk B", score=0.8),
        SourceEvidence(chunk_id="c", page_number=2, text="Chunk C", score=0.7),
    ]
    grounded_answer = GroundedAnswer(
        answer="Supported by C and A",
        source_chunk_ids=["c", "a"],
    )

    def fake_extract_pdf_pages(pdf_path: object) -> list[PDFPage]:
        call_order.append("extract_pdf_pages")
        received["pdf_path"] = pdf_path
        return pages

    def fake_extract_purchase_order(
        actual_client: object,
        actual_pages: list[PDFPage],
        model: str,
    ) -> PurchaseOrder:
        call_order.append("extract_purchase_order")
        received["purchase_order_args"] = (actual_client, actual_pages, model)
        return purchase_order

    def fake_chunk_pages(actual_pages: list[PDFPage]) -> list[Chunk]:
        call_order.append("chunk_pages")
        received["chunk_pages"] = actual_pages
        return chunks

    def fake_embed_texts(
        actual_client: object,
        texts: list[str],
        model: str,
    ) -> list[list[float]]:
        call_order.append("embed_texts")
        received["embed_args"] = (actual_client, texts, model)
        return vectors

    def fake_top_k_chunks(
        actual_chunks: list[Chunk],
        chunk_embeddings: list[list[float]],
        query_embedding: list[float],
        k: int,
    ) -> list[SourceEvidence]:
        call_order.append("top_k_chunks")
        received["retrieval_args"] = (
            actual_chunks,
            chunk_embeddings,
            query_embedding,
            k,
        )
        return retrieved_sources

    def fake_answer_question(
        actual_client: object,
        question: str,
        sources: list[SourceEvidence],
        model: str,
    ) -> GroundedAnswer:
        call_order.append("answer_question")
        received["answer_args"] = (actual_client, question, sources, model)
        return grounded_answer

    monkeypatch.setattr(pipeline, "extract_pdf_pages", fake_extract_pdf_pages)
    monkeypatch.setattr(
        pipeline,
        "extract_purchase_order",
        fake_extract_purchase_order,
    )
    monkeypatch.setattr(pipeline, "chunk_pages", fake_chunk_pages)
    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(pipeline, "top_k_chunks", fake_top_k_chunks)
    monkeypatch.setattr(pipeline, "answer_question", fake_answer_question)

    result = pipeline.process_document(
        "sample.pdf",
        "Which chunks support the answer?",
        client,
        chat_model="test-chat-model",
        embedding_model="test-embedding-model",
    )

    assert isinstance(result, PipelineResult)
    assert result.purchase_order == purchase_order
    assert result.answer == grounded_answer.answer
    assert result.sources == [retrieved_sources[2], retrieved_sources[0]]
    assert result.sources[0] is retrieved_sources[2]
    assert result.sources[1] is retrieved_sources[0]
    assert call_order == [
        "extract_pdf_pages",
        "extract_purchase_order",
        "chunk_pages",
        "embed_texts",
        "top_k_chunks",
        "answer_question",
    ]
    assert received["pdf_path"] == "sample.pdf"
    purchase_order_args = received["purchase_order_args"]
    assert purchase_order_args[0] is client
    assert purchase_order_args[1] is pages
    assert purchase_order_args[2] == "test-chat-model"
    assert received["chunk_pages"] is pages

    embed_args = received["embed_args"]
    assert embed_args[0] is client
    assert embed_args[1] == [
        "Which chunks support the answer?",
        "Chunk A",
        "Chunk B",
        "Chunk C",
    ]
    assert embed_args[2] == "test-embedding-model"

    retrieval_args = received["retrieval_args"]
    assert retrieval_args[0] is chunks
    assert retrieval_args[1] == vectors[1:]
    assert retrieval_args[2] is vectors[0]
    assert retrieval_args[3] == 3

    answer_args = received["answer_args"]
    assert answer_args[0] is client
    assert answer_args[1] == "Which chunks support the answer?"
    assert answer_args[2] is retrieved_sources
    assert answer_args[3] == "test-chat-model"


def test_process_document_preserves_insufficiency_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    insufficient_answer = "I don't have enough information in the provided sources."
    pages = [PDFPage(page_number=1, text="Purchase Order")]
    chunks = [Chunk(chunk_id="a", page_number=1, text="Purchase Order")]
    retrieved_sources = [
        SourceEvidence(
            chunk_id="a",
            page_number=1,
            text="Purchase Order",
            score=0.2,
        )
    ]

    monkeypatch.setattr(pipeline, "extract_pdf_pages", lambda path: pages)
    monkeypatch.setattr(
        pipeline,
        "extract_purchase_order",
        lambda client, actual_pages, model: PurchaseOrder(po_number="PO-1001"),
    )
    monkeypatch.setattr(pipeline, "chunk_pages", lambda actual_pages: chunks)
    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda client, texts, model: [[1.0], [1.0]],
    )
    monkeypatch.setattr(
        pipeline,
        "top_k_chunks",
        lambda actual_chunks, embeddings, query, k: retrieved_sources,
    )
    monkeypatch.setattr(
        pipeline,
        "answer_question",
        lambda client, question, sources, model: GroundedAnswer(
            answer=insufficient_answer,
            source_chunk_ids=[],
        ),
    )

    result = pipeline.process_document(
        "sample.pdf",
        "What are the payment terms?",
        object(),
        chat_model="test-chat-model",
        embedding_model="test-embedding-model",
    )

    assert result.answer == insufficient_answer
    assert result.sources == []


def test_process_document_rejects_zero_chunks_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    pages = [PDFPage(page_number=1, text="Purchase Order")]

    def fake_extract_pdf_pages(path: object) -> list[PDFPage]:
        call_order.append("extract_pdf_pages")
        return pages

    def fake_extract_purchase_order(
        client: object,
        actual_pages: list[PDFPage],
        model: str,
    ) -> PurchaseOrder:
        call_order.append("extract_purchase_order")
        return PurchaseOrder(po_number="PO-1001")

    def fake_chunk_pages(actual_pages: list[PDFPage]) -> list[Chunk]:
        call_order.append("chunk_pages")
        return []

    def unexpected_call(*args: object, **kwargs: object) -> None:
        pytest.fail("Downstream primitive called after zero chunks")

    monkeypatch.setattr(pipeline, "extract_pdf_pages", fake_extract_pdf_pages)
    monkeypatch.setattr(
        pipeline,
        "extract_purchase_order",
        fake_extract_purchase_order,
    )
    monkeypatch.setattr(pipeline, "chunk_pages", fake_chunk_pages)
    monkeypatch.setattr(pipeline, "embed_texts", unexpected_call)
    monkeypatch.setattr(pipeline, "top_k_chunks", unexpected_call)
    monkeypatch.setattr(pipeline, "answer_question", unexpected_call)

    with pytest.raises(ValueError, match="No chunks available for retrieval"):
        pipeline.process_document(
            "sample.pdf",
            "What are the payment terms?",
            object(),
            chat_model="test-chat-model",
            embedding_model="test-embedding-model",
        )

    assert call_order == [
        "extract_pdf_pages",
        "extract_purchase_order",
        "chunk_pages",
    ]


def test_process_document_rejects_impossible_missing_evidence_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [PDFPage(page_number=1, text="Purchase Order")]
    chunks = [Chunk(chunk_id="a", page_number=1, text="Chunk A")]
    retrieved_sources = [
        SourceEvidence(chunk_id="a", page_number=1, text="Chunk A", score=0.8)
    ]

    monkeypatch.setattr(pipeline, "extract_pdf_pages", lambda path: pages)
    monkeypatch.setattr(
        pipeline,
        "extract_purchase_order",
        lambda client, actual_pages, model: PurchaseOrder(po_number="PO-1001"),
    )
    monkeypatch.setattr(pipeline, "chunk_pages", lambda actual_pages: chunks)
    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda client, texts, model: [[1.0], [1.0]],
    )
    monkeypatch.setattr(
        pipeline,
        "top_k_chunks",
        lambda actual_chunks, embeddings, query, k: retrieved_sources,
    )
    monkeypatch.setattr(
        pipeline,
        "answer_question",
        lambda client, question, sources, model: GroundedAnswer(
            answer="Unsupported fake result",
            source_chunk_ids=["c"],
        ),
    )

    with pytest.raises(
        ValueError,
        match="Grounded answer source ID was not retrieved: c",
    ):
        pipeline.process_document(
            "sample.pdf",
            "What are the payment terms?",
            object(),
            chat_model="test-chat-model",
            embedding_model="test-embedding-model",
        )


@pytest.mark.parametrize(
    ("failing_stage", "expected_calls", "error"),
    [
        ("extract_pdf_pages", ["extract_pdf_pages"], FileNotFoundError("missing")),
        (
            "extract_purchase_order",
            ["extract_pdf_pages", "extract_purchase_order"],
            RuntimeError("extraction failed"),
        ),
        (
            "chunk_pages",
            ["extract_pdf_pages", "extract_purchase_order", "chunk_pages"],
            ValueError("chunking failed"),
        ),
        (
            "embed_texts",
            [
                "extract_pdf_pages",
                "extract_purchase_order",
                "chunk_pages",
                "embed_texts",
            ],
            RuntimeError("embedding failed"),
        ),
        (
            "top_k_chunks",
            [
                "extract_pdf_pages",
                "extract_purchase_order",
                "chunk_pages",
                "embed_texts",
                "top_k_chunks",
            ],
            ValueError("retrieval failed"),
        ),
        (
            "answer_question",
            [
                "extract_pdf_pages",
                "extract_purchase_order",
                "chunk_pages",
                "embed_texts",
                "top_k_chunks",
                "answer_question",
            ],
            RuntimeError("answering failed"),
        ),
    ],
)
def test_process_document_propagates_first_stage_failure_and_stops(
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
    expected_calls: list[str],
    error: Exception,
) -> None:
    calls: list[str] = []
    pages = [PDFPage(page_number=1, text="Purchase Order")]
    chunks = [Chunk(chunk_id="a", page_number=1, text="Chunk A")]
    sources = [
        SourceEvidence(chunk_id="a", page_number=1, text="Chunk A", score=0.8)
    ]

    def stage(name: str, result: object):
        def fake(*args: object, **kwargs: object) -> object:
            calls.append(name)
            if name == failing_stage:
                raise error
            return result

        return fake

    monkeypatch.setattr(
        pipeline,
        "extract_pdf_pages",
        stage("extract_pdf_pages", pages),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_purchase_order",
        stage("extract_purchase_order", PurchaseOrder(po_number="PO-1001")),
    )
    monkeypatch.setattr(pipeline, "chunk_pages", stage("chunk_pages", chunks))
    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        stage("embed_texts", [[1.0], [1.0]]),
    )
    monkeypatch.setattr(
        pipeline,
        "top_k_chunks",
        stage("top_k_chunks", sources),
    )
    monkeypatch.setattr(
        pipeline,
        "answer_question",
        stage(
            "answer_question",
            GroundedAnswer(answer="Chunk A", source_chunk_ids=["a"]),
        ),
    )

    with pytest.raises(type(error)) as exc_info:
        pipeline.process_document(
            "sample.pdf",
            "What is stated?",
            object(),
            chat_model="test-chat-model",
            embedding_model="test-embedding-model",
        )

    assert exc_info.value is error
    assert calls == expected_calls


def test_process_document_end_to_end_offline(tmp_path: Path) -> None:
    pdf_path = tmp_path / "purchase-order.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    for lines in (
        (
            "Purchase Order: PO-1001",
            "Supplier: ACME Components",
            "Currency: USD",
        ),
        (
            "Payment Terms: Net 30",
            "Total Amount: 1250.00",
        ),
    ):
        y = 720
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 24
        pdf.showPage()
    pdf.save()

    expected_pages = extract_pdf_pages(pdf_path)
    expected_chunks = chunk_pages(expected_pages)
    assert [chunk.chunk_id for chunk in expected_chunks] == [
        "page-001-chunk-001",
        "page-002-chunk-001",
    ]

    expected_purchase_order = PurchaseOrder(
        po_number="PO-1001",
        supplier_name="ACME Components",
        payment_terms="Net 30",
        currency="USD",
        total_amount=Decimal("1250.00"),
    )

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def parse(self, **kwargs: object) -> SimpleNamespace:
            self.calls.append(kwargs)
            text_format = kwargs["text_format"]
            if text_format is PurchaseOrder:
                parsed = expected_purchase_order
            elif text_format is GroundedAnswer:
                parsed = GroundedAnswer(
                    answer="Net 30",
                    source_chunk_ids=["page-002-chunk-001"],
                )
            else:
                pytest.fail(f"Unexpected structured schema: {text_format}")
            return SimpleNamespace(output_parsed=parsed, output=[])

    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=2, embedding=[1.0, 0.0]),
                    SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                    SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                ]
            )

    client = SimpleNamespace(
        responses=FakeResponses(),
        embeddings=FakeEmbeddings(),
    )
    question = "What are the payment terms?"

    result = pipeline.process_document(
        pdf_path,
        question,
        client,
        chat_model="test-chat-model",
        embedding_model="test-embedding-model",
    )

    assert isinstance(result, PipelineResult)
    assert result.purchase_order == expected_purchase_order
    assert result.answer == "Net 30"
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "page-002-chunk-001"
    assert result.sources[0].page_number == 2
    assert result.sources[0].text == expected_chunks[1].text
    assert result.sources[0].text == expected_pages[1].text
    assert result.sources[0].score == pytest.approx(1.0)

    assert len(client.responses.calls) == 2
    assert [call["text_format"] for call in client.responses.calls] == [
        PurchaseOrder,
        GroundedAnswer,
    ]
    assert [call["model"] for call in client.responses.calls] == [
        "test-chat-model",
        "test-chat-model",
    ]

    assert client.embeddings.calls == [
        {
            "model": "test-embedding-model",
            "input": [question, *(chunk.text for chunk in expected_chunks)],
        }
    ]

    answer_input = client.responses.calls[1]["input"]
    assert answer_input == (
        f"Question:\n{question}\n\nSources:\n"
        f"--- Source: page-002-chunk-001 | Page: 2 ---\n"
        f"{expected_chunks[1].text}\n\n"
        f"--- Source: page-001-chunk-001 | Page: 1 ---\n"
        f"{expected_chunks[0].text}"
    )
    assert "score" not in answer_input.lower()

from decimal import Decimal
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseOutputRefusal

from isc.llm import (
    _GROUNDED_ANSWER_INSTRUCTIONS,
    _INSUFFICIENT_ANSWER,
    _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS,
    _format_pages,
    _format_sources,
    answer_question,
    embed_texts,
    extract_purchase_order,
)
from isc.models import (
    GroundedAnswer,
    LineItem,
    PDFPage,
    PurchaseOrder,
    SourceEvidence,
)


class FakeResponses:
    def __init__(
        self,
        parsed: GroundedAnswer | PurchaseOrder | None,
        *,
        output: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.parsed = parsed
        self.output = output or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.parsed, output=self.output)


class FakeEmbeddings:
    def __init__(
        self,
        data: list[object] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.data = data or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=self.data)


class FakeClient:
    def __init__(
        self,
        parsed: GroundedAnswer | PurchaseOrder | None = None,
        *,
        output: list[object] | None = None,
        error: Exception | None = None,
        embeddings: list[object] | None = None,
        embedding_error: Exception | None = None,
    ) -> None:
        self.responses = FakeResponses(parsed, output=output, error=error)
        self.embeddings = FakeEmbeddings(embeddings, error=embedding_error)


def test_grounded_answer_instructions_require_sources_only() -> None:
    instructions = " ".join(_GROUNDED_ANSWER_INSTRUCTIONS.lower().split())

    assert "only the provided sources" in instructions
    assert "outside or general knowledge" in instructions
    assert "infer unsupported facts" in instructions
    assert "invent facts" in instructions


def test_grounded_answer_instructions_preserve_exact_document_values() -> None:
    instructions = " ".join(_GROUNDED_ANSWER_INSTRUCTIONS.lower().split())

    for value_type in (
        "names",
        "dates",
        "quantities",
        "monetary amounts",
        "payment terms",
        "identifiers",
    ):
        assert value_type in instructions


def test_grounded_answer_instructions_define_insufficiency_contract() -> None:
    assert _INSUFFICIENT_ANSWER == (
        "I don't have enough information in the provided sources."
    )
    assert _INSUFFICIENT_ANSWER in _GROUNDED_ANSWER_INSTRUCTIONS
    assert "source_chunk_ids = []" in _GROUNDED_ANSWER_INSTRUCTIONS


def test_grounded_answer_instructions_restrict_ids_and_score_meaning() -> None:
    instructions = " ".join(_GROUNDED_ANSWER_INSTRUCTIONS.lower().split())

    assert "only supplied source chunk ids" in instructions
    assert "only ids that support the answer" in instructions
    assert "not factual or answer confidence" in instructions


def test_format_sources_formats_single_source_with_original_metadata() -> None:
    source = SourceEvidence(
        doc_id="po-001",
        chunk_id="page-001-chunk-002",
        page_number=1,
        text="Payment Terms: Net 30",
        score=0.92,
    )

    assert _format_sources([source]) == (
        "--- Source: page-001-chunk-002 | Page: 1 ---\n"
        "Payment Terms: Net 30"
    )


def test_format_sources_preserves_input_order_for_multiple_sources() -> None:
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-010-chunk-003",
            page_number=10,
            text="Ranked first",
            score=0.8,
        ),
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-002-chunk-001",
            page_number=2,
            text="Ranked second",
            score=0.9,
        ),
    ]

    formatted = _format_sources(sources)

    assert formatted.index("page-010-chunk-003") < formatted.index(
        "page-002-chunk-001"
    )
    assert formatted.index("Page: 10") < formatted.index("Page: 2")
    assert formatted.index("Ranked first") < formatted.index("Ranked second")


def test_format_sources_preserves_exact_whitespace_and_unicode() -> None:
    exact_text = "  Supplier: Müller Components\n\nPayment Terms: Net 30\n"
    source = SourceEvidence(
        doc_id="po-001",
        chunk_id="page-004-chunk-002",
        page_number=4,
        text=exact_text,
        score=0.5,
    )

    assert _format_sources([source]) == (
        "--- Source: page-004-chunk-002 | Page: 4 ---\n" + exact_text
    )


def test_format_sources_does_not_expose_retrieval_score() -> None:
    source = SourceEvidence(
        doc_id="po-001",
        chunk_id="page-003-chunk-001",
        page_number=3,
        text="Supplier: Example Components",
        score=0.731928475,
    )

    formatted = _format_sources([source])

    assert "0.731928475" not in formatted
    assert "score" not in formatted.lower()


def test_format_sources_rejects_empty_sources() -> None:
    with pytest.raises(ValueError, match="No sources provided for answering"):
        _format_sources([])


def test_answer_question_returns_structured_grounded_answer_in_one_call() -> None:
    expected = GroundedAnswer(
        answer="Net 30",
        source_chunk_ids=["page-001-chunk-002"],
    )
    client = FakeClient(parsed=expected)
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-002",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.91,
        )
    ]

    answer = answer_question(
        client,
        "What are the payment terms?",
        sources,
        model="test-chat-model",
    )

    assert answer == expected
    assert answer.answer == "Net 30"
    assert answer.source_chunk_ids == ["page-001-chunk-002"]
    assert client.responses.calls == [
        {
            "model": "test-chat-model",
            "instructions": _GROUNDED_ANSWER_INSTRUCTIONS,
            "input": (
                "Question:\nWhat are the payment terms?\n\nSources:\n"
                "--- Source: page-001-chunk-002 | Page: 1 ---\n"
                "Payment Terms: Net 30"
            ),
            "text_format": GroundedAnswer,
        }
    ]


def test_answer_question_preserves_question_and_ranked_source_context() -> None:
    client = FakeClient(
        parsed=GroundedAnswer(
            answer="Müller Components",
            source_chunk_ids=["page-010-chunk-003"],
        )
    )
    question = "  Which supplier(s)?  "
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-010-chunk-003",
            page_number=10,
            text="  Supplier: Müller Components\n",
            score=0.731928475,
        ),
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-002-chunk-001",
            page_number=2,
            text="Second supplier: Example Parts",
            score=0.99,
        ),
    ]

    answer_question(client, question, sources, model="test-chat-model")

    provider_input = client.responses.calls[0]["input"]
    assert provider_input == (
        f"Question:\n{question}\n\nSources:\n{_format_sources(sources)}"
    )
    assert provider_input.index("page-010-chunk-003") < provider_input.index(
        "page-002-chunk-001"
    )
    assert "0.731928475" not in provider_input
    assert len(client.responses.calls) == 1


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_answer_question_rejects_blank_question_without_provider_call(
    question: str,
) -> None:
    client = FakeClient()
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.5,
        )
    ]

    with pytest.raises(ValueError, match="Question must not be blank"):
        answer_question(client, question, sources, model="test-chat-model")

    assert client.responses.calls == []


@pytest.mark.parametrize("model", ["", "   ", "\n"])
def test_answer_question_rejects_blank_model_without_provider_call(
    model: str,
) -> None:
    client = FakeClient()
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.5,
        )
    ]

    with pytest.raises(ValueError, match="Model name must not be blank"):
        answer_question(client, "What are the terms?", sources, model=model)

    assert client.responses.calls == []


def test_answer_question_rejects_empty_sources_without_provider_call() -> None:
    client = FakeClient()

    with pytest.raises(ValueError, match="No sources provided for answering"):
        answer_question(client, "What are the terms?", [], model="test-chat-model")

    assert client.responses.calls == []


def test_answer_question_propagates_provider_exception_unchanged() -> None:
    provider_error = RuntimeError("answer provider unavailable")
    client = FakeClient(error=provider_error)
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.5,
        )
    ]

    with pytest.raises(RuntimeError) as exc_info:
        answer_question(
            client,
            "What are the terms?",
            sources,
            model="test-chat-model",
        )

    assert exc_info.value is provider_error
    assert len(client.responses.calls) == 1


def test_answer_question_rejects_unknown_source_id_without_retry() -> None:
    client = FakeClient(
        parsed=GroundedAnswer(
            answer="Net 30",
            source_chunk_ids=["page-999-chunk-999"],
        )
    )
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.5,
        )
    ]

    with pytest.raises(ValueError, match="unknown source ID: page-999-chunk-999"):
        answer_question(
            client,
            "What are the terms?",
            sources,
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_answer_question_deduplicates_source_ids_in_first_occurrence_order() -> None:
    client = FakeClient(
        parsed=GroundedAnswer(
            answer="Net 30",
            source_chunk_ids=[
                "page-002-chunk-001",
                "page-001-chunk-001",
                "page-002-chunk-001",
            ],
        )
    )
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Purchase order terms",
            score=0.8,
        ),
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-002-chunk-001",
            page_number=2,
            text="Payment Terms: Net 30",
            score=0.7,
        ),
    ]

    answer = answer_question(
        client,
        "What are the terms?",
        sources,
        model="test-chat-model",
    )

    assert answer.source_chunk_ids == [
        "page-002-chunk-001",
        "page-001-chunk-001",
    ]
    assert len(client.responses.calls) == 1


def test_answer_question_preserves_valid_source_id_order_and_answer_text() -> None:
    exact_answer = "  Net 30; delivery is FOB destination.\n"
    client = FakeClient(
        parsed=GroundedAnswer(
            answer=exact_answer,
            source_chunk_ids=["page-002-chunk-001", "page-001-chunk-001"],
        )
    )
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Delivery: FOB destination",
            score=-0.1,
        ),
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-002-chunk-001",
            page_number=2,
            text="Payment Terms: Net 30",
            score=0.9,
        ),
    ]

    answer = answer_question(
        client,
        "What are the terms?",
        sources,
        model="test-chat-model",
    )

    assert answer.answer == exact_answer
    assert answer.source_chunk_ids == [
        "page-002-chunk-001",
        "page-001-chunk-001",
    ]
    assert len(client.responses.calls) == 1


def test_answer_question_disambiguates_pooled_sources_sharing_a_bare_chunk_id() -> None:
    # Two different documents' page-1-chunk-1: under the old doc-unqualified chunk_id
    # format these would have been identical strings, and validating a returned ID against
    # {source.chunk_id for source in sources} would have silently accepted either one's
    # evidence for the other's ID. Doc-qualified chunk_ids make that collision impossible.
    client = FakeClient(
        parsed=GroundedAnswer(
            answer="4500123457",
            source_chunk_ids=["po-005:page-001-chunk-001"],
        )
    )
    sources = [
        SourceEvidence(
            doc_id="po-004",
            chunk_id="po-004:page-001-chunk-001",
            page_number=1,
            text="Part Number: 4500123456",
            score=0.4,
        ),
        SourceEvidence(
            doc_id="po-005",
            chunk_id="po-005:page-001-chunk-001",
            page_number=1,
            text="Part Number: 4500123457",
            score=0.9,
        ),
    ]

    answer = answer_question(
        client,
        "What is the part number?",
        sources,
        model="test-chat-model",
    )

    assert answer.source_chunk_ids == ["po-005:page-001-chunk-001"]
    assert answer.answer == "4500123457"


def test_answer_question_accepts_exact_insufficiency_without_source_ids() -> None:
    client = FakeClient(
        parsed=GroundedAnswer(
            answer=_INSUFFICIENT_ANSWER,
            source_chunk_ids=[],
        )
    )
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Supplier: Example Components",
            score=0.2,
        )
    ]

    answer = answer_question(
        client,
        "What are the terms?",
        sources,
        model="test-chat-model",
    )

    assert answer == GroundedAnswer(answer=_INSUFFICIENT_ANSWER, source_chunk_ids=[])
    assert len(client.responses.calls) == 1


def test_answer_question_rejects_insufficiency_with_source_ids() -> None:
    client = FakeClient(
        parsed=GroundedAnswer(
            answer=_INSUFFICIENT_ANSWER,
            source_chunk_ids=["page-001-chunk-001"],
        )
    )
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Supplier: Example Components",
            score=0.2,
        )
    ]

    with pytest.raises(ValueError, match="must not include source IDs"):
        answer_question(
            client,
            "What are the terms?",
            sources,
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_answer_question_rejects_supported_answer_without_source_ids() -> None:
    client = FakeClient(
        parsed=GroundedAnswer(answer="Net 30", source_chunk_ids=[]),
    )
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.8,
        )
    ]

    with pytest.raises(ValueError, match="must include at least one source ID"):
        answer_question(
            client,
            "What are the terms?",
            sources,
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_answer_question_rejects_missing_structured_output_without_retry() -> None:
    client = FakeClient()
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.8,
        )
    ]

    with pytest.raises(ValueError, match="Structured grounded answer was not returned"):
        answer_question(
            client,
            "What are the terms?",
            sources,
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_answer_question_reports_explicit_refusal_without_retry() -> None:
    refusal = ResponseOutputRefusal(type="refusal", refusal="Request declined")
    message = SimpleNamespace(type="message", content=[refusal])
    client = FakeClient(output=[message])
    sources = [
        SourceEvidence(
            doc_id="po-001",
            chunk_id="page-001-chunk-001",
            page_number=1,
            text="Payment Terms: Net 30",
            score=0.8,
        )
    ]

    with pytest.raises(ValueError, match="refused grounded answering.*Request declined"):
        answer_question(
            client,
            "What are the terms?",
            sources,
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_format_pages_adds_single_page_marker() -> None:
    text = "PURCHASE ORDER\nPO-1001"

    formatted = _format_pages([PDFPage(page_number=1, text=text)])

    assert "--- Page 1 ---" in formatted
    assert text in formatted


def test_format_pages_preserves_multiple_page_order() -> None:
    pages = [
        PDFPage(page_number=1, text="First page"),
        PDFPage(page_number=2, text="Second page"),
        PDFPage(page_number=3, text="Third page"),
    ]

    formatted = _format_pages(pages)

    markers = [formatted.index(f"--- Page {number} ---") for number in (1, 2, 3)]
    assert markers == sorted(markers)


def test_format_pages_uses_actual_page_numbers() -> None:
    pages = [
        PDFPage(page_number=2, text="Second page"),
        PDFPage(page_number=5, text="Fifth page"),
    ]

    formatted = _format_pages(pages)

    assert "--- Page 2 ---" in formatted
    assert "--- Page 5 ---" in formatted
    assert formatted.index("--- Page 2 ---") < formatted.index("--- Page 5 ---")


def test_format_pages_includes_blank_page_marker() -> None:
    formatted = _format_pages([PDFPage(page_number=2, text="")])

    assert formatted == "--- Page 2 ---\n"


def test_format_pages_rejects_empty_page_list() -> None:
    with pytest.raises(ValueError, match="At least one PDF page is required"):
        _format_pages([])


def test_format_pages_preserves_document_text() -> None:
    text = "  03/04/2026\nUSD 1,250.00\nABC Components Pte. Ltd.  \n"

    formatted = _format_pages([PDFPage(page_number=1, text=text)])

    assert text in formatted


def test_extraction_instructions_require_null_for_absent_fields() -> None:
    instructions = _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS.lower()

    assert "use null" in instructions
    assert "field is absent" in instructions


def test_extraction_instructions_prohibit_inference_and_invention() -> None:
    instructions = _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS.lower()

    assert "do not infer or invent" in instructions
    assert "supplier names" in instructions
    assert "quantities" in instructions
    assert "prices" in instructions


def test_extraction_instructions_prohibit_arithmetic() -> None:
    instructions = _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS.lower()

    assert "do not calculate missing values" in instructions
    assert "total_amount" in instructions
    assert "unit_price" in instructions
    assert "line totals" in instructions
    assert "currency from geography" in instructions


def test_extraction_instructions_preserve_ambiguous_dates() -> None:
    instructions = _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS.lower()

    assert "preserve ambiguous dates exactly as printed" in instructions
    assert "03/04/2026" in instructions


def test_extraction_instructions_prohibit_outside_knowledge() -> None:
    instructions = _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS.lower()

    assert "do not use outside or general knowledge" in instructions


def test_extraction_instructions_define_page_markers_as_metadata() -> None:
    instructions = _PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS.lower()

    assert "page markers are metadata" in instructions
    assert "not part of the purchase order" in instructions


def test_extract_purchase_order_returns_parsed_fields_and_line_items() -> None:
    line_item = LineItem(
        part_number="PART-001",
        description="Replacement filter",
        quantity=Decimal("2"),
        unit_price=Decimal("19.95"),
    )
    parsed = PurchaseOrder(
        po_number="PO-1001",
        po_date="03/04/2026",
        supplier_name="ABC Components Pte. Ltd.",
        payment_terms="Net 30",
        currency="USD",
        total_amount=Decimal("39.90"),
        line_items=[line_item],
    )
    client = FakeClient(parsed)

    result = extract_purchase_order(
        client,
        [PDFPage(page_number=1, text="Purchase Order PO-1001")],
        model="test-chat-model",
    )

    assert result is parsed
    assert isinstance(result, PurchaseOrder)
    assert result.po_number == "PO-1001"
    assert result.po_date == "03/04/2026"
    assert result.supplier_name == "ABC Components Pte. Ltd."
    assert result.payment_terms == "Net 30"
    assert result.currency == "USD"
    assert result.total_amount == Decimal("39.90")
    assert result.line_items == [line_item]


def test_extract_purchase_order_configures_structured_parse_call() -> None:
    client = FakeClient(PurchaseOrder(po_number="PO-1001"))

    extract_purchase_order(
        client,
        [PDFPage(page_number=1, text="Purchase Order PO-1001")],
        model="test-chat-model",
    )

    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "test-chat-model"
    assert request["text_format"] is PurchaseOrder
    assert "use null" in str(request["instructions"]).lower()
    assert "do not infer or invent" in str(request["instructions"]).lower()


def test_extract_purchase_order_sends_formatted_page_content_once() -> None:
    client = FakeClient(PurchaseOrder(po_number="PO-1001"))
    pages = [
        PDFPage(page_number=1, text="Purchase Order PO-1001"),
        PDFPage(page_number=2, text="Supplier ABC Components"),
    ]

    extract_purchase_order(client, pages, model="test-chat-model")

    assert len(client.responses.calls) == 1
    model_input = str(client.responses.calls[0]["input"])
    assert "--- Page 1 ---" in model_input
    assert "Purchase Order PO-1001" in model_input
    assert "--- Page 2 ---" in model_input
    assert "Supplier ABC Components" in model_input


@pytest.mark.parametrize("model", ["", "   ", "\n"])
def test_extract_purchase_order_rejects_blank_model_without_calling_provider(
    model: str,
) -> None:
    client = FakeClient(PurchaseOrder())

    with pytest.raises(ValueError, match="Model name must not be blank"):
        extract_purchase_order(
            client,
            [PDFPage(page_number=1, text="Purchase Order")],
            model=model,
        )

    assert client.responses.calls == []


def test_extract_purchase_order_rejects_empty_pages_without_calling_provider() -> None:
    client = FakeClient(PurchaseOrder())

    with pytest.raises(ValueError, match="At least one PDF page is required"):
        extract_purchase_order(client, [], model="test-chat-model")

    assert client.responses.calls == []


def test_extract_purchase_order_rejects_missing_parsed_output_without_retry() -> None:
    client = FakeClient()

    with pytest.raises(ValueError, match="Structured Purchase Order output was not returned"):
        extract_purchase_order(
            client,
            [PDFPage(page_number=1, text="Purchase Order")],
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_extract_purchase_order_reports_explicit_refusal_without_retry() -> None:
    refusal = ResponseOutputRefusal(type="refusal", refusal="Request declined")
    message = SimpleNamespace(type="message", content=[refusal])
    client = FakeClient(output=[message])

    with pytest.raises(ValueError, match="Model refused.*Request declined"):
        extract_purchase_order(
            client,
            [PDFPage(page_number=1, text="Purchase Order")],
            model="test-chat-model",
        )

    assert len(client.responses.calls) == 1


def test_extract_purchase_order_propagates_provider_exception_unchanged() -> None:
    provider_error = RuntimeError("provider unavailable")
    client = FakeClient(error=provider_error)

    with pytest.raises(RuntimeError) as exc_info:
        extract_purchase_order(
            client,
            [PDFPage(page_number=1, text="Purchase Order")],
            model="test-chat-model",
        )

    assert exc_info.value is provider_error
    assert len(client.responses.calls) == 1


def test_embed_single_text_returns_exact_vector() -> None:
    client = FakeClient(
        embeddings=[SimpleNamespace(index=0, embedding=[0.25, -0.5, 1.0])]
    )

    vectors = embed_texts(
        client,
        ["Purchase Order PO-1001"],
        model="test-embedding-model",
    )

    assert vectors == [[0.25, -0.5, 1.0]]


def test_embed_multiple_texts_forwards_one_request() -> None:
    texts = ["question", "chunk one", "chunk two"]
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            SimpleNamespace(index=1, embedding=[0.5, 0.5]),
            SimpleNamespace(index=2, embedding=[0.0, 1.0]),
        ]
    )

    vectors = embed_texts(client, texts, model="test-embedding-model")

    assert vectors == [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    assert client.embeddings.calls == [
        {"model": "test-embedding-model", "input": texts}
    ]


def test_embed_texts_restores_response_index_order() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=2, embedding=[2.0, 2.0]),
            SimpleNamespace(index=0, embedding=[0.0, 1.0]),
            SimpleNamespace(index=1, embedding=[1.0, 1.0]),
        ]
    )

    vectors = embed_texts(
        client,
        ["question", "chunk one", "chunk two"],
        model="test-embedding-model",
    )

    assert vectors == [[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
    assert len(client.embeddings.calls) == 1


def test_embed_empty_text_list_returns_without_provider_call() -> None:
    client = FakeClient()

    assert embed_texts(client, [], model="test-embedding-model") == []
    assert client.embeddings.calls == []


@pytest.mark.parametrize("model", ["", "   ", "\n"])
def test_embed_texts_rejects_blank_model_without_provider_call(model: str) -> None:
    client = FakeClient()

    with pytest.raises(ValueError, match="Embedding model name must not be blank"):
        embed_texts(client, ["Purchase Order"], model=model)

    assert client.embeddings.calls == []


@pytest.mark.parametrize("blank_text", ["", "   ", "\n"])
def test_embed_texts_rejects_blank_text_without_provider_call(
    blank_text: str,
) -> None:
    client = FakeClient()

    with pytest.raises(ValueError, match="Text to embed must not be blank"):
        embed_texts(
            client,
            ["question", blank_text, "chunk"],
            model="test-embedding-model",
        )

    assert client.embeddings.calls == []


def test_embed_texts_rejects_missing_response_index_after_one_call() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=0, embedding=[0.0]),
            SimpleNamespace(index=2, embedding=[2.0]),
        ]
    )

    with pytest.raises(ValueError, match="Embedding count does not match text count"):
        embed_texts(
            client,
            ["question", "chunk one", "chunk two"],
            model="test-embedding-model",
        )

    assert len(client.embeddings.calls) == 1


def test_embed_texts_rejects_duplicate_response_index_after_one_call() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=0, embedding=[0.0]),
            SimpleNamespace(index=1, embedding=[1.0]),
            SimpleNamespace(index=1, embedding=[2.0]),
        ]
    )

    with pytest.raises(ValueError, match="duplicate index"):
        embed_texts(
            client,
            ["question", "chunk one", "chunk two"],
            model="test-embedding-model",
        )

    assert len(client.embeddings.calls) == 1


def test_embed_texts_rejects_negative_response_index_after_one_call() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=-1, embedding=[0.0]),
            SimpleNamespace(index=1, embedding=[1.0]),
        ]
    )

    with pytest.raises(ValueError, match="invalid index"):
        embed_texts(
            client,
            ["question", "chunk"],
            model="test-embedding-model",
        )

    assert len(client.embeddings.calls) == 1


def test_embed_texts_rejects_out_of_range_response_index_after_one_call() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=0, embedding=[0.0]),
            SimpleNamespace(index=2, embedding=[1.0]),
        ]
    )

    with pytest.raises(ValueError, match="invalid index"):
        embed_texts(
            client,
            ["question", "chunk"],
            model="test-embedding-model",
        )

    assert len(client.embeddings.calls) == 1


def test_embed_texts_rejects_extra_response_vector_after_one_call() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=0, embedding=[0.0]),
            SimpleNamespace(index=1, embedding=[1.0]),
            SimpleNamespace(index=2, embedding=[2.0]),
        ]
    )

    with pytest.raises(ValueError, match="Embedding count does not match text count"):
        embed_texts(
            client,
            ["question", "chunk"],
            model="test-embedding-model",
        )

    assert len(client.embeddings.calls) == 1


def test_embed_texts_rejects_empty_vector_after_one_call() -> None:
    client = FakeClient(embeddings=[SimpleNamespace(index=0, embedding=[])])

    with pytest.raises(ValueError, match="finite values"):
        embed_texts(client, ["question"], model="test-embedding-model")

    assert len(client.embeddings.calls) == 1


def test_embed_texts_rejects_inconsistent_dimensions_after_one_call() -> None:
    client = FakeClient(
        embeddings=[
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            SimpleNamespace(index=1, embedding=[0.3, 0.4, 0.5]),
        ]
    )

    with pytest.raises(ValueError, match="same dimension"):
        embed_texts(
            client,
            ["question", "chunk"],
            model="test-embedding-model",
        )

    assert len(client.embeddings.calls) == 1


@pytest.mark.parametrize("component", [float("nan"), float("inf"), float("-inf")])
def test_embed_texts_rejects_non_finite_component_after_one_call(
    component: float,
) -> None:
    client = FakeClient(
        embeddings=[SimpleNamespace(index=0, embedding=[0.0, component])]
    )

    with pytest.raises(ValueError, match="finite values"):
        embed_texts(client, ["question"], model="test-embedding-model")

    assert len(client.embeddings.calls) == 1


def test_embed_texts_accepts_negative_zero_and_positive_components() -> None:
    client = FakeClient(
        embeddings=[SimpleNamespace(index=0, embedding=[-0.25, 0.0, 0.75])]
    )

    vectors = embed_texts(client, ["question"], model="test-embedding-model")

    assert vectors == [[-0.25, 0.0, 0.75]]
    assert len(client.embeddings.calls) == 1


def test_embed_texts_propagates_provider_exception_unchanged() -> None:
    provider_error = RuntimeError("embedding provider unavailable")
    client = FakeClient(embedding_error=provider_error)

    with pytest.raises(RuntimeError) as exc_info:
        embed_texts(client, ["question"], model="test-embedding-model")

    assert exc_info.value is provider_error
    assert len(client.embeddings.calls) == 1

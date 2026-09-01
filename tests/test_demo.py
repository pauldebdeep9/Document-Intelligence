from decimal import Decimal

import pytest

import demo
from isc.models import LineItem, PipelineResult, PurchaseOrder, SourceEvidence


@pytest.mark.parametrize("value", [None, "", "   ", "\n"])
def test_require_env_rejects_missing_or_blank_value(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    name = "POC_TEST_REQUIRED_VALUE"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        demo._require_env(name)


def test_require_env_returns_configured_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POC_TEST_REQUIRED_VALUE", "configured-value")

    assert demo._require_env("POC_TEST_REQUIRED_VALUE") == "configured-value"


def test_print_result_displays_purchase_order_answer_and_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exact_text = "  Payment Terms: Net 30\n"
    result = PipelineResult(
        purchase_order=PurchaseOrder(
            po_number="PO-1001",
            supplier_name="ACME Components",
            payment_terms="Net 30",
            currency="USD",
            total_amount=Decimal("1250.00"),
            line_items=[
                LineItem(
                    part_number="PART-1",
                    description="Component",
                    quantity=Decimal("2"),
                    unit_price=Decimal("10.50"),
                )
            ],
        ),
        answer="Net 30",
        sources=[
            SourceEvidence(
                chunk_id="page-002-chunk-001",
                page_number=2,
                text=exact_text,
                score=0.731928475,
            )
        ],
    )

    demo._print_result(result)

    output = capsys.readouterr().out
    for expected in (
        "Purchase Order",
        "PO Number: PO-1001",
        "Supplier: ACME Components",
        "PART-1",
        "Answer",
        "Net 30",
        "Sources",
        "Chunk: page-002-chunk-001",
        "Page: 2",
        exact_text,
        "Cosine retrieval similarity",
    ):
        assert expected in output
    assert "confidence" not in output.lower()


def test_print_result_displays_empty_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = PipelineResult(
        purchase_order=PurchaseOrder(po_number="PO-1001"),
        answer="I don't have enough information in the provided sources.",
        sources=[],
    )

    demo._print_result(result)

    output = capsys.readouterr().out
    assert result.answer in output
    assert "No supporting sources returned." in output


def test_main_forwards_operator_input_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (
        ("OPENAI_API_KEY", "test-api-key"),
        ("OPENAI_CHAT_MODEL", "test-chat-model"),
        ("OPENAI_EMBEDDING_MODEL", "test-embedding-model"),
    ):
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(demo, "load_dotenv", lambda: None)
    inputs = iter(["  sample.pdf  ", "  What are the payment terms?  "])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    client = object()
    client_api_keys: list[str] = []

    def fake_openai(*, api_key: str) -> object:
        client_api_keys.append(api_key)
        return client

    received: dict[str, object] = {}
    expected_result = PipelineResult(
        purchase_order=PurchaseOrder(po_number="PO-1001"),
        answer="Net 30",
        sources=[],
    )

    def fake_process_document(**kwargs: object) -> PipelineResult:
        received.update(kwargs)
        return expected_result

    displayed: list[PipelineResult] = []
    monkeypatch.setattr(demo, "OpenAI", fake_openai)
    monkeypatch.setattr(demo, "process_document", fake_process_document)
    monkeypatch.setattr(demo, "_print_result", displayed.append)

    demo.main(argv=[])

    assert client_api_keys == ["test-api-key"]
    assert received == {
        "pdf_path": "sample.pdf",
        "question": "  What are the payment terms?  ",
        "client": client,
        "chat_model": "test-chat-model",
        "embedding_model": "test-embedding-model",
    }
    assert displayed == [expected_result]


def test_main_uses_cli_args_without_prompting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (
        ("OPENAI_API_KEY", "test-api-key"),
        ("OPENAI_CHAT_MODEL", "test-chat-model"),
        ("OPENAI_EMBEDDING_MODEL", "test-embedding-model"),
    ):
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(demo, "load_dotenv", lambda: None)

    def fail_on_input(prompt: str) -> str:
        raise AssertionError("input() must not be called when CLI args are supplied")

    monkeypatch.setattr("builtins.input", fail_on_input)

    client = object()
    monkeypatch.setattr(demo, "OpenAI", lambda *, api_key: client)

    received: dict[str, object] = {}
    expected_result = PipelineResult(
        purchase_order=PurchaseOrder(po_number="PO-1001"),
        answer="Net 30",
        sources=[],
    )

    def fake_process_document(**kwargs: object) -> PipelineResult:
        received.update(kwargs)
        return expected_result

    displayed: list[PipelineResult] = []
    monkeypatch.setattr(demo, "process_document", fake_process_document)
    monkeypatch.setattr(demo, "_print_result", displayed.append)

    demo.main(argv=["--pdf", "sample.pdf", "--question", "What are the payment terms?"])

    assert received["pdf_path"] == "sample.pdf"
    assert received["question"] == "What are the payment terms?"
    assert displayed == [expected_result]


def test_main_falls_back_to_prompt_for_omitted_cli_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (
        ("OPENAI_API_KEY", "test-api-key"),
        ("OPENAI_CHAT_MODEL", "test-chat-model"),
        ("OPENAI_EMBEDDING_MODEL", "test-embedding-model"),
    ):
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(demo, "load_dotenv", lambda: None)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "What are the payment terms?"

    monkeypatch.setattr("builtins.input", fake_input)

    client = object()
    monkeypatch.setattr(demo, "OpenAI", lambda *, api_key: client)

    received: dict[str, object] = {}
    expected_result = PipelineResult(
        purchase_order=PurchaseOrder(po_number="PO-1001"),
        answer="Net 30",
        sources=[],
    )

    def fake_process_document(**kwargs: object) -> PipelineResult:
        received.update(kwargs)
        return expected_result

    monkeypatch.setattr(demo, "process_document", fake_process_document)
    monkeypatch.setattr(demo, "_print_result", lambda result: None)

    demo.main(argv=["--pdf", "sample.pdf"])

    assert prompts == ["Question: "]
    assert received["pdf_path"] == "sample.pdf"
    assert received["question"] == "What are the payment terms?"

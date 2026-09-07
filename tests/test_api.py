from collections.abc import Iterator
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError
from pypdf import PdfWriter
from pypdf.errors import PdfReadError
from reportlab.pdfgen import canvas

from isc import api
from isc.models import PipelineResult, PurchaseOrder, SourceEvidence
from isc.pdf import extract_pdf_pages

_PRIVATE_DETAIL = "OPENAI_API_KEY=sk-test-only /private/internal.pdf internal exception details"


@pytest.fixture(autouse=True)
def openai_constructor(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(api, "load_dotenv", lambda: None)
    constructor = MagicMock()
    monkeypatch.setattr(api, "OpenAI", constructor)
    return constructor


@pytest.fixture(autouse=True)
def mocked_pipeline(monkeypatch: pytest.MonkeyPatch) -> Mock:
    pipeline = Mock(
        return_value=PipelineResult(
            purchase_order=PurchaseOrder(
                po_number="PO-1001",
                payment_terms="Net 30",
                total_amount=Decimal("1250.00"),
            ),
            answer="Net 30",
            sources=[
                SourceEvidence(
                    doc_id="order",
                    chunk_id="order:page-001-chunk-001",
                    page_number=1,
                    text="  Payment Terms: Net 30\n",
                    score=0.9,
                )
            ],
        )
    )
    monkeypatch.setattr(api, "process_document", pipeline)
    return pipeline


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(api.app) as test_client:
        yield test_client


def test_index_serves_html_without_running_pipeline(
    client: TestClient, mocked_pipeline: Mock, openai_constructor: MagicMock
) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    mocked_pipeline.assert_not_called()
    openai_constructor.assert_not_called()


def test_index_contains_upload_question_and_result_ui(client: TestClient) -> None:
    page = client.get("/").text

    assert "<title>Document Intelligence</title>" in page
    assert "<h1>Document Intelligence</h1>" in page
    assert (
        '<input id="file" name="file" type="file" accept=".pdf,application/pdf" required>' in page
    )
    assert '<textarea id="question" name="question" rows="3" required>' in page
    assert '<button id="ask-button" type="submit">Ask</button>' in page
    assert 'id="status" role="status"' in page
    assert 'id="error" role="alert"' in page
    assert 'id="answer"' in page
    assert 'id="sources"' in page


def test_health_remains_available(client: TestClient, openai_constructor: MagicMock) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    openai_constructor.assert_not_called()


@pytest.mark.parametrize("question", ["", "   ", "\t\r\n"])
def test_ask_rejects_blank_question(
    client: TestClient,
    mocked_pipeline: Mock,
    openai_constructor: MagicMock,
    question: str,
) -> None:
    response = client.post(
        "/ask",
        files={"file": ("order.pdf", b"pdf bytes", "application/pdf")},
        data={"question": question},
    )

    assert response.status_code == 400
    mocked_pipeline.assert_not_called()
    openai_constructor.assert_not_called()


@pytest.mark.parametrize("filename", ["order.txt", "order.pdf.txt", "order"])
def test_ask_rejects_non_pdf_filename(
    client: TestClient,
    mocked_pipeline: Mock,
    openai_constructor: MagicMock,
    filename: str,
) -> None:
    response = client.post(
        "/ask",
        files={"file": (filename, b"pdf bytes", "application/pdf")},
        data={"question": "What are the payment terms?"},
    )

    assert response.status_code == 400
    mocked_pipeline.assert_not_called()
    openai_constructor.assert_not_called()


@pytest.mark.parametrize("filename", ["order.pdf", "ORDER.PDF", "../../order.pdf"])
def test_ask_forwards_upload_and_returns_pipeline_json_without_openai_calls(
    client: TestClient,
    mocked_pipeline: Mock,
    openai_constructor: MagicMock,
    filename: str,
) -> None:
    pdf_bytes = b"%PDF-1.4\nmock upload contents\n"
    question = "  What are the payment terms?  "

    def process_upload(**kwargs: object) -> PipelineResult:
        pdf_path = kwargs["pdf_path"]
        assert isinstance(pdf_path, Path)
        assert pdf_path.name == Path(filename).name
        assert pdf_path.suffix.lower() == ".pdf"
        assert pdf_path.read_bytes() == pdf_bytes
        return mocked_pipeline.return_value

    mocked_pipeline.side_effect = process_upload

    response = client.post(
        "/ask",
        files={"file": (filename, pdf_bytes, "application/pdf")},
        data={"question": question},
    )

    assert response.status_code == 200
    assert response.json() == mocked_pipeline.return_value.model_dump(mode="json")
    pdf_path = mocked_pipeline.call_args.kwargs["pdf_path"]
    mocked_pipeline.assert_called_once_with(
        pdf_path=pdf_path,
        question=question,
        client=openai_constructor.return_value.__enter__.return_value,
        chat_model="test-chat-model",
        embedding_model="test-embedding-model",
    )
    openai_constructor.assert_called_once_with(api_key="test-api-key")
    openai_constructor.return_value.__exit__.assert_called_once()
    assert not pdf_path.exists()
    assert not pdf_path.parent.exists()


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        pytest.param(
            ValueError(f"PDF could not be read: {_PRIVATE_DETAIL}"),
            400,
            "The PDF could not be read. Upload a valid, unencrypted PDF.",
            id="wrapped-pdf-error",
        ),
        pytest.param(
            PdfReadError(_PRIVATE_DETAIL),
            400,
            "The PDF could not be read. Upload a valid, unencrypted PDF.",
            id="pdf-read-error",
        ),
        pytest.param(
            ValueError("Encrypted PDFs are not supported"),
            400,
            "The PDF could not be read. Upload a valid, unencrypted PDF.",
            id="encrypted-pdf",
        ),
        pytest.param(
            ValueError("No native PDF text found; OCR is not supported by this PoC"),
            422,
            "No extractable text found. Upload a PDF containing selectable text.",
            id="no-text",
        ),
        pytest.param(
            ValueError("No chunks available for retrieval"),
            422,
            "No extractable text found. Upload a PDF containing selectable text.",
            id="no-chunks",
        ),
        pytest.param(
            OpenAIError(_PRIVATE_DETAIL),
            502,
            "The AI service could not complete the request.",
            id="openai-error",
        ),
        pytest.param(
            APIConnectionError(
                message=_PRIVATE_DETAIL, request=httpx.Request("POST", "https://example.invalid")
            ),
            502,
            "The AI service could not complete the request.",
            id="openai-connection-error",
        ),
        pytest.param(
            APITimeoutError(request=httpx.Request("POST", "https://example.invalid")),
            504,
            "The AI service timed out. Please try again later.",
            id="openai-timeout",
        ),
        pytest.param(
            RuntimeError(_PRIVATE_DETAIL), 500, "Unable to process the document.",
            id="unexpected-error",
        ),
        pytest.param(
            ValueError(_PRIVATE_DETAIL), 500, "Unable to process the document.",
            id="non-pdf-value-error",
        ),
        pytest.param(
            PermissionError(_PRIVATE_DETAIL), 500, "Unable to process the document.",
            id="local-io-error",
        ),
    ],
)
def test_ask_returns_safe_error_logs_exception_and_deletes_temporary_file(
    client: TestClient,
    mocked_pipeline: Mock,
    openai_constructor: MagicMock,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    def fail_processing(**kwargs: object) -> PipelineResult:
        pdf_path = kwargs["pdf_path"]
        assert isinstance(pdf_path, Path)
        assert pdf_path.is_file()
        raise error

    mocked_pipeline.side_effect = fail_processing

    response = client.post(
        "/ask",
        files={"file": ("order.pdf", b"pdf bytes", "application/pdf")},
        data={"question": "What are the payment terms?"},
    )

    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": detail}
    assert _PRIVATE_DETAIL not in response.text
    assert "Traceback" not in response.text
    assert any(
        record.name == "isc.api"
        and record.exc_info is not None
        and record.exc_info[1] is error
        and record.exc_info[2] is not None
        for record in caplog.records
    )
    mocked_pipeline.assert_called_once()
    pdf_path = mocked_pipeline.call_args.kwargs["pdf_path"]
    assert not pdf_path.exists()
    assert not pdf_path.parent.exists()
    openai_constructor.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("upstream_status", [401, 429, 500])
def test_ask_hides_upstream_status_and_response_details(
    client: TestClient, mocked_pipeline: Mock, upstream_status: int
) -> None:
    mocked_pipeline.side_effect = APIStatusError(
        _PRIVATE_DETAIL,
        response=httpx.Response(
            upstream_status, request=httpx.Request("POST", "https://example.invalid")
        ),
        body={"message": _PRIVATE_DETAIL},
    )

    response = client.post(
        "/ask",
        files={"file": ("order.pdf", b"pdf bytes", "application/pdf")},
        data={"question": "What are the payment terms?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "The AI service could not complete the request."}
    mocked_pipeline.assert_called_once()


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL"])
def test_ask_hides_missing_server_configuration(
    client: TestClient, mocked_pipeline: Mock, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv(name)

    response = client.post(
        "/ask",
        files={"file": ("order.pdf", b"pdf bytes", "application/pdf")},
        data={"question": "What are the payment terms?"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to process the document."}
    assert name not in response.text
    mocked_pipeline.assert_not_called()


@pytest.mark.parametrize(
    "document_kind", ["corrupt", "zero-bytes", "no-pages", "blank-page", "missing-contents"]
)
def test_ask_handles_actual_pdf_extraction_failures(
    client: TestClient, mocked_pipeline: Mock, document_kind: str
) -> None:
    if document_kind == "corrupt":
        content = b"not a PDF"
    elif document_kind == "zero-bytes":
        content = b""
    elif document_kind == "blank-page":
        output = BytesIO()
        pdf = canvas.Canvas(output)
        pdf.showPage()
        pdf.save()
        content = output.getvalue()
    else:
        writer = PdfWriter()
        if document_kind == "missing-contents":
            writer.add_blank_page(width=72, height=72)
        output = BytesIO()
        writer.write(output)
        content = output.getvalue()

    def extract_upload(**kwargs: object) -> PipelineResult:
        pdf_path = kwargs["pdf_path"]
        assert isinstance(pdf_path, Path)
        extract_pdf_pages(pdf_path)
        pytest.fail("PDF extraction should fail before any model call")

    mocked_pipeline.side_effect = extract_upload

    response = client.post(
        "/ask",
        files={"file": ("order.pdf", content, "application/pdf")},
        data={"question": "What are the payment terms?"},
    )

    if document_kind in ("corrupt", "zero-bytes"):
        assert response.status_code == 400
        assert response.json() == {
            "detail": "The PDF could not be read. Upload a valid, unencrypted PDF."
        }
    elif document_kind == "missing-contents":
        # The existing extractor raises KeyError for a page without a content stream.
        assert response.status_code == 500
        assert response.json() == {"detail": "Unable to process the document."}
    else:
        assert response.status_code == 422
        assert response.json() == {
            "detail": "No extractable text found. Upload a PDF containing selectable text."
        }
    pdf_path = mocked_pipeline.call_args.kwargs["pdf_path"]
    assert not pdf_path.exists()
    assert not pdf_path.parent.exists()

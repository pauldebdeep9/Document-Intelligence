"""Prompt preparation for the proof-of-concept LLM calls."""

import math
from collections.abc import Iterable
from typing import cast

from openai import OpenAI
from openai.types.responses import ParsedResponse

from isc.models import GroundedAnswer, PDFPage, PurchaseOrder, SourceEvidence

_PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS = """\
The supplied document is a Purchase Order.
Extract only information explicitly supported by the supplied PDF text.
Use null when a requested field is absent.
Do not infer or invent PO numbers, supplier names, site names, dates, payment terms,
currency, quantities, or prices.
Do not calculate missing values such as total_amount, unit_price, or line totals, and do
not infer currency from geography.
Preserve ambiguous dates exactly as printed, for example 03/04/2026.
Preserve supplier and site names as printed, apart from surrounding whitespace.
Include line items only when supported by the supplied text; do not create placeholder
line items.
Do not use outside or general knowledge.
Application-added page markers are metadata, not part of the Purchase Order.
"""

_INSUFFICIENT_ANSWER = "I don't have enough information in the provided sources."

_GROUNDED_ANSWER_INSTRUCTIONS = f"""\
Answer the question using only the provided sources. Do not use outside or general
knowledge, infer unsupported facts, invent facts, or fill gaps.
Preserve exact names, dates, quantities, monetary amounts, payment terms, and
identifiers where relevant.
Use only supplied source chunk IDs, and include only IDs that support the answer.
If the sources are insufficient, return exactly: {_INSUFFICIENT_ANSWER}
For that insufficiency answer, return source_chunk_ids = [].
Retrieval score is a ranking signal, not factual or answer confidence.
Application-added source markers and page numbers are metadata, not document content.
"""


def _format_pages(pages: list[PDFPage]) -> str:
    """Format PDF pages with deterministic markers while preserving their text."""
    if not pages:
        raise ValueError("At least one PDF page is required")

    return "\n\n".join(
        f"--- Page {page.page_number} ---\n{page.text}"
        for page in pages
    )


def _format_sources(sources: list[SourceEvidence]) -> str:
    """Format retrieved sources in supplied rank order without changing their text."""
    if not sources:
        raise ValueError("No sources provided for answering")

    return "\n\n".join(
        f"--- Source: {source.chunk_id} | Page: {source.page_number} ---\n"
        f"{source.text}"
        for source in sources
    )


def _response_refusal(
    response: ParsedResponse[GroundedAnswer] | ParsedResponse[PurchaseOrder],
) -> str | None:
    """Return the first explicit model refusal, if present."""
    for output in response.output:
        if output.type == "message":
            for content in output.content:
                if content.type == "refusal":
                    return content.refusal
    return None


def extract_purchase_order(
    client: OpenAI,
    pages: list[PDFPage],
    model: str,
) -> PurchaseOrder:
    """Extract a structured purchase order from native PDF page text."""
    if not model.strip():
        raise ValueError("Model name must not be blank")

    formatted_pages = _format_pages(pages)
    response = client.responses.parse(
        model=model,
        instructions=_PURCHASE_ORDER_EXTRACTION_INSTRUCTIONS,
        input=formatted_pages,
        text_format=PurchaseOrder,
    )
    if response.output_parsed is not None:
        return response.output_parsed

    refusal = _response_refusal(response)
    if refusal is not None:
        raise ValueError(f"Model refused Purchase Order extraction: {refusal}")

    raise ValueError("Structured Purchase Order output was not returned")


def answer_question(
    client: OpenAI,
    question: str,
    sources: list[SourceEvidence],
    model: str,
) -> GroundedAnswer:
    """Answer a question using only the supplied retrieved sources."""
    if not question.strip():
        raise ValueError("Question must not be blank")
    if not model.strip():
        raise ValueError("Model name must not be blank")

    formatted_sources = _format_sources(sources)
    response = client.responses.parse(
        model=model,
        instructions=_GROUNDED_ANSWER_INSTRUCTIONS,
        input=f"Question:\n{question}\n\nSources:\n{formatted_sources}",
        text_format=GroundedAnswer,
    )
    parsed = response.output_parsed
    if parsed is None:
        refusal = _response_refusal(response)
        if refusal is not None:
            raise ValueError(f"Model refused grounded answering: {refusal}")
        raise ValueError("Structured grounded answer was not returned")

    source_ids = list(dict.fromkeys(parsed.source_chunk_ids))
    allowed_source_ids = {source.chunk_id for source in sources}
    for source_id in source_ids:
        if source_id not in allowed_source_ids:
            raise ValueError(f"Grounded answer contains unknown source ID: {source_id}")

    if parsed.answer == _INSUFFICIENT_ANSWER:
        if source_ids:
            raise ValueError("Insufficiency answer must not include source IDs")
    elif not source_ids:
        raise ValueError("Grounded answer must include at least one source ID")

    return GroundedAnswer(answer=parsed.answer, source_chunk_ids=source_ids)


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Create finite, consistently sized embeddings in input-text order."""
    if not model.strip():
        raise ValueError("Embedding model name must not be blank")
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Text to embed must not be blank")

    response = client.embeddings.create(model=model, input=texts)
    if len(response.data) != len(texts):
        raise ValueError("Embedding count does not match text count")

    vectors_by_index: dict[int, list[float]] = {}
    dimension: int | None = None
    for item in response.data:
        index = getattr(item, "index", None)
        if type(index) is not int or not 0 <= index < len(texts):
            raise ValueError("Embedding response contains an invalid index")
        if index in vectors_by_index:
            raise ValueError("Embedding response contains a duplicate index")

        raw_vector = cast(Iterable[float], getattr(item, "embedding", None))
        try:
            vector = [float(component) for component in raw_vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding vector must contain numeric values") from exc

        if not vector or not all(math.isfinite(component) for component in vector):
            raise ValueError("Embedding vector must contain finite values")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("Embedding vectors must have the same dimension")

        vectors_by_index[index] = vector

    if set(vectors_by_index) != set(range(len(texts))):
        raise ValueError("Embedding response indexes do not match input texts")

    return [vectors_by_index[index] for index in range(len(texts))]

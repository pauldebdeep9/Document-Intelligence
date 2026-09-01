"""Minimal end-to-end orchestration for the Document Intelligence proof of concept."""

from pathlib import Path

from openai import OpenAI

from isc.chunking import chunk_pages
from isc.llm import answer_question, embed_texts, extract_purchase_order
from isc.models import PipelineResult
from isc.pdf import extract_pdf_pages
from isc.retrieval import top_k_chunks


def process_document(
    pdf_path: str | Path,
    question: str,
    client: OpenAI,
    chat_model: str,
    embedding_model: str,
) -> PipelineResult:
    """Process a PDF and return structured extraction plus grounded evidence."""
    pages = extract_pdf_pages(pdf_path)
    purchase_order = extract_purchase_order(client, pages, chat_model)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No chunks available for retrieval")

    texts = [question, *(chunk.text for chunk in chunks)]
    vectors = embed_texts(client, texts, embedding_model)
    query_embedding = vectors[0]
    chunk_embeddings = vectors[1:]

    retrieved_sources = top_k_chunks(
        chunks,
        chunk_embeddings,
        query_embedding,
        k=3,
    )
    grounded_answer = answer_question(
        client,
        question,
        retrieved_sources,
        chat_model,
    )

    evidence_by_id = {source.chunk_id: source for source in retrieved_sources}
    try:
        resolved_sources = [
            evidence_by_id[source_id]
            for source_id in grounded_answer.source_chunk_ids
        ]
    except KeyError as exc:
        raise ValueError(f"Grounded answer source ID was not retrieved: {exc.args[0]}") from exc

    return PipelineResult(
        purchase_order=purchase_order,
        answer=grounded_answer.answer,
        sources=resolved_sources,
    )

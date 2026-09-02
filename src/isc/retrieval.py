"""Small, provider-independent retrieval helpers for the proof of concept."""

import math
from collections.abc import Sequence

from isc.models import Chunk, SourceEvidence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity for two finite, non-zero vectors."""
    if not a or not b:
        raise ValueError("Vectors must not be empty")
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("Vector components must be finite")

    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Vectors must have non-zero magnitude")

    dot_product = sum(left * right for left, right in zip(a, b, strict=True))
    return dot_product / (magnitude_a * magnitude_b)


def top_k_chunks(
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    query_embedding: list[float],
    k: int = 3,
) -> list[SourceEvidence]:
    """Rank chunks by cosine similarity while preserving stable score ties."""
    if not chunks:
        raise ValueError("No chunks available for retrieval")
    if len(chunks) != len(chunk_embeddings):
        raise ValueError("Chunk and embedding counts must match")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scored_chunks = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in zip(chunks, chunk_embeddings, strict=True)
    ]
    scored_chunks.sort(key=lambda item: item[1], reverse=True)

    return [
        SourceEvidence(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            text=chunk.text,
            score=score,
        )
        for chunk, score in scored_chunks[:k]
    ]

"""Reciprocal Rank Fusion.

Chosen over score normalisation because dense cosine and BM25 live on
incomparable scales, and any normalisation constant is a hyperparameter that
quietly rots. RRF only needs rank order.
"""

from __future__ import annotations

from isc.models.chunk import ScoredChunk


def rrf(result_lists: list[list[ScoredChunk]], k: int = 60,
        top_n: int = 8) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, sc in enumerate(results):
            cid = sc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            prev = best.get(cid)
            best[cid] = ScoredChunk(
                chunk=sc.chunk,
                score=scores[cid],
                dense_score=sc.dense_score or (prev.dense_score if prev else None),
                lexical_score=sc.lexical_score or (prev.lexical_score if prev else None),
            )
    ordered = sorted(best.values(), key=lambda s: -scores[s.chunk.id])[:top_n]
    return [
        ScoredChunk(chunk=s.chunk, score=scores[s.chunk.id], dense_score=s.dense_score,
                    lexical_score=s.lexical_score, rank=i)
        for i, s in enumerate(ordered)
    ]

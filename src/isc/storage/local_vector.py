"""Brute-force vector + BM25-ish lexical store with mandatory pre-filtering.

Correct, inspectable, and fast enough to five figures of chunks. Swapping in
Azure AI Search means translating `_permitted` into an OData filter over an
`acl_terms` Collection(Edm.String) field — the semantics are already right.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from isc.common.errors import AclViolation
from isc.common.tracing import span
from isc.models.acl import Principal
from isc.models.chunk import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_/\.]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LocalVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._tokens: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        if path.exists():
            self.load()

    # -- write -------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for c in chunks:
            # Belt and braces: Chunk already validates this, but the index is the
            # last place a permissionless record could enter the system.
            if not c.acl.allow_terms:
                raise AclViolation(f"chunk {c.id} has no allow_terms at index time")

        arr = np.asarray(vectors, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        self._vectors = arr if self._vectors is None else np.vstack([self._vectors, arr])
        self._chunks.extend(chunks)
        for c in chunks:
            counts = Counter(_tokenize(c.text))
            self._tokens.append(counts)
            self._df.update(counts.keys())

    # -- read --------------------------------------------------------------
    def _permitted(
        self, principal: Principal, filters: dict[str, str] | None
    ) -> np.ndarray:
        """Boolean mask of chunks this principal may read. Computed BEFORE ranking."""
        terms = principal.terms()
        mask = np.zeros(len(self._chunks), dtype=bool)
        for i, c in enumerate(self._chunks):
            if not principal.may_read(c.acl):
                continue
            if filters and any(c.filters.get(k) != v for k, v in filters.items()):
                continue
            mask[i] = True
        return mask

    def search(
        self,
        query_vector: Sequence[float],
        principal: Principal,
        *,
        k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []
        with span("vector.search", k=k, principal=principal.id):
            mask = self._permitted(principal, filters)
            if not mask.any():
                return []
            q = np.asarray(query_vector, dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-9
            sims = self._vectors @ q
            sims[~mask] = -np.inf
            idx = np.argsort(-sims)[:k]
            return [
                ScoredChunk(
                    chunk=self._chunks[i], score=float(sims[i]),
                    dense_score=float(sims[i]), rank=rank,
                )
                for rank, i in enumerate(idx)
                if math.isfinite(sims[i])
            ]

    def search_lexical(
        self,
        query: str,
        principal: Principal,
        *,
        k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """BM25. Matters more than it looks: part numbers, PO numbers and CAS
        numbers are exactly what dense retrieval is worst at."""
        if not self._chunks:
            return []
        with span("lexical.search", k=k, principal=principal.id):
            mask = self._permitted(principal, filters)
            terms = _tokenize(query)
            n = len(self._chunks)
            avgdl = sum(sum(t.values()) for t in self._tokens) / max(n, 1)
            k1, b = 1.5, 0.75
            scored: list[tuple[float, int]] = []
            for i in range(n):
                if not mask[i]:
                    continue
                counts = self._tokens[i]
                dl = sum(counts.values())
                s = 0.0
                for t in terms:
                    f = counts.get(t, 0)
                    if not f:
                        continue
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
                if s > 0:
                    scored.append((s, i))
            scored.sort(reverse=True)
            return [
                ScoredChunk(chunk=self._chunks[i], score=s, lexical_score=s, rank=r)
                for r, (s, i) in enumerate(scored[:k])
            ]

    def count(self) -> int:
        return len(self._chunks)

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump(
                {
                    "chunks": [c.model_dump(mode="json") for c in self._chunks],
                    "vectors": self._vectors,
                    "tokens": self._tokens,
                    "df": self._df,
                },
                fh,
            )

    def load(self) -> None:
        with self.path.open("rb") as fh:
            state = pickle.load(fh)
        self._chunks = [Chunk.model_validate(c) for c in state["chunks"]]
        self._vectors = state["vectors"]
        self._tokens = state["tokens"]
        self._df = state["df"]

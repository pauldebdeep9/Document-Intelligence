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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from isc.common.errors import AclViolation, ChunkSettingsMismatch
from isc.common.tracing import span
from isc.models.acl import Principal
from isc.models.chunk import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_/\.]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class UpsertResult:
    inserted: int = 0
    replaced: int = 0


class LocalVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        # Source of truth for mutation. id -> (chunk, normalized vector, token
        # counts). self._chunks/_vectors/_tokens/_df below are index-aligned
        # views rebuilt wholesale from this dict on every add()/remove_document()
        # -- see _rebuild_indexes for why that (not incremental patching) is
        # what keeps df from ever going stale on a replace or removal.
        self._by_id: dict[str, tuple[Chunk, np.ndarray, Counter[str]]] = {}
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._tokens: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        # Set by the first add() call that supplies one; see
        # settings_fingerprint's own docstring for what this guards against.
        self._settings_fingerprint: str | None = None
        if path.exists():
            self.load()

    def _rebuild_indexes(self) -> None:
        """Recompute the search-facing, index-aligned structures (including df)
        from self._by_id wholesale rather than patching them in place. A
        replace or removal that only patched df incrementally is exactly the
        kind of bookkeeping that goes quietly wrong (a stale count for a term
        that no longer appears anywhere skews idf across the whole index, and
        nothing surfaces it); recomputing from the current chunk set can't
        drift because there is no persisted delta to get wrong."""
        self._chunks = []
        self._tokens = []
        self._df = Counter()
        vectors: list[np.ndarray] = []
        for chunk, vec, counts in self._by_id.values():
            self._chunks.append(chunk)
            self._tokens.append(counts)
            self._df.update(counts.keys())
            vectors.append(vec)
        self._vectors = np.vstack(vectors) if vectors else None

    # -- write -------------------------------------------------------------
    def add(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        settings_fingerprint: str | None = None,
    ) -> UpsertResult:
        """Upserts by chunk id. Chunk ids are content-addressed (a hash of
        document id, ordinal and text -- see common/ids.py's chunk_id), so a
        chunk whose text or position changed gets a new id and lands as an
        insert; a chunk re-embedded unchanged lands as a byte-identical
        replace. This is what makes re-running the index pipeline over the
        same documents idempotent instead of duplicating every chunk."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for c in chunks:
            # Belt and braces: Chunk already validates this, but the index is the
            # last place a permissionless record could enter the system.
            if not c.acl.allow_terms:
                raise AclViolation(f"chunk {c.id} has no allow_terms at index time")

        if settings_fingerprint is not None:
            if self._settings_fingerprint is None:
                self._settings_fingerprint = settings_fingerprint
            elif settings_fingerprint != self._settings_fingerprint:
                raise ChunkSettingsMismatch(
                    f"store at {self.path} was built with chunk settings "
                    f"{self._settings_fingerprint!r}; this add() used "
                    f"{settings_fingerprint!r} -- re-chunking under different "
                    "settings into the same store silently mixes incompatible "
                    "chunk boundaries. Start a fresh store or match the settings."
                )

        arr = np.asarray(vectors, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9

        result = UpsertResult()
        for c, vec in zip(chunks, arr, strict=True):
            if c.id in self._by_id:
                result.replaced += 1
            else:
                result.inserted += 1
            self._by_id[c.id] = (c, vec, Counter(_tokenize(c.text)))
        self._rebuild_indexes()
        return result

    def remove_document(
        self, document_id: str, *, keep_ids: frozenset[str] = frozenset()
    ) -> int:
        """Removes chunks belonging to `document_id`, except any whose id is
        in keep_ids. Upsert by id alone leaves orphans behind when a document
        is re-chunked into FEWER chunks than it had before -- ordinals/ids no
        longer produced by the new chunking are simply never touched by add().

        Call this AFTER add()-ing a document's freshly computed chunks, with
        keep_ids set to those chunks' ids -- not before. add()'s own
        replace-vs-insert bookkeeping (see UpsertResult) depends on a
        document's previous chunk ids still being present in the store when
        add() runs; removing them first would make every upsert look like a
        fresh insert even when nothing changed, and would also mean a failure
        inside add() (fingerprint mismatch, a bad chunk) leaves that
        document's prior chunks already gone with nothing put back. Calling
        this after, scoped to exactly the ids add() did NOT just write, keeps
        both the reporting and the failure mode correct. Returns the number
        of chunks removed."""
        to_remove = [
            cid for cid, (c, _, _) in self._by_id.items()
            if c.document_id == document_id and cid not in keep_ids
        ]
        for cid in to_remove:
            del self._by_id[cid]
        if to_remove:
            self._rebuild_indexes()
        return len(to_remove)

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

    def settings_fingerprint(self) -> str | None:
        """None means either an empty store or one built before this guard
        existed -- both legitimate, so callers that care must check
        explicitly rather than treat None as "matches everything"."""
        return self._settings_fingerprint

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
                    "settings_fingerprint": self._settings_fingerprint,
                },
                fh,
            )

    def load(self) -> None:
        with self.path.open("rb") as fh:
            state = pickle.load(fh)
        chunks = [Chunk.model_validate(c) for c in state["chunks"]]
        vectors = state["vectors"]
        tokens = state["tokens"]
        # get(): stores saved before this guard existed have no key at all.
        self._settings_fingerprint = state.get("settings_fingerprint")
        # Rebuild _by_id (the mutation source of truth) from the persisted
        # parallel arrays, then derive _chunks/_vectors/_tokens/_df from it --
        # df is never trusted from disk, only ever recomputed from tokens, so
        # a store saved by a pre-upsert version self-heals on load rather than
        # carrying forward whatever was pickled. A duplicate id from an old,
        # pre-upsert store (back when add() could not replace) collapses to
        # last-one-wins here, same as it would from a fresh add().
        self._by_id = {}
        for chunk, vec, counts in zip(
            chunks, vectors if vectors is not None else [], tokens, strict=True
        ):
            self._by_id[chunk.id] = (chunk, vec, counts)
        self._rebuild_indexes()

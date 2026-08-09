"""Stage 5: question + principal -> ranked, permitted chunks.

The Principal is a required positional argument at every level. There is no
overload that retrieves without one.
"""

from __future__ import annotations

from isc.common.config import Settings
from isc.common.tracing import span
from isc.llm.ports import ChatModel, EmbeddingModel
from isc.models.acl import Principal
from isc.models.chunk import ScoredChunk
from isc.retrieve.fusion import rrf
from isc.storage.local_vector import LocalVectorStore


class Retriever:
    def __init__(self, store: LocalVectorStore, embedder: EmbeddingModel,
                 chat: ChatModel, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._chat = chat
        self._s = settings

    def retrieve(self, question: str, principal: Principal) -> list[ScoredChunk]:
        with span("retrieve", principal=principal.id):
            queries = self.rewrite(question)
            filters = self.infer_filters(question)
            lists: list[list[ScoredChunk]] = []
            for q in queries:
                vec = self._embedder.embed([q])[0]
                lists.append(self._store.search(
                    vec, principal, k=self._s.retrieval.top_k_dense, filters=filters))
                lists.append(self._store.search_lexical(
                    q, principal, k=self._s.retrieval.top_k_lexical, filters=filters))
            return rrf(lists, self._s.retrieval.rrf_k, self._s.retrieval.final_k)

    def rewrite(self, question: str) -> list[str]:
        """Original + paraphrases. Always keeps the original: rewrites lose the
        exact part numbers that BM25 depends on."""
        raise NotImplementedError("first slice: return [question]; week 2: LLM rewrite")

    def infer_filters(self, question: str) -> dict[str, str]:
        """'What did we pay Fastenal on PO 4500123' -> {po_number: 4500123}.

        Deterministic regex first, model only for the residue. A model that
        hallucinates a filter silently empties the result set.
        """
        raise NotImplementedError("week 2: regex over PO/part/supplier patterns")

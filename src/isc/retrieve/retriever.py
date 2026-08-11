"""Stage 5: question + principal -> ranked, permitted chunks.

The Principal is a required positional argument at every level. There is no
overload that retrieves without one.
"""

from __future__ import annotations

import re

from isc.common.config import Settings
from isc.common.tracing import span
from isc.extract.validators import PART_NUMBER, PO_NUMBER
from isc.llm.ports import ChatModel, EmbeddingModel
from isc.models.acl import Principal
from isc.models.chunk import ScoredChunk
from isc.retrieve.fusion import rrf
from isc.storage.local_vector import LocalVectorStore

# PO_NUMBER/PART_NUMBER are anchored (^...$) for validating an already-isolated
# extracted value, not for scanning a sentence. Reused as-is (never
# redefined) by testing each whitespace/punctuation-delimited token from the
# question against them, rather than loosening the anchors -- a second,
# looser copy of either pattern is exactly the duplication the WBS says not
# to write.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


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
        exact part numbers that BM25 depends on.

        First slice: no LLM rewrite. The WBS defers it, and P1-08's gold does
        not need it -- every gold question is already phrased plainly enough
        for dense + lexical search directly."""
        return [question]

    def infer_filters(self, question: str) -> dict[str, str]:
        """'What did we pay Fastenal on PO 4500123' -> {po_number: 4500123}.

        Deterministic regex only, no model: a model that hallucinates a
        filter silently empties the result set exactly the way a bad regex
        match would, for no offsetting benefit at this corpus size.

        PO numbers become an exact-match filter -- Chunk.filters carries
        po_number (see chunker.py's filters_from_record()), so this is a
        real, enforceable narrowing.

        Part numbers are matched with PART_NUMBER (imported from
        extract/validators.py, not redefined) but deliberately NOT added to
        the returned filters: no chunk carries a part_number key -- a table
        chunk covers many rows, each with its own part, so there is no single
        value a chunk-level exact filter could hold. Filtering on it would
        zero out every chunk regardless of whether the part is actually in
        the corpus, which is a worse failure than not filtering at all:
        search_lexical's own BM25 scoring already surfaces part numbers well
        from the query text (see its docstring), and a broken exact filter on
        top would silently suppress that, not just decline to narrow it.

        Supplier is never inferred, for a different reason than the part
        number gap: "Kestrel Industrial" resolves to one vendor code and
        silently drops the other, which would empty half of exactly the ambiguous
        case this retriever is supposed to preserve for the answer stage
        (config/prompts/answer/grounded_answer.v1.md already instructs the
        model to state disagreement and cite both sources when it sees
        chunks from both). A confidently narrow wrong answer erodes trust
        faster than no filter at all.

        A filter that matches nothing must return empty, never fall back to
        unfiltered -- see LocalVectorStore._permitted(), which already has no
        such fallback path. This function only has to avoid handing it a
        filter key nothing can ever match.
        """
        filters: dict[str, str] = {}
        for token in _TOKEN.findall(question):
            if PO_NUMBER.match(token):
                filters["po_number"] = token
            elif PART_NUMBER.match(token):
                pass  # detected, deliberately not filtered on -- see above
        return filters

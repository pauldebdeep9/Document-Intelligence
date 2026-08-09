"""Stage 6: grounded answering with citation binding and abstention.

Order of operations is deliberate:
  1. retrieve (already ACL-filtered)
  2. abstain early if support is weak — cheaper and safer than generating then checking
  3. generate with numbered context blocks
  4. bind citations: every claim must map to a retrieved chunk id
  5. if binding fails, abstain rather than emit an uncited answer

Step 5 is the one people skip. An answer that cannot be traced to a source is
indistinguishable from a fabrication, and in a supply-chain context it will be
acted on.
"""

from __future__ import annotations

from isc.common.config import Settings, load_prompt
from isc.common.confidence import Confidence, Signal
from isc.common.tracing import span
from isc.llm.ports import ChatModel, Message
from isc.models.acl import Principal
from isc.models.answer import AbstentionReason, Answer, Citation
from isc.models.chunk import ScoredChunk
from isc.retrieve.retriever import Retriever


class AnswerOrchestrator:
    def __init__(self, retriever: Retriever, chat: ChatModel, settings: Settings) -> None:
        self._retriever = retriever
        self._chat = chat
        self._s = settings

    def ask(self, question: str, principal: Principal) -> Answer:
        with span("answer.ask", principal=principal.id):
            hits = self._retriever.retrieve(question, principal)
            if not hits:
                return Answer.abstain(question, AbstentionReason.NO_RESULTS)
            if hits[0].score < self._s.retrieval.min_support_score:
                return Answer.abstain(question, AbstentionReason.LOW_SUPPORT)

            draft = self._generate(question, hits)
            citations = self._bind_citations(draft, hits)
            if not citations:
                return Answer.abstain(question, AbstentionReason.UNGROUNDED_DRAFT)

            return Answer(
                question=question,
                text=draft,
                citations=citations,
                supporting=hits,
                confidence=Confidence.independent(
                    Confidence.of(Signal.RETRIEVAL, min(hits[0].score, 1.0), "top hit"),
                    Confidence.of(Signal.AGREEMENT, min(len(citations) / 3, 1.0),
                                  f"{len(citations)} citations"),
                ),
            )

    def _generate(self, question: str, hits: list[ScoredChunk]) -> str:
        raise NotImplementedError("first slice: numbered context blocks + grounded prompt")

    def _bind_citations(self, draft: str, hits: list[ScoredChunk]) -> list[Citation]:
        """Parse [n] markers back to chunk ids and drop any that do not resolve."""
        raise NotImplementedError("first slice: regex [\\d+] -> hits[n-1]")

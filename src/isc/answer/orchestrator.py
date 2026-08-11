"""Stage 6: grounded answering with citation binding and abstention.

Order of operations is deliberate:
  1. retrieve (already ACL-filtered)
  2. abstain early if support is weak — cheaper and safer than generating then checking
  3. generate with numbered context blocks
  4. bind citations: every claim must map to a retrieved chunk id
  5. if binding fails, abstain rather than emit an uncited answer
  6. verify attribution: a cited claim's named entities must appear in the
     chunk(s) it cites — binding only checks the marker resolves to a real
     chunk, not that the sentence describes what is in it

Step 5 is the one people skip. An answer that cannot be traced to a source is
indistinguishable from a fabrication, and in a supply-chain context it will be
acted on. Step 6 exists because step 5 is not sufficient: a citation marker
can resolve to a real, permitted chunk while the sentence next to it names
the wrong supplier — found live in P1-07's sample run, not hypothesised.
"""

from __future__ import annotations

from isc.answer.citations import bind_citations, verify_attribution
from isc.common.config import Settings, load_prompt
from isc.common.confidence import Confidence, Signal
from isc.common.tracing import span
from isc.extract.masters import supplier_ids_by_name
from isc.llm.ports import ChatModel, Message
from isc.models.acl import Principal
from isc.models.answer import AbstentionReason, Answer, Citation
from isc.models.chunk import ScoredChunk
from isc.retrieve.retriever import Retriever

# Must match config/prompts/answer/grounded_answer.v1.md's rule 2 exactly --
# that prompt instructs the model to reply with this string and nothing
# else when the context does not contain the answer. Checked before citation
# binding even runs: parsing [n] markers out of a refusal is pointless work,
# and "no markers found" would land on UNGROUNDED_DRAFT anyway -- checking
# explicitly here just skips the wasted regex pass and names the real cause.
_INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class AnswerOrchestrator:
    def __init__(self, retriever: Retriever, chat: ChatModel, settings: Settings) -> None:
        self._retriever = retriever
        self._chat = chat
        self._s = settings
        # Loaded once per orchestrator, not per ask(): the master list is
        # small and static within a run, same reasoning as
        # extract/masters.py's own @lru_cache on the file read this calls
        # through to.
        self._supplier_ids = supplier_ids_by_name(settings.paths.data / "masters")

    def ask(self, question: str, principal: Principal) -> Answer:
        with span("answer.ask", principal=principal.id):
            hits = self._retriever.retrieve(question, principal)
            if not hits:
                return Answer.abstain(question, AbstentionReason.NO_RESULTS)
            if hits[0].score < self._s.retrieval.min_support_score:
                return Answer.abstain(question, AbstentionReason.LOW_SUPPORT)

            # hits is passed to BOTH calls below, unmodified and unreordered
            # in between -- _bind_citations() resolves marker [n] to
            # hits[n-1], the exact list _generate() numbered its context
            # blocks from. A re-fetch or a re-sort of hits anywhere between
            # these two lines makes every citation resolve to a chunk id
            # that looks valid and is wrong. See
            # test_ask_binds_citations_against_the_same_hits_list_used_for_generation.
            draft = self._generate(question, hits)
            if draft.strip() == _INSUFFICIENT_CONTEXT:
                # The model's own admission it found nothing -- never passed
                # through as answer text, and recorded under its own reason
                # (INSUFFICIENT_CONTEXT), not UNGROUNDED_DRAFT: this is a
                # model that read the context and correctly declined, not
                # one that produced an uncited draft. P1-09 needs the two
                # distinguishable -- the first is a PASS on the unanswerable
                # slice, the second is a suspected-fabrication FAILURE on
                # any slice. user_facing_reason()/Answer.abstain()'s text
                # stay identical between them either way, same pattern as
                # NO_PERMITTED_RESULTS vs NO_RESULTS -- only the recorded
                # reason differs.
                return Answer.abstain(question, AbstentionReason.INSUFFICIENT_CONTEXT)

            citations = self._bind_citations(draft, hits)
            if not citations:
                return Answer.abstain(question, AbstentionReason.UNGROUNDED_DRAFT)

            # Binding above only checked that [n] resolves to a chunk that
            # EXISTS. This checks that the sentence citing it describes
            # what is IN it -- a marker pointing at a real, permitted chunk
            # is not evidence the prose next to it is true. Checked against
            # Chunk.filters, not raw chunk.text (see docs/adr/0009): a
            # table/footer chunk never repeats a document's header fields
            # in its text, but carries the same filters as every other
            # chunk of that document. unverifiable entities (metadata
            # genuinely absent from the source) do not block the answer;
            # only a real mismatch does. On any mismatch the whole draft is
            # discarded, not just the offending sentence: a draft that
            # misattributes one fact is not trustworthy on the others it
            # happened to get right.
            if verify_attribution(draft, hits, self._supplier_ids).mismatches:
                return Answer.abstain(question, AbstentionReason.ATTRIBUTION_MISMATCH)

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
        """One model call: grounded_answer.v1.md as the system prompt, the
        question + numbered context blocks as the user message.

        Each block's header is its citation_label() verbatim, not a
        paraphrase -- that is what carries line_range into the prompt, so a
        block from a split table reads as "chk pp.2 (rows 350-420)" rather
        than "the table", and blocks are numbered [1]..[n] in hits' own
        order (already ranked by retrieve()'s RRF fusion), so a citation
        marker the model emits maps back to hits[n-1] with no reordering
        step in between for _bind_citations() to get wrong.

        Returns the raw draft text, unexamined -- binding [n] markers to
        chunk ids and deciding whether the draft is grounded is
        _bind_citations()'s job, kept separate so a binding bug can never be
        mistaken for a generation bug.
        """
        prompt = load_prompt("answer/grounded_answer.v1.md")
        blocks = "\n\n".join(
            f"[{i}] {sc.chunk.citation_label()}\n{sc.chunk.text}"
            for i, sc in enumerate(hits, start=1)
        )
        user = f"Question: {question}\n\nContext:\n\n{blocks}"
        result = self._chat.complete([Message.system(prompt), Message.user(user)])
        return result.text

    def _bind_citations(self, draft: str, hits: list[ScoredChunk]) -> list[Citation]:
        """Parse [n] markers back to chunk ids and drop any that do not
        resolve. Thin wrapper -- see answer/citations.py's bind_citations()
        for the actual parsing (bare/adjacent/grouped markers, dedup,
        out-of-range handling), kept as a free function so it is testable
        without an AnswerOrchestrator instance."""
        return bind_citations(draft, hits)

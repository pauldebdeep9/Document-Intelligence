"""Answer objects. Abstention is a first-class outcome, not an error path."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from isc.common.confidence import Confidence
from isc.models.chunk import ScoredChunk


class AbstentionReason(StrEnum):
    NO_RESULTS = "no_results"                  # nothing matched
    NO_PERMITTED_RESULTS = "no_permitted"      # matched, but not for this principal
    LOW_SUPPORT = "low_support"                # retrieved, but weakly relevant
    CONFLICTING_SOURCES = "conflicting"        # sources disagree
    # The model read the context and explicitly said the answer is not in
    # it -- correct behaviour on an unanswerable question, and a PASS at
    # P1-09. Distinct from UNGROUNDED_DRAFT below: that one is a model that
    # DID produce an answer and could not be tied to any source, a suspected
    # fabrication and a FAILURE regardless of question class. Collapsing the
    # two would make abstention precision on the unanswerable slice
    # unmeasurable -- P1-09 could not tell "abstained correctly" from "tried
    # to fabricate and got caught".
    INSUFFICIENT_CONTEXT = "insufficient_context"
    UNGROUNDED_DRAFT = "ungrounded"            # draft failed citation binding
    # A cited claim names a supplier or PO number that does not appear in
    # any chunk that claim cites -- the citation marker resolves to a real,
    # permitted chunk, so UNGROUNDED_DRAFT's check (does [n] exist) passes;
    # what fails is that the PROSE does not match what is IN the chunk. A
    # confidently wrong supplier attribution is worse than an abstention in
    # a supply-chain context, so the whole answer is discarded, not just the
    # offending sentence -- see answer/citations.py's verify_attribution().
    ATTRIBUTION_MISMATCH = "attribution_mismatch"


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    label: str
    quote: str = ""
    page_start: int = 1
    page_end: int = 1


class Answer(BaseModel):
    """Either `text` with citations, or an abstention. Never both.

    Note on NO_PERMITTED_RESULTS: it is recorded internally for eval and audit,
    but the user-facing message must be indistinguishable from NO_RESULTS.
    Telling someone a document exists that they cannot read is itself a leak.
    """

    question: str
    text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    supporting: list[ScoredChunk] = Field(default_factory=list)
    run_id: str = ""

    @classmethod
    def abstain(cls, question: str, reason: AbstentionReason) -> "Answer":
        return cls(
            question=question,
            abstained=True,
            abstention_reason=reason,
            text="I could not find supporting information for that in the documents "
                 "available to you.",
        )

    def user_facing_reason(self) -> str:
        if self.abstention_reason is AbstentionReason.NO_PERMITTED_RESULTS:
            return AbstentionReason.NO_RESULTS.value
        return self.abstention_reason.value if self.abstention_reason else ""

"""Gold-set schema: retrieval anchors, extraction ground truth, and anchor resolvability.

Retrieval gold is keyed on text anchors, not chunk IDs. A chunk ID like
`page-003-chunk-002` is a function of chunk_size and overlap, so gold keyed on chunk IDs is
invalidated by any chunking change — precisely when measurement matters most. An `Anchor`
instead names a verbatim span of page text plus the page it appears on; at eval time an
anchor is satisfied if its exact text is a substring of at least one retrieved chunk's text.

For a `RetrievalGold` with multiple anchors (e.g. a "cross_page" question whose evidence
spans two pages), every anchor must be satisfied for the item to count as a retrieval
success — one anchor per required piece of evidence, not "any one of these".

`question_class == "absent"` with `anchors == []` is itself the assertion that the correct
answer is the pipeline's insufficiency string (`isc.llm`'s "I don't have enough information
in the provided sources."): there is no evidence to anchor because none should be found. This
schema does not duplicate that literal string anywhere; a future metrics/runner session owns
interpreting "absent" against the pipeline's actual insufficiency constant.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.corpus.generate import PDF_DIR
from isc.chunking import chunk_pages
from isc.models import PurchaseOrder
from isc.pdf import extract_pdf_pages

QuestionClass = Literal["header_field", "line_item", "cross_page", "absent", "near_duplicate"]


class Anchor(BaseModel):
    """A verbatim span of page text, plus the page it appears on."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str

    @model_validator(mode="after")
    def _text_is_non_empty(self) -> "Anchor":
        if not self.text.strip():
            raise ValueError("Anchor text must not be blank")
        return self


class RetrievalGold(BaseModel):
    """One retrieval question against one document, with the anchors that answer it."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    doc_id: str
    question: str
    question_class: QuestionClass
    anchors: list[Anchor]

    @model_validator(mode="after")
    def _anchors_empty_only_when_absent(self) -> "RetrievalGold":
        if not self.anchors and self.question_class != "absent":
            raise ValueError(
                f"{self.question_id}: anchors may only be empty when "
                f"question_class == 'absent', got {self.question_class!r}"
            )
        return self


class ExtractionGold(BaseModel):
    """The expected structured extraction for one document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    expected: PurchaseOrder


class ChunkingConfig(BaseModel):
    """The chunking configuration a goldset's anchors were validated against."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int
    overlap: int


class GoldItem(BaseModel):
    """One document's extraction gold plus its retrieval gold questions."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    extraction: ExtractionGold
    retrieval: list[RetrievalGold]

    @model_validator(mode="after")
    def _child_doc_ids_match(self) -> "GoldItem":
        if self.extraction.doc_id != self.doc_id:
            raise ValueError(
                f"{self.doc_id}: extraction.doc_id {self.extraction.doc_id!r} does not match"
            )
        mismatched = [r.question_id for r in self.retrieval if r.doc_id != self.doc_id]
        if mismatched:
            raise ValueError(f"{self.doc_id}: retrieval doc_id mismatch for {mismatched}")
        return self


class GoldSet(BaseModel):
    """The full gold set: every document's ground truth, and how it was validated."""

    model_config = ConfigDict(extra="forbid")

    version: str
    items: list[GoldItem]
    chunking: ChunkingConfig

    @model_validator(mode="after")
    def _doc_ids_and_question_ids_are_unique(self) -> "GoldSet":
        doc_ids = [item.doc_id for item in self.items]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("GoldSet items contain duplicate doc_ids")

        question_ids = [
            retrieval.question_id for item in self.items for retrieval in item.retrieval
        ]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("GoldSet retrieval items contain duplicate question_ids")
        return self


def validate_anchors(
    goldset: GoldSet,
    chunk_size: int,
    overlap: int,
    pdf_dir: Path = PDF_DIR,
) -> None:
    """Raise if any anchor's text is not a substring of a same-page chunk at this config."""
    for item in goldset.items:
        pages = extract_pdf_pages(pdf_dir / f"{item.doc_id}.pdf")
        chunks = chunk_pages(pages, doc_id=item.doc_id, chunk_size=chunk_size, overlap=overlap)
        for retrieval in item.retrieval:
            for anchor in retrieval.anchors:
                page_chunks = [c for c in chunks if c.page_number == anchor.page_number]
                if not any(anchor.text in chunk.text for chunk in page_chunks):
                    raise ValueError(
                        f"Unresolvable anchor in question {retrieval.question_id!r} "
                        f"(doc {item.doc_id}, page {anchor.page_number}, "
                        f"chunk_size={chunk_size}, overlap={overlap}): {anchor.text!r} is not "
                        "a substring of any chunk on that page"
                    )

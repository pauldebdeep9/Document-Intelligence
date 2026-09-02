"""Minimal domain models for the Document Intelligence proof of concept."""

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator

_StructuredDecimal = Annotated[Decimal, WithJsonSchema({"type": "number"})]


def _require_non_blank_doc_id(value: str) -> str:
    if not value.strip():
        raise ValueError("doc_id must not be blank")
    return value


class PDFPage(BaseModel):
    """Native text extracted from one PDF page."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str


class LineItem(BaseModel):
    """One purchase order line item."""

    model_config = ConfigDict(extra="forbid")

    part_number: str | None = None
    description: str | None = None
    quantity: _StructuredDecimal | None = None
    unit_price: _StructuredDecimal | None = None


class PurchaseOrder(BaseModel):
    """Small structured representation of a purchase order."""

    model_config = ConfigDict(extra="forbid")

    po_number: str | None = None
    po_date: str | None = None
    supplier_name: str | None = None
    ship_to_site: str | None = None
    payment_terms: str | None = None
    currency: str | None = None
    total_amount: _StructuredDecimal | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class Chunk(BaseModel):
    """An unscored, page-local text chunk.

    chunk_id is globally unique across a corpus, not just within one document: it embeds
    doc_id (f"{doc_id}:page-{page:03d}-chunk-{n:03d}"). doc_id is also its own field so
    provenance is a structured comparison, not something parsed back out of the ID string.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_id: str
    page_number: int = Field(ge=1)
    text: str

    _validate_doc_id = field_validator("doc_id")(_require_non_blank_doc_id)


class SourceEvidence(BaseModel):
    """A retrieved chunk and its cosine similarity score."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_id: str
    page_number: int = Field(ge=1)
    text: str
    score: float

    _validate_doc_id = field_validator("doc_id")(_require_non_blank_doc_id)


class GroundedAnswer(BaseModel):
    """An answer and the retrieved chunk IDs that support it."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Final structured extraction and grounded answer result."""

    model_config = ConfigDict(extra="forbid")

    purchase_order: PurchaseOrder
    answer: str
    sources: list[SourceEvidence] = Field(default_factory=list)

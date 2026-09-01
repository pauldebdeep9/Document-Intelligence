"""Minimal domain models for the Document Intelligence proof of concept."""

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

_StructuredDecimal = Annotated[Decimal, WithJsonSchema({"type": "number"})]


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
    """An unscored, page-local text chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_number: int = Field(ge=1)
    text: str


class SourceEvidence(BaseModel):
    """A retrieved chunk and its cosine similarity score."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_number: int = Field(ge=1)
    text: str
    score: float


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

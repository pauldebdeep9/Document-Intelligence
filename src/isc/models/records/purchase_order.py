"""Purchase order: the walking-skeleton document type.

Chosen first because it exercises every hard part in miniature — header key/value
pairs, a line-item table, currency and date normalisation, and two fields
(supplier, part number) that can be corroborated against master data.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from isc.models.document import DocType
from isc.models.records.base import ExtractedField, ExtractionRecord, register


# --- what the model is asked for -----------------------------------------

class POLineRaw(BaseModel):
    line_number: int | None = None
    part_number: str | None = None
    description: str | None = None
    quantity: float | None = None
    unit_of_measure: str | None = None
    unit_price: float | None = None
    extended_price: float | None = None
    promised_date: str | None = None   # ISO 8601; normalised downstream


class PurchaseOrderRaw(BaseModel):
    """Flat and nullable by design. Null means 'not in the document'."""

    po_number: str | None = None
    po_date: str | None = None
    supplier_name: str | None = None
    supplier_id: str | None = None
    ship_to_site: str | None = None
    incoterms: str | None = None
    payment_terms: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    buyer_contact: str | None = None
    lines: list[POLineRaw] = Field(default_factory=list)


# --- what the pipeline stores --------------------------------------------

class POLine(BaseModel):
    line_number: ExtractedField[int] = Field(default_factory=ExtractedField[int])
    part_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    description: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    quantity: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    unit_of_measure: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    unit_price: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    extended_price: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    promised_date: ExtractedField[date] = Field(default_factory=ExtractedField[date])


@register
class PurchaseOrder(ExtractionRecord):
    doc_type = DocType.PURCHASE_ORDER
    raw_model = PurchaseOrderRaw

    po_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    po_date: ExtractedField[date] = Field(default_factory=ExtractedField[date])
    supplier_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    supplier_id: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    ship_to_site: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    incoterms: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    payment_terms: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    currency: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    total_amount: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    buyer_contact: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    lines: list[POLine] = Field(default_factory=list)

    @classmethod
    def required_fields(cls) -> set[str]:
        return {"po_number", "supplier_name", "po_date"}

    def line_total_agrees(self, tolerance: Decimal = Decimal("0.01")) -> bool | None:
        """Arithmetic cross-check. Feeds Signal.AGREEMENT on total_amount:
        a total that reconciles against the lines is evidence both were read right."""
        if not self.present_total() or not self.lines:
            return None
        summed = sum(
            (ln.extended_price.value for ln in self.lines if ln.extended_price.present),
            Decimal(0),
        )
        assert self.total_amount.value is not None
        return abs(summed - self.total_amount.value) <= tolerance

    def present_total(self) -> bool:
        return self.total_amount.present

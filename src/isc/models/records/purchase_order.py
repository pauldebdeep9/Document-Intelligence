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

    def extended_price_agrees(self, tolerance: Decimal = Decimal("0.01")) -> bool | None:
        """Per-line cross-check: printed extended price vs quantity x unit
        price. None (not just False) when extended_price was not printed --
        the prompt instructs the model to return null rather than compute a
        value itself, so an absent extended price is a correct read, not a
        check that failed."""
        if not (self.quantity.present and self.unit_price.present and self.extended_price.present):
            return None
        assert self.quantity.value is not None and self.unit_price.value is not None
        assert self.extended_price.value is not None
        computed = self.quantity.value * self.unit_price.value
        return abs(computed - self.extended_price.value) <= tolerance


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
        """Arithmetic cross-check. Feeds Signal.AGREEMENT on total_amount: a
        total that reconciles against quantity x unit_price for every line is
        evidence both were read right.

        Sums quantity x unit_price, NOT the printed extended price: 4/20
        corpus documents omit the extended-price column entirely while still
        printing a true total, so summing extendeds there yields zero and
        fires a false conflict on a fifth of the corpus. Requires every line
        to have both quantity and unit_price before summing anything -- a
        partial sum compared against the full total would misreport a gap in
        extraction as an arithmetic disagreement, which is a different claim.
        """
        if not self.present_total() or not self.lines:
            return None
        if not all(ln.quantity.present and ln.unit_price.present for ln in self.lines):
            return None
        summed = Decimal(0)
        for ln in self.lines:
            assert ln.quantity.value is not None and ln.unit_price.value is not None
            summed += ln.quantity.value * ln.unit_price.value
        assert self.total_amount.value is not None
        return abs(summed - self.total_amount.value) <= tolerance

    def present_total(self) -> bool:
        return self.total_amount.present

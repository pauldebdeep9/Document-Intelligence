"""Invoice. Second doc type — added once the PO slice is green end to end.

Shares supplier/currency/total with PurchaseOrder, which is what makes the
three-way match (PO <-> ASN/receipt <-> invoice) reachable from P1 into P2.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from isc.models.document import DocType
from isc.models.records.base import ExtractedField, ExtractionRecord, register


class InvoiceRaw(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = None
    po_number: str | None = None
    supplier_name: str | None = None
    supplier_tax_id: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    payment_terms: str | None = None
    due_date: str | None = None


@register
class Invoice(ExtractionRecord):
    doc_type = DocType.INVOICE
    raw_model = InvoiceRaw

    invoice_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    invoice_date: ExtractedField[date] = Field(default_factory=ExtractedField[date])
    po_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    supplier_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    supplier_tax_id: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    currency: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    subtotal: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    tax_amount: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    total_amount: ExtractedField[Decimal] = Field(default_factory=ExtractedField[Decimal])
    payment_terms: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    due_date: ExtractedField[date] = Field(default_factory=ExtractedField[date])

    @classmethod
    def required_fields(cls) -> set[str]:
        return {"invoice_number", "supplier_name", "total_amount"}

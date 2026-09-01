from decimal import Decimal

import pytest
from pydantic import ValidationError

from isc.models import (
    Chunk,
    GroundedAnswer,
    LineItem,
    PDFPage,
    PipelineResult,
    PurchaseOrder,
    SourceEvidence,
)


def test_pdf_page_accepts_first_page_and_empty_text() -> None:
    page = PDFPage(page_number=1, text="")

    assert page.page_number == 1
    assert page.text == ""


def test_pdf_page_rejects_zero_page_number() -> None:
    with pytest.raises(ValidationError):
        PDFPage(page_number=0, text="Purchase order")


def test_line_item_defaults_optional_fields_to_none() -> None:
    item = LineItem()

    assert item.part_number is None
    assert item.description is None
    assert item.quantity is None
    assert item.unit_price is None


def test_line_item_preserves_decimal_values() -> None:
    item = LineItem(quantity=Decimal("2.5"), unit_price=Decimal("19.95"))

    assert item.quantity == Decimal("2.5")
    assert isinstance(item.quantity, Decimal)
    assert item.unit_price == Decimal("19.95")
    assert isinstance(item.unit_price, Decimal)


def test_purchase_order_defaults_optional_fields_and_line_items() -> None:
    purchase_order = PurchaseOrder()

    assert purchase_order.po_number is None
    assert purchase_order.po_date is None
    assert purchase_order.supplier_name is None
    assert purchase_order.ship_to_site is None
    assert purchase_order.payment_terms is None
    assert purchase_order.currency is None
    assert purchase_order.total_amount is None
    assert purchase_order.line_items == []


def test_purchase_order_instances_have_independent_line_item_lists() -> None:
    first = PurchaseOrder()
    second = PurchaseOrder()

    first.line_items.append(LineItem(part_number="PART-001"))

    assert len(first.line_items) == 1
    assert second.line_items == []


def test_purchase_order_accepts_nested_line_item_and_preserves_values() -> None:
    item = LineItem(
        part_number="PART-001",
        description="Replacement filter",
        quantity=Decimal("3"),
        unit_price=Decimal("12.50"),
    )
    purchase_order = PurchaseOrder(
        po_date="03/04/2026",
        total_amount=Decimal("37.50"),
        line_items=[item],
    )

    assert purchase_order.po_date == "03/04/2026"
    assert purchase_order.total_amount == Decimal("37.50")
    assert isinstance(purchase_order.total_amount, Decimal)
    assert purchase_order.line_items == [item]


def test_purchase_order_decimal_fields_use_supported_number_schema() -> None:
    schema = PurchaseOrder.model_json_schema()

    decimal_fields = (
        schema["$defs"]["LineItem"]["properties"]["quantity"],
        schema["$defs"]["LineItem"]["properties"]["unit_price"],
        schema["properties"]["total_amount"],
    )
    expected_types = {"number", "null"}

    for field_schema in decimal_fields:
        assert {variant["type"] for variant in field_schema["anyOf"]} == expected_types
        assert all("pattern" not in variant for variant in field_schema["anyOf"])


def test_purchase_order_decimal_fields_parse_provider_numbers_as_decimal() -> None:
    purchase_order = PurchaseOrder.model_validate(
        {
            "total_amount": 37.5,
            "line_items": [
                {
                    "quantity": 3,
                    "unit_price": 12.5,
                }
            ],
        }
    )

    assert purchase_order.total_amount == Decimal("37.5")
    assert isinstance(purchase_order.total_amount, Decimal)
    assert purchase_order.line_items[0].quantity == Decimal("3")
    assert isinstance(purchase_order.line_items[0].quantity, Decimal)
    assert purchase_order.line_items[0].unit_price == Decimal("12.5")
    assert isinstance(purchase_order.line_items[0].unit_price, Decimal)


def test_chunk_requires_one_based_page_number() -> None:
    chunk = Chunk(chunk_id="page-001-chunk-001", page_number=1, text="PO number 42")

    assert chunk.page_number == 1

    with pytest.raises(ValidationError):
        Chunk(chunk_id="page-000-chunk-001", page_number=0, text="PO number 42")


def test_source_evidence_preserves_retrieval_values() -> None:
    source = SourceEvidence(
        chunk_id="page-002-chunk-003",
        page_number=2,
        text="Payment terms: Net 30",
        score=0.875,
    )

    assert source.chunk_id == "page-002-chunk-003"
    assert source.page_number == 2
    assert source.text == "Payment terms: Net 30"
    assert source.score == 0.875


def test_grounded_answer_instances_have_independent_source_id_lists() -> None:
    first = GroundedAnswer(answer="Net 30")
    second = GroundedAnswer(answer="Insufficient information")

    first.source_chunk_ids.append("page-001-chunk-001")

    assert first.source_chunk_ids == ["page-001-chunk-001"]
    assert second.source_chunk_ids == []


def test_pipeline_result_instances_have_independent_source_lists() -> None:
    first = PipelineResult(purchase_order=PurchaseOrder(), answer="Net 30")
    second = PipelineResult(purchase_order=PurchaseOrder(), answer="No answer")
    source = SourceEvidence(
        chunk_id="page-001-chunk-001",
        page_number=1,
        text="Payment terms: Net 30",
        score=0.9,
    )

    first.sources.append(source)

    assert first.sources == [source]
    assert second.sources == []


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (PDFPage, {"page_number": 1, "text": "", "unexpected": True}),
        (LineItem, {"unexpected": True}),
        (PurchaseOrder, {"unexpected": True}),
        (Chunk, {"chunk_id": "chunk-1", "page_number": 1, "text": "x", "unexpected": True}),
        (
            SourceEvidence,
            {
                "chunk_id": "chunk-1",
                "page_number": 1,
                "text": "x",
                "score": 0.5,
                "unexpected": True,
            },
        ),
        (GroundedAnswer, {"answer": "x", "unexpected": True}),
        (
            PipelineResult,
            {"purchase_order": PurchaseOrder(), "answer": "x", "unexpected": True},
        ),
    ],
)
def test_models_reject_unknown_fields(model: type, values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)

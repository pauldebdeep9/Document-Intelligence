from decimal import Decimal

import pytest

import evals.metrics as metrics
from evals.gold.schema import Anchor
from evals.metrics import (
    FieldVerdict,
    LineItemScore,
    anchors_satisfied,
    first_hit_rank,
    format_kn,
    hit_at_k,
    is_insufficiency_response,
    reciprocal_rank,
    score_field,
    score_line_items,
    score_purchase_order,
)
from isc.models import LineItem, PurchaseOrder, SourceEvidence

_INSUFFICIENT_ANSWER = "I don't have enough information in the provided sources."


def _source(doc_id: str, chunk_id: str, text: str, page_number: int = 1) -> SourceEvidence:
    return SourceEvidence(
        doc_id=doc_id, chunk_id=chunk_id, page_number=page_number, text=text, score=0.5
    )


# --- anchors_satisfied -------------------------------------------------------------------------


def test_anchors_satisfied_true_for_matching_text_and_doc_id() -> None:
    retrieved = [_source("po-001", "po-001:c1", "PO Number: PO-1001")]
    anchors = [Anchor(page_number=1, text="PO Number: PO-1001")]

    assert anchors_satisfied(retrieved, "po-001", anchors) == [True]


def test_anchors_satisfied_false_for_wrong_doc_id() -> None:
    # Anchor text is present verbatim, but in a chunk from a different document — this is
    # the exact near_duplicate scenario: po-004 and po-005 share near-identical text.
    retrieved = [_source("po-005", "po-005:c1", "1. Part Number: 4500123456")]
    anchors = [Anchor(page_number=1, text="1. Part Number: 4500123456")]

    assert anchors_satisfied(retrieved, "po-004", anchors) == [False]


def test_anchors_satisfied_false_when_text_not_present_anywhere() -> None:
    retrieved = [_source("po-001", "po-001:c1", "Supplier: ACME Components")]
    anchors = [Anchor(page_number=1, text="PO Number: PO-1001")]

    assert anchors_satisfied(retrieved, "po-001", anchors) == [False]


def test_anchors_satisfied_returns_one_bool_per_anchor_independently() -> None:
    retrieved = [_source("po-006", "po-006:c1", "Part Number: BRG-1001", page_number=1)]
    anchors = [
        Anchor(page_number=1, text="Part Number: BRG-1001"),
        Anchor(page_number=2, text="Part Number: BRG-1003"),
    ]

    assert anchors_satisfied(retrieved, "po-006", anchors) == [True, False]


def test_anchors_satisfied_anchor_straddling_two_chunks_is_not_satisfied() -> None:
    # The full anchor text appears only when the two chunks are concatenated; neither chunk
    # contains it alone, so it must not be satisfied.
    full_text = "Part Number: BRG-1001 Description: Deep groove ball bearing, 30mm"
    retrieved = [
        _source("po-006", "po-006:c1", "Part Number: BRG-1001 Descrip"),
        _source("po-006", "po-006:c2", "tion: Deep groove ball bearing, 30mm"),
    ]
    anchors = [Anchor(page_number=1, text=full_text)]

    assert anchors_satisfied(retrieved, "po-006", anchors) == [False]


# --- hit_at_k ------------------------------------------------------------------------------


def test_hit_at_k_true_when_any_anchor_satisfied() -> None:
    retrieved = [_source("po-001", "po-001:c1", "PO Number: PO-1001")]
    anchors = [
        Anchor(page_number=1, text="PO Number: PO-1001"),
        Anchor(page_number=1, text="not present"),
    ]

    assert hit_at_k(retrieved, "po-001", anchors) is True


def test_hit_at_k_false_when_no_anchor_satisfied() -> None:
    retrieved = [_source("po-001", "po-001:c1", "Supplier: ACME Components")]
    anchors = [Anchor(page_number=1, text="PO Number: PO-1001")]

    assert hit_at_k(retrieved, "po-001", anchors) is False


def test_hit_at_k_false_for_empty_anchors_list() -> None:
    # An "absent" class question has no anchors; hit_at_k has nothing to hit.
    retrieved = [_source("po-002", "po-002:c1", "Ship-to Site: Dockside Warehouse 2")]

    assert hit_at_k(retrieved, "po-002", []) is False


def test_hit_at_k_false_when_only_wrong_doc_id_matches() -> None:
    retrieved = [_source("po-005", "po-005:c1", "1. Part Number: 4500123456")]
    anchors = [Anchor(page_number=1, text="1. Part Number: 4500123456")]

    assert hit_at_k(retrieved, "po-004", anchors) is False


# --- first_hit_rank -------------------------------------------------------------------------


def test_first_hit_rank_returns_one_based_rank_of_first_satisfying_chunk() -> None:
    retrieved = [
        _source("po-001", "po-001:c1", "Supplier: ACME Components"),
        _source("po-001", "po-001:c2", "PO Number: PO-1001"),
        _source("po-001", "po-001:c3", "PO Number: PO-1001"),
    ]
    anchors = [Anchor(page_number=1, text="PO Number: PO-1001")]

    assert first_hit_rank(retrieved, "po-001", anchors) == 2


def test_first_hit_rank_returns_none_on_total_miss() -> None:
    retrieved = [_source("po-001", "po-001:c1", "Supplier: ACME Components")]
    anchors = [Anchor(page_number=1, text="PO Number: PO-1001")]

    assert first_hit_rank(retrieved, "po-001", anchors) is None


def test_first_hit_rank_ignores_earlier_wrong_doc_id_matches() -> None:
    retrieved = [
        _source("po-005", "po-005:c1", "1. Part Number: 4500123456"),  # wrong doc, rank 1
        _source("po-004", "po-004:c1", "1. Part Number: 4500123456"),  # right doc, rank 2
    ]
    anchors = [Anchor(page_number=1, text="1. Part Number: 4500123456")]

    assert first_hit_rank(retrieved, "po-004", anchors) == 2


def test_first_hit_rank_returns_none_for_empty_anchors() -> None:
    retrieved = [_source("po-002", "po-002:c1", "Ship-to Site: Dockside Warehouse 2")]

    assert first_hit_rank(retrieved, "po-002", []) is None


# --- reciprocal_rank -------------------------------------------------------------------------


def test_reciprocal_rank_of_rank_one_is_one() -> None:
    assert reciprocal_rank(1) == pytest.approx(1.0)


def test_reciprocal_rank_of_rank_four_is_one_quarter() -> None:
    assert reciprocal_rank(4) == pytest.approx(0.25)


def test_reciprocal_rank_of_none_is_zero() -> None:
    assert reciprocal_rank(None) == 0.0


# --- score_field -----------------------------------------------------------------------------


def test_score_field_decimal_equal_values_different_representations_is_correct() -> None:
    assert score_field(Decimal("10.00"), Decimal("10")) == FieldVerdict.CORRECT


def test_score_field_both_none_is_correct() -> None:
    assert score_field(None, None) == FieldVerdict.CORRECT


def test_score_field_expected_none_actual_present_is_hallucinated() -> None:
    assert score_field(None, "USD") == FieldVerdict.HALLUCINATED


def test_score_field_expected_present_actual_none_is_missed() -> None:
    assert score_field("USD", None) == FieldVerdict.MISSED


def test_score_field_string_exact_match_is_correct() -> None:
    assert score_field("Net 30", "Net 30") == FieldVerdict.CORRECT


def test_score_field_string_differing_only_in_surrounding_whitespace_is_correct() -> None:
    assert score_field("Net 30", "  Net 30\n") == FieldVerdict.CORRECT


def test_score_field_string_case_difference_is_wrong() -> None:
    assert score_field("Smith & Sons Machining", "smith & sons machining") == FieldVerdict.WRONG


def test_score_field_ampersand_vs_and_is_wrong() -> None:
    assert score_field("Smith & Sons Machining", "Smith and Sons Machining") == FieldVerdict.WRONG


def test_score_field_decimal_different_values_is_wrong() -> None:
    assert score_field(Decimal("10.00"), Decimal("10.50")) == FieldVerdict.WRONG


# --- score_purchase_order --------------------------------------------------------------------


def test_score_purchase_order_covers_all_seven_header_fields() -> None:
    result = score_purchase_order(PurchaseOrder(), PurchaseOrder())

    assert set(result) == {
        "po_number",
        "po_date",
        "supplier_name",
        "ship_to_site",
        "payment_terms",
        "currency",
        "total_amount",
    }


def test_score_purchase_order_mixed_verdicts() -> None:
    expected = PurchaseOrder(
        po_number="PO-1001",
        po_date="2026-03-04",
        supplier_name="ACME Components",
        ship_to_site="Riverside Plant",
        payment_terms="Net 30",
        currency="USD",
        total_amount=Decimal("1985.00"),
    )
    actual = PurchaseOrder(
        po_number="PO-1001",  # correct
        po_date="2026-03-05",  # wrong
        supplier_name=None,  # missed
        ship_to_site="Riverside Plant",  # correct
        payment_terms="Net 30",  # correct
        currency=None,  # missed
        total_amount=Decimal("1985.00"),  # correct
    )

    result = score_purchase_order(expected, actual)

    assert result["po_number"] == FieldVerdict.CORRECT
    assert result["po_date"] == FieldVerdict.WRONG
    assert result["supplier_name"] == FieldVerdict.MISSED
    assert result["ship_to_site"] == FieldVerdict.CORRECT
    assert result["payment_terms"] == FieldVerdict.CORRECT
    assert result["currency"] == FieldVerdict.MISSED
    assert result["total_amount"] == FieldVerdict.CORRECT


def test_score_purchase_order_excludes_line_items_key() -> None:
    expected = PurchaseOrder(line_items=[LineItem(part_number="A")])
    actual = PurchaseOrder(line_items=[])

    result = score_purchase_order(expected, actual)

    assert "line_items" not in result


# --- score_line_items ------------------------------------------------------------------------


def test_score_line_items_matches_by_part_number_regardless_of_order() -> None:
    expected = [LineItem(part_number="A"), LineItem(part_number="B")]
    actual = [LineItem(part_number="B"), LineItem(part_number="A")]

    assert score_line_items(expected, actual) == LineItemScore(matched=2, missing=0, spurious=0)


def test_score_line_items_matches_by_position_when_part_number_is_none() -> None:
    expected = [LineItem(description="first"), LineItem(description="second")]
    actual = [LineItem(description="first, differently worded"), LineItem(description="second")]

    assert score_line_items(expected, actual) == LineItemScore(matched=2, missing=0, spurious=0)


def test_score_line_items_counts_missing_expected_items() -> None:
    expected = [LineItem(part_number="A"), LineItem(part_number="B")]
    actual = [LineItem(part_number="A")]

    assert score_line_items(expected, actual) == LineItemScore(matched=1, missing=1, spurious=0)


def test_score_line_items_counts_spurious_actual_items() -> None:
    expected = [LineItem(part_number="A")]
    actual = [LineItem(part_number="A"), LineItem(part_number="B")]

    assert score_line_items(expected, actual) == LineItemScore(matched=1, missing=0, spurious=1)


def test_score_line_items_empty_expected_and_actual() -> None:
    assert score_line_items([], []) == LineItemScore(matched=0, missing=0, spurious=0)


def test_score_line_items_does_not_double_count_one_actual_item() -> None:
    # Both expected items have the same part_number; only one actual item can satisfy it.
    expected = [LineItem(part_number="A"), LineItem(part_number="A")]
    actual = [LineItem(part_number="A")]

    assert score_line_items(expected, actual) == LineItemScore(matched=1, missing=1, spurious=0)


# --- is_insufficiency_response ---------------------------------------------------------------


def test_is_insufficiency_response_true_for_exact_string() -> None:
    assert is_insufficiency_response(_INSUFFICIENT_ANSWER) is True


def test_is_insufficiency_response_false_for_trailing_whitespace() -> None:
    assert is_insufficiency_response(_INSUFFICIENT_ANSWER + " ") is False


def test_is_insufficiency_response_false_for_differing_final_period() -> None:
    assert _INSUFFICIENT_ANSWER.endswith(".")
    changed_punctuation = _INSUFFICIENT_ANSWER[:-1] + "!"

    assert is_insufficiency_response(changed_punctuation) is False


def test_is_insufficiency_response_false_for_case_difference() -> None:
    assert is_insufficiency_response(_INSUFFICIENT_ANSWER.lower()) is False


def test_is_insufficiency_response_false_for_empty_string() -> None:
    assert is_insufficiency_response("") is False


# --- format_kn -------------------------------------------------------------------------------


def test_format_kn_basic() -> None:
    assert format_kn(3, 24) == "3/24"


def test_format_kn_zero_k() -> None:
    assert format_kn(0, 5) == "0/5"


def test_format_kn_k_equals_n() -> None:
    assert format_kn(24, 24) == "24/24"


# --- structural: no percentage/rate formatter -------------------------------------------------


def test_metrics_module_has_no_percentage_or_rate_formatter() -> None:
    forbidden_substrings = ("percent", "pct", "rate")
    violations = [
        name
        for name, obj in vars(metrics).items()
        if not name.startswith("_")
        and callable(obj)
        and any(substr in name.lower() for substr in forbidden_substrings)
    ]
    assert violations == []

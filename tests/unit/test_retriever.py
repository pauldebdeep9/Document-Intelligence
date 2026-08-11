"""Unit tests for retrieve/retriever.py's regex-only filter inference (P1-06
step 1). No embedder/store/chat model needed for rewrite()/infer_filters()
alone -- both are pure functions of the question string.
"""

from __future__ import annotations

from isc.retrieve.retriever import Retriever

# rewrite()/infer_filters() are pure functions of the question string alone
# -- neither touches store/embedder/chat/settings -- so a Retriever built
# with every dependency as None is enough to exercise them without mocking
# the whole pipeline.
_retriever = Retriever(store=None, embedder=None, chat=None, settings=None)  # type: ignore[arg-type]


def _infer(question: str) -> dict[str, str]:
    return _retriever.infer_filters(question)


def test_rewrite_returns_the_question_unchanged():
    assert _retriever.rewrite("What is the total on PO 4522345741?") == [
        "What is the total on PO 4522345741?"
    ]


def test_infer_filters_extracts_a_po_number():
    assert _infer("What is the total order amount on PO 4522345741?") == {
        "po_number": "4522345741"
    }


def test_infer_filters_is_empty_when_no_po_number_present():
    assert _infer("What did we spend with Omron Electronics Asia in total?") == {}


def test_infer_filters_detects_but_does_not_filter_on_part_number():
    """No chunk carries a part_number key (a table chunk covers many rows,
    each with its own part) -- filtering on it would zero out every chunk
    regardless of whether the part is in the corpus. The part number must
    be recognized (it matches PART_NUMBER) but never appear in the filters
    returned, or search_lexical's own part-number matching would be
    silently suppressed for no benefit."""
    filters = _infer("What did we pay for part PLC-1756-L83 across our purchase orders?")
    assert "part_number" not in filters
    assert filters == {}


def test_infer_filters_never_infers_a_supplier():
    """'Kestrel Industrial' resolves to one vendor code and silently drops
    the other -- exactly the ambiguity this retriever must preserve for the
    answer stage, not resolve on its own."""
    filters = _infer("What did we order from Kestrel Industrial?")
    assert filters == {}
    assert "supplier_id" not in filters
    assert "supplier_name" not in filters


def test_infer_filters_extracts_po_number_alongside_an_unfiltered_part_number():
    filters = _infer("What is the unit price of part PLC-1756-L83 on PO 4522345741?")
    assert filters == {"po_number": "4522345741"}

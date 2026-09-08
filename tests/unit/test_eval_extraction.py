"""Unit tests for eval/extraction.py's compare() and ExtractionReport.

Fixtures build a PurchaseOrder directly (not through _wrap) since compare()
only needs record.fields() -- value + confidence per field -- not span
location or the rest of the wrap pipeline.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from isc.common.confidence import Confidence
from isc.eval.extraction import ExtractionReport, compare, compare_lines
from isc.models.records.base import ExtractedField
from isc.models.records.purchase_order import POLine, PurchaseOrder


def _field(value, confidence=0.9, absent_reason=None):
    if absent_reason is not None:
        return ExtractedField(
            value=None, confidence=Confidence.certain(), absent_reason=absent_reason,
        )
    return ExtractedField(value=value, confidence=Confidence(confidence))


def _record(**fields) -> PurchaseOrder:
    defaults = dict(
        document_id="doc_test",
        po_number=_field("4500123456"),
        po_date=_field(date(2025, 8, 16)),
        supplier_name=_field("Acme Supply Co"),
        supplier_id=_field("V1"),
        ship_to_site=_field("Plant 1"),
        incoterms=_field("FOB"),
        payment_terms=_field("Net 30"),
        currency=_field("USD"),
        total_amount=_field(Decimal("2200.00")),
        buyer_contact=_field(None, absent_reason="not present in document"),
    )
    defaults.update(fields)
    return PurchaseOrder(**defaults)


def _gold(**overrides):
    raw = dict(po_number="4500123456", po_date="16/08/2025", supplier_name="Acme Supply Co",
               supplier_id="V1", ship_to_site="Plant 1", incoterms="FOB", payment_terms="Net 30",
               currency="USD", total_amount="2,200.00", buyer_contact=None)
    normalised = dict(po_number="4500123456", po_date="2025-08-16", supplier_name="Acme Supply Co",
                       supplier_id="V1", ship_to_site="Plant 1", incoterms="FOB",
                       payment_terms="Net 30", currency="USD", total_amount="2200.00",
                       buyer_contact=None)
    for key, (r, n) in overrides.items():
        raw[key] = r
        normalised[key] = n
    return {"raw": raw, "normalised": normalised}


def _raw_predicted(**overrides):
    defaults = dict(po_number="4500123456", po_date="16/08/2025", supplier_name="Acme Supply Co",
                     supplier_id="V1", ship_to_site="Plant 1", incoterms="FOB",
                     payment_terms="Net 30", currency="USD", total_amount=2200.0,
                     buyer_contact=None)
    defaults.update(overrides)
    return defaults


def test_all_correct_when_everything_matches():
    outcomes = compare("doc_test", _gold(), _raw_predicted(), _record())
    assert all(o.outcome in {"correct", "correct_absent"} for o in outcomes)
    # buyer_contact is genuinely absent both sides
    absent = [o for o in outcomes if o.field_name == "buyer_contact"]
    assert all(o.outcome == "correct_absent" for o in absent)


def test_wrong_value_detected_on_both_axes():
    gold = _gold(supplier_name=("Acme Supply Co", "Acme Supply Co"))
    raw_pred = _raw_predicted(supplier_name="Wrong Supplier Inc")
    record = _record(supplier_name=_field("Wrong Supplier Inc"))
    outcomes = compare("doc_test", gold, raw_pred, record)
    supplier_outcomes = {o.axis: o.outcome for o in outcomes if o.field_name == "supplier_name"}
    assert supplier_outcomes == {"extraction": "wrong", "normalisation": "wrong"}


def test_missed_when_gold_has_value_and_predicted_is_none():
    gold = _gold(supplier_id=("V1", "V1"))
    raw_pred = _raw_predicted(supplier_id=None)
    record = _record(supplier_id=_field(None, absent_reason="not present in document"))
    outcomes = compare("doc_test", gold, raw_pred, record)
    result = {o.axis: o.outcome for o in outcomes if o.field_name == "supplier_id"}
    assert result == {"extraction": "missed", "normalisation": "missed"}


def test_normalisation_bug_scores_wrong_here_but_correct_on_extraction():
    """The regression case this whole two-axis design exists for: the model
    read the date correctly (extraction axis), but our own normalisation
    failed on it (e.g. an unhandled date format) -- must not look like an
    extraction error."""
    gold = _gold(po_date=("16.08.2025", "2025-08-16"))
    raw_pred = _raw_predicted(po_date="16.08.2025")  # model read it correctly
    # simulate a normalisation bug: parse_iso_date failed, value ended up None
    record = _record(po_date=_field(None))
    outcomes = compare("doc_test", gold, raw_pred, record)
    result = {o.axis: o.outcome for o in outcomes if o.field_name == "po_date"}
    assert result == {"extraction": "correct", "normalisation": "missed"}


def test_hallucinated_value_where_gold_is_null_scores_wrong():
    gold = _gold(buyer_contact=(None, None))
    raw_pred = _raw_predicted(buyer_contact="A. Nobody")
    record = _record(buyer_contact=_field("A. Nobody"))
    outcomes = compare("doc_test", gold, raw_pred, record)
    result = {o.axis: o.outcome for o in outcomes if o.field_name == "buyer_contact"}
    assert result == {"extraction": "wrong", "normalisation": "wrong"}


def test_by_field_defaults_to_normalisation_axis():
    gold = _gold(po_date=("16.08.2025", "2025-08-16"))
    raw_pred = _raw_predicted(po_date="16.08.2025")
    record = _record(po_date=_field(None))  # normalisation bug again
    report = ExtractionReport(compare("doc_test", gold, raw_pred, record))
    by_field = report.by_field()  # default axis
    assert by_field["po_date"].support == 1
    assert by_field["po_date"].recall == 0.0  # missed: tp=0, fn=1 on normalisation
    extraction_by_field = report.by_field(axis="extraction")
    assert extraction_by_field["po_date"].recall == 1.0  # correct: tp=1, fn=0 on extraction


def test_auto_accept_error_rate_counts_missed_not_just_wrong():
    """Regression: the original implementation only counted `wrong`, so a
    silently-omitted field -- wrapped as missing(), which is always
    Confidence.certain() == 1.0 -- would sail through auto-accept without
    ever counting as an error."""
    gold = _gold(supplier_id=("V1", "V1"))
    raw_pred = _raw_predicted(supplier_id=None)
    record = _record(supplier_id=_field(None, absent_reason="not present in document"))
    report = ExtractionReport(compare("doc_test", gold, raw_pred, record))
    # supplier_id's normalisation-axis confidence is 1.0 (missing() == certain())
    rate = report.auto_accept_error_rate(threshold=0.90)
    assert rate > 0.0


def test_missed_field_at_certain_confidence_appears_in_both_n_and_wrong():
    """States the claim directly, not just its effect on a rate: a field the
    model silently omits is wrapped ExtractedField.missing() (base.py:46-50),
    which is Confidence.certain() == 1.0 (confidence.py:110-113) -- clearing
    any realistic auto-accept threshold -- and "missed" is in
    _ERROR_OUTCOMES (extraction.py:70). Those three facts mean a missed
    field lands in BOTH auto_accept_band()'s denominator and its numerator.
    The chain is exercised only incidentally elsewhere
    (test_auto_accept_error_rate_counts_missed_not_just_wrong asserts
    rate > 0.0, via a hand-rolled equivalent of missing() rather than the
    classmethod itself) -- worth its own named test since this is the
    metric the README leads with at 0.0%, and the one with a documented
    history of a wrong denominator."""
    gold = _gold(supplier_id=("V1", "V1"))
    raw_pred = _raw_predicted(supplier_id=None)
    record = _record(
        supplier_id=ExtractedField.missing(),
        # _record()'s own default for buyer_contact is also missing()-shaped
        # (confidence 1.0) -- dropped to 0.5 here so it falls below the
        # threshold below and does not join the population this test means
        # to isolate to supplier_id alone.
        buyer_contact=ExtractedField(value=None, confidence=Confidence(0.5)),
    )
    report = ExtractionReport(compare("doc_test", gold, raw_pred, record))

    outcome = next(
        o for o in report.outcomes
        if o.field_name == "supplier_id" and o.axis == "normalisation"
    )
    assert outcome.outcome == "missed"
    assert outcome.confidence == 1.0

    # Threshold above every other field's confidence (0.9, _field()'s
    # default) isolates supplier_id as the sole member of the auto-accept
    # population -- so (1, 1) is exactly "the missed field, and nothing
    # else, is both counted and wrong", not an inference from a moved rate.
    wrong, n = report.auto_accept_band(threshold=0.95)
    assert (wrong, n) == (1, 1)


def test_false_negatives_lists_wrong_and_missed_fields_routed_to_accept():
    gold = _gold(supplier_id=("V1", "V1"))
    raw_pred = _raw_predicted(supplier_id=None)
    record = _record(supplier_id=_field(None, absent_reason="not present in document"))
    report = ExtractionReport(compare("doc_test", gold, raw_pred, record))
    fns = report.false_negatives(threshold=0.90)
    assert any(o.field_name == "supplier_id" for o in fns)


def test_false_negatives_empty_when_nothing_disagrees_with_gold():
    report = ExtractionReport(compare("doc_test", _gold(), _raw_predicted(), _record()))
    assert report.false_negatives(threshold=0.90) == []


# --- compare_lines(): line-item alignment by line_number, not index --------
# The regression this guards: a real corpus document (po_008) with 41
# extracted lines against 42 gold lines. Index alignment shifts every line
# after the drop, turning one missing line into 8 fields x 21 lines of
# phantom mismatches. These fixtures reproduce the same shape (a dropped
# line with a surviving line after it) at a size small enough to assert on
# directly.

_LINE_FIELD_NAMES = (
    "line_number", "part_number", "description", "quantity",
    "unit_of_measure", "unit_price", "extended_price", "promised_date",
)


def _po_line(line_number=10, **overrides) -> POLine:
    defaults = dict(
        line_number=_field(line_number),
        part_number=_field("PLC-1756-L83"),
        description=_field("ControlLogix module"),
        quantity=_field(Decimal("3")),
        unit_of_measure=_field("EA"),
        unit_price=_field(Decimal("500.00")),
        extended_price=_field(Decimal("1500.00")),
        promised_date=_field(date(2025, 8, 16)),
    )
    defaults.update(overrides)
    return POLine(**defaults)


def _raw_po_line(line_number=10, **overrides) -> dict:
    defaults = dict(line_number=line_number, part_number="PLC-1756-L83",
                     description="ControlLogix module", quantity=3.0,
                     unit_of_measure="EA", unit_price=500.0,
                     extended_price=1500.0, promised_date="16/08/2025")
    defaults.update(overrides)
    return defaults


def _gold_po_line(line_number=10, **overrides) -> tuple[dict, dict]:
    raw = dict(line_number=line_number, part_number="PLC-1756-L83",
               description="ControlLogix module", quantity="3",
               unit_of_measure="EA", unit_price="500.00",
               extended_price="1500.00", promised_date="16/08/2025")
    normalised = dict(line_number=line_number, part_number="PLC-1756-L83",
                       description="ControlLogix module", quantity="3.00",
                       unit_of_measure="EA", unit_price="500.00",
                       extended_price="1500.00", promised_date="2025-08-16")
    for key, (r, n) in overrides.items():
        raw[key] = r
        normalised[key] = n
    return raw, normalised


def _lines_gold(*pairs: tuple[dict, dict]) -> dict:
    raws, norms = zip(*pairs, strict=True) if pairs else ((), ())
    return {"raw": {"lines": list(raws)}, "normalised": {"lines": list(norms)}}


def test_lines_align_by_line_number_and_score_correct():
    lines = [_po_line(10), _po_line(20, part_number=_field("PSU-24V-10A"))]
    raw_predicted = {"lines": [_raw_po_line(10), _raw_po_line(20, part_number="PSU-24V-10A")]}
    gold = _lines_gold(
        _gold_po_line(10),
        _gold_po_line(20, part_number=("PSU-24V-10A", "PSU-24V-10A")),
    )
    outcomes = compare_lines("doc_test", gold, raw_predicted, _record(lines=lines))
    assert len(outcomes) == len(_LINE_FIELD_NAMES) * 2 * 2  # fields x lines x axes
    assert all(o.outcome == "correct" for o in outcomes)
    assert {o.field_name for o in outcomes} == {f"lines.{n}" for n in _LINE_FIELD_NAMES}


def test_dropped_line_reported_once_and_does_not_shift_remaining_lines():
    """Gold has lines 10/20/30; extraction dropped 20. Index alignment would
    smear this into 8 phantom mismatches on line 30. By line_number, line 30
    must still score entirely correct, and the drop must show up as exactly
    one outcome per axis, not one per field."""
    lines = [_po_line(10), _po_line(30)]
    raw_predicted = {"lines": [_raw_po_line(10), _raw_po_line(30)]}
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20), _gold_po_line(30))
    outcomes = compare_lines("doc_test", gold, raw_predicted, _record(lines=lines))

    dropped = [o for o in outcomes if o.outcome == "dropped_line"]
    assert len(dropped) == 2  # one per axis, not one per field
    assert {o.axis for o in dropped} == {"extraction", "normalisation"}
    assert all(o.field_name == "lines[20]" for o in dropped)

    assert not any(o.outcome == "missed" for o in outcomes)
    surviving_part_numbers = [o for o in outcomes if o.field_name == "lines.part_number"]
    assert surviving_part_numbers  # lines 10 and 30, both axes
    assert all(o.outcome == "correct" for o in surviving_part_numbers)


def test_hallucinated_line_is_a_distinct_outcome_from_dropped():
    lines = [_po_line(10), _po_line(20), _po_line(99, part_number=_field("GHOST-1"))]
    raw_predicted = {
        "lines": [_raw_po_line(10), _raw_po_line(20), _raw_po_line(99, part_number="GHOST-1")],
    }
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20))
    outcomes = compare_lines("doc_test", gold, raw_predicted, _record(lines=lines))

    hallucinated = [o for o in outcomes if o.outcome == "hallucinated_line"]
    assert len(hallucinated) == 2  # one per axis
    assert {o.axis for o in hallucinated} == {"extraction", "normalisation"}
    assert all(o.field_name == "lines[99]" for o in hallucinated)
    assert not any(o.outcome == "dropped_line" for o in outcomes)

    norm_outcome = next(o for o in hallucinated if o.axis == "normalisation")
    assert norm_outcome.gold is None
    assert norm_outcome.predicted["part_number"]["value"] == "GHOST-1"


def test_line_number_mismatch_falls_back_to_positional_alignment():
    """line_number itself misread as 11 instead of 10 -- but the line count
    still matches gold (2 vs 2), so this must not be scored as a drop of
    line 10 plus a hallucination of line 11. It is reported once as a
    mismatch, and the document's lines are aligned positionally instead:
    every field other than line_number itself must still compare against
    the right gold line."""
    lines = [_po_line(11), _po_line(20)]
    raw_predicted = {"lines": [_raw_po_line(11), _raw_po_line(20)]}
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20))
    outcomes = compare_lines("doc_test", gold, raw_predicted, _record(lines=lines))

    mismatch = [o for o in outcomes if o.outcome == "line_number_mismatch"]
    assert len(mismatch) == 2  # one per axis, once for the document
    assert {o.axis for o in mismatch} == {"extraction", "normalisation"}
    assert all(o.field_name == "lines" for o in mismatch)
    assert not any(o.outcome in {"dropped_line", "hallucinated_line"} for o in outcomes)

    line_number_outcomes = {
        o.axis: o.outcome for o in outcomes
        if o.field_name == "lines.line_number" and o.predicted == "11"
    }
    assert line_number_outcomes == {"extraction": "wrong", "normalisation": "wrong"}

    # every other field on that same (positionally aligned) line still
    # matches gold line 10 -- proof the fallback paired position, not label.
    part_number_outcomes = [o for o in outcomes if o.field_name == "lines.part_number"]
    assert all(o.outcome == "correct" for o in part_number_outcomes)


def test_compare_lines_is_a_noop_for_a_record_with_no_lines():
    outcomes = compare_lines("doc_test", _lines_gold(), {"lines": []}, _record(lines=[]))
    assert outcomes == []


# --- line outcomes vs by_field()/false_negatives() aggregation -------------

def test_by_field_excludes_line_level_outcomes():
    """dropped_line/hallucinated_line/line_number_mismatch must not appear
    in the per-field P/R/F1 table -- they are record-level events, and a
    P/R/F1 row would misreport a dropped line as either noise or a
    miscounted field."""
    lines = [_po_line(10), _po_line(30)]  # line 20 dropped
    raw_predicted = {"lines": [_raw_po_line(10), _raw_po_line(30)]}
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20), _gold_po_line(30))
    report = ExtractionReport(compare_lines("doc_test", gold, raw_predicted, _record(lines=lines)))
    by_field = report.by_field()
    assert not any(name.startswith("lines[") or name == "lines" for name in by_field)
    assert "lines.part_number" in by_field  # the two surviving lines are still reported


def test_line_outcomes_reports_dropped_and_hallucinated_separately_from_by_field():
    # 3 gold lines, 2 extracted (10 kept, 20/30 dropped, 99 hallucinated) --
    # counts differ, so this exercises drop+hallucinate together rather than
    # the equal-count line_number_mismatch fallback.
    lines = [_po_line(10), _po_line(99, part_number=_field("GHOST-1"))]
    raw_predicted = {"lines": [_raw_po_line(10), _raw_po_line(99, part_number="GHOST-1")]}
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20), _gold_po_line(30))
    report = ExtractionReport(compare_lines("doc_test", gold, raw_predicted, _record(lines=lines)))
    outcomes = {o.outcome for o in report.line_outcomes()}
    assert outcomes == {"dropped_line", "hallucinated_line"}


def test_dropped_line_reaches_false_negatives_at_the_records_rollup_confidence():
    """A dropped line has no ExtractedField of its own -- nothing about it
    is ever individually routed to review -- so the record's rollup() is
    the only confidence signal that could have caught it. When rollup()
    clears the auto-accept threshold (every surviving field here is at 0.9),
    a dropped line is exactly as bad as a false negative on a single field
    and must show up as one, not disappear because line-level outcomes used
    to carry a hardcoded 0.0."""
    lines = [_po_line(10), _po_line(30)]  # line 20 dropped
    raw_predicted = {"lines": [_raw_po_line(10), _raw_po_line(30)]}
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20), _gold_po_line(30))
    record = _record(lines=lines)
    report = ExtractionReport(compare_lines("doc_test", gold, raw_predicted, record))

    dropped_fns = [o for o in report.false_negatives(threshold=0.90) if o.outcome == "dropped_line"]
    assert len(dropped_fns) == 1  # false_negatives() is per-axis; default is normalisation
    assert dropped_fns[0].confidence == pytest.approx(record.rollup().score)
    assert record.rollup().score >= 0.90  # the precondition that makes this a real false negative

    rate = report.auto_accept_error_rate(threshold=0.90)
    assert rate > 0.0  # the same drop must move the headline rate, not just the list


def test_dropped_line_does_not_reach_false_negatives_when_rollup_is_low():
    """The other half of the same fact: when something else (AGREEMENT
    disagreeing, e.g.) has already pulled rollup() below threshold, the
    record would have been queued for review anyway -- the drop was
    (incidentally) caught, and must not double-count as a fresh false
    negative."""
    lines = [_po_line(10), _po_line(30, unit_price=_field(Decimal("999999.00"), confidence=0.05))]
    raw_predicted = {"lines": [_raw_po_line(10), _raw_po_line(30, unit_price=999999.0)]}
    gold = _lines_gold(_gold_po_line(10), _gold_po_line(20), _gold_po_line(30))
    record = _record(lines=lines)
    assert record.rollup().score < 0.90
    report = ExtractionReport(compare_lines("doc_test", gold, raw_predicted, record))
    assert report.false_negatives(threshold=0.90) == []

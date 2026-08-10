"""Extraction harness: field-level precision/recall against gold.

Four outcomes per field, and collapsing them is how extraction evals lie:

  correct        value matches gold
  wrong          value present, differs from gold          <- the dangerous one
  missed         gold has a value, extraction returned null
  correct_absent gold is null and extraction returned null <- must not count as
                 a win in the same bucket as `correct`, or a model that returns
                 null for everything scores well on sparse documents

Reported per field name and per document type, never as one aggregate number.

Two axes, not one -- see eval/normalise.py's module docstring for why:
extraction (gold["raw"] vs the model's raw output) and normalisation
(gold["normalised"] vs the wrapped record). by_field()/calibration()/
auto_accept_error_rate() all default to the normalisation axis, because that
is literally what our confidence and routing decisions are made on -- the
extraction axis exists to localise a wrong/missed outcome to "the model
misread this" vs "our own code normalised it wrong", not to be the headline
number itself.

Line items (compare_lines(), below compare()) are a separate pass with
their own three outcomes -- dropped_line, hallucinated_line,
line_number_mismatch -- because a whole line going missing or extra is not
expressible as any of the four field outcomes above without either hiding
it inside 8 per-field FieldOutcomes or, worse, an index-based alignment that
turns one dropped line into a phantom error on every field of every line
after it. See compare_lines()'s docstring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isc.eval.metrics import PRF, calibration_bins
from isc.eval.normalise import normalise_date, normalise_decimal, normalise_string
from isc.models.document import DocType
from isc.models.records.base import ExtractedField, ExtractionRecord, registry

_DATE_FIELDS = frozenset({"po_date"})
_DECIMAL_FIELDS = frozenset({"total_amount"})

# Line-item field classification, mirroring _DATE_FIELDS/_DECIMAL_FIELDS
# above but for POLine rather than the header -- see compare_lines().
_LINE_DATE_FIELDS = frozenset({"promised_date"})
_LINE_DECIMAL_FIELDS = frozenset({"quantity", "unit_price", "extended_price"})

# compare_lines()'s three line-level outcomes. Record-level events, not
# field-level ones -- a dropped line is one error, not eight -- so they are
# excluded from by_field()'s P/R/F1 table (see ExtractionReport.by_field())
# and reported through ExtractionReport.line_outcomes() instead.
_LINE_OUTCOMES = frozenset({"dropped_line", "hallucinated_line", "line_number_mismatch"})
# Of those three, the two that are themselves a disagreement with gold (a
# whole line the record got wrong) and must therefore be eligible to appear
# in false_negatives()/auto_accept_error_rate() -- exactly like `wrong` and
# `missed` do for a single field. line_number_mismatch is excluded: it is a
# statement about which alignment strategy was used for the document, not a
# value judgement -- the mismatched line's own fields are already scored
# correct/wrong/missed by the positional-fallback comparison and enter
# false_negatives() that way if they disagree.
_LINE_ERROR_OUTCOMES = frozenset({"dropped_line", "hallucinated_line"})

# Shared by error_outcomes(), false_negatives(), auto_accept_error_rate() and
# band_precision() -- one definition of "disagrees with gold" for all four,
# so they can never quietly drift apart on what counts as an error.
_ERROR_OUTCOMES = frozenset({"wrong", "missed"}) | _LINE_ERROR_OUTCOMES


@dataclass
class FieldOutcome:
    document_id: str
    field_name: str
    outcome: str
    axis: str = "normalisation"  # "extraction" | "normalisation" -- see module docstring
    predicted: Any = None
    gold: Any = None
    confidence: float = 0.0


@dataclass
class ExtractionReport:
    outcomes: list[FieldOutcome] = field(default_factory=list)

    def _axis(self, axis: str) -> list[FieldOutcome]:
        return [o for o in self.outcomes if o.axis == axis]

    def by_field(self, axis: str = "normalisation") -> dict[str, PRF]:
        """Per-field P/R/F1. Excludes compare_lines()'s three line-level
        outcomes on purpose: a dropped_line is one error about a whole line,
        not eight per-field ones, and folding it into this table would
        misreport it as either noise (0/0/0 support under a synthetic
        "lines[20]" field name) or, worse, silently miscounted against
        whichever of the four field outcomes it happened to resemble. See
        line_outcomes() for the section this belongs in instead."""
        rows = [o for o in self._axis(axis) if o.outcome not in _LINE_OUTCOMES]
        names = {o.field_name for o in rows}
        out = {}
        for name in sorted(names):
            field_rows = [o for o in rows if o.field_name == name]
            tp = sum(o.outcome == "correct" for o in field_rows)
            fp = sum(o.outcome == "wrong" for o in field_rows)
            fn = sum(o.outcome == "missed" for o in field_rows)
            out[name] = PRF.from_counts(tp, fp, fn)
        return out

    def line_outcomes(self, axis: str = "normalisation") -> list[FieldOutcome]:
        """dropped_line / hallucinated_line / line_number_mismatch entries,
        as their own section -- not a P/R/F1 table (see by_field()'s
        docstring for why), just the raw occurrences: document, line_number
        (encoded in field_name as "lines[N]", or "lines" for a document-wide
        mismatch marker), and the confidence that outcome was scored at."""
        return [o for o in self._axis(axis) if o.outcome in _LINE_OUTCOMES]

    def error_outcomes(self, axis: str = "normalisation") -> list[FieldOutcome]:
        """Every field or line outcome that disagrees with gold on this axis
        -- wrong, missed, dropped_line, hallucinated_line (_ERROR_OUTCOMES).
        The denominator for detection_rate(): every real error there is, not
        just the ones false_negatives() finds (those that also cleared the
        auto-accept threshold)."""
        return [o for o in self._axis(axis) if o.outcome in _ERROR_OUTCOMES]

    def detection_rate(self, threshold: float, axis: str = "normalisation") -> tuple[int, int]:
        """(detected, total) -- of every real error on this axis, how many
        scored low enough to not auto-accept. Report both counts, not just
        their ratio: a ratio alone cannot show whether it moved because more
        errors were caught (the numerator) or because fewer errors exist at
        all (the denominator) -- collapsing that distinction is exactly how
        a rate that improved because a whole error class was eliminated gets
        misread as a rate that improved because detection got better."""
        total = len(self.error_outcomes(axis))
        if total == 0:
            return 0, 0
        detected = total - len(self.false_negatives(threshold, axis))
        return detected, total

    def band_precision(
        self, low: float, high: float, axis: str = "normalisation"
    ) -> tuple[int, int]:
        """(wrong, total) for every outcome whose confidence falls in
        [low, high) on this axis -- a routing band's own precision, not just
        its size. Pass a Thresholds instance's own cut points (e.g.
        `.review, .auto_accept` for the review band) so this always asks
        about the band the pipeline would actually route to. Two counts, not
        a lone percentage: an empty or tiny band and a genuinely good one
        can both report 0%, and only the total tells them apart."""
        band = [o for o in self._axis(axis) if low <= o.confidence < high]
        if not band:
            return 0, 0
        wrong = sum(o.outcome in _ERROR_OUTCOMES for o in band)
        return wrong, len(band)

    def calibration(self, axis: str = "normalisation") -> list[dict[str, float]]:
        return calibration_bins(
            [(o.confidence, o.outcome in {"correct", "correct_absent"}) for o in self._axis(axis)]
        )

    def auto_accept_error_rate(self, threshold: float, axis: str = "normalisation") -> float:
        """The number a process owner actually asks for: of everything we would
        push through without review, what fraction disagrees with gold?

        Counts `missed` as well as `wrong`. A field the model silently omitted
        is wrapped as ExtractedField.missing(), which is Confidence.certain()
        (1.0) by construction -- a silent omission of a real value routes
        straight to auto-accept at maximum confidence, and a metric that only
        counted `wrong` would never see it.

        Also counts dropped_line/hallucinated_line (see _LINE_ERROR_OUTCOMES):
        a whole line the record got wrong is exactly the same kind of
        disagreement-with-gold as a `wrong` field, scored at the record's
        rollup() confidence (see compare_lines()) -- excluding it here would
        make this rate blind to the single worst outcome the harness can
        represent, for no reason other than it happening at line rather than
        field granularity.
        """
        auto = [o for o in self._axis(axis) if o.confidence >= threshold]
        if not auto:
            return 0.0
        return sum(o.outcome in _ERROR_OUTCOMES for o in auto) / len(auto)

    def false_negatives(self, threshold: float, axis: str = "normalisation") -> list[FieldOutcome]:
        """Every field or line outcome that disagrees with gold (wrong,
        missed, dropped_line, hallucinated_line) but scored high enough to
        auto-accept -- the actual headline number for this harness. See
        auto_accept_error_rate's docstring for why `missed` and the two line
        outcomes belong in this set.

        A dropped line is invisible to per-field routing -- there is no
        ExtractedField for a line that was never extracted, so nothing ever
        enqueues it for review on its own. The record's rollup() is the only
        signal that could have caught it (see compare_lines()), so it is the
        confidence this check runs against: a dropped line whose document
        still rolled up above threshold is a real false negative, full stop.
        """
        return [o for o in self.error_outcomes(axis) if o.confidence >= threshold]


def load_artifact(path: Path) -> tuple[ExtractionRecord, dict[str, Any]]:
    """Read one runs/<id>/extract/<doc_id>.json artifact back into the
    wrapped record and the model's raw structured output. No LLM calls: the
    artifact already holds both halves extract/pipeline.py's run() wrote,
    so eval scores exactly what that run produced rather than a fresh
    completion that is not even guaranteed to reproduce it.

    Returns `(record, raw)` where `raw` is a plain dict shaped like
    `PurchaseOrderRaw.model_dump()` -- exactly what compare()'s
    `raw_predicted` parameter expects.
    """
    data = json.loads(path.read_text())
    record_cls = registry.get(DocType(data["doc_type"]))
    record = record_cls.model_validate(data["record"])
    return record, data["raw"]


def compare(
    document_id: str,
    gold: dict[str, Any],
    raw_predicted: dict[str, Any],
    record: ExtractionRecord,
) -> list[FieldOutcome]:
    """Header-field comparison, both axes, for one document. Line items are a
    separate pass (compare_lines() below) -- alignment by line_number rather
    than position is enough of a different problem that folding it in here
    would hide it inside a function whose name no longer describes what it
    does.

    `raw_predicted` is the model's raw structured output as a dict (e.g.
    PurchaseOrderRaw.model_dump()) -- neither side of the extraction axis has
    been through our own normalisation code. `record` is the wrapped record;
    `gold` is the full gold JSON for this document (`gold["raw"]` and
    `gold["normalised"]` sections, both dicts of field name -> value).
    """
    out: list[FieldOutcome] = []
    for name, wrapped_field in record.fields().items():
        normalise = (
            normalise_date if name in _DATE_FIELDS else
            normalise_decimal if name in _DECIMAL_FIELDS else
            normalise_string
        )
        confidence = wrapped_field.confidence.score

        out.append(_score(
            document_id, name, "extraction", confidence,
            gold_value=normalise(gold["raw"].get(name)),
            predicted_value=normalise(raw_predicted.get(name)),
        ))
        out.append(_score(
            document_id, name, "normalisation", confidence,
            gold_value=normalise(gold["normalised"].get(name)),
            predicted_value=normalise(wrapped_field.value),
        ))
    return out


def _score(
    document_id: str, field_name: str, axis: str, confidence: float,
    *, gold_value: Any, predicted_value: Any,
) -> FieldOutcome:
    if gold_value is None and predicted_value is None:
        outcome = "correct_absent"
    elif gold_value is None:
        outcome = "wrong"  # predicted a value gold does not have
    elif predicted_value is None:
        outcome = "missed"
    elif gold_value == predicted_value:
        outcome = "correct"
    else:
        outcome = "wrong"
    return FieldOutcome(
        document_id, field_name, outcome, axis, predicted_value, gold_value, confidence,
    )


def compare_lines(
    document_id: str,
    gold: dict[str, Any],
    raw_predicted: dict[str, Any],
    record: ExtractionRecord,
) -> list[FieldOutcome]:
    """Line-item comparison, both axes, for one document -- the separate pass
    compare() defers to.

    Aligned by line_number, never by index. A single dropped or inserted
    line shifts every position after it, and positional alignment turns one
    real error into one phantom error per field on every line downstream of
    it: the regression this guards is a real corpus document with 41
    extracted lines against 42 gold lines, where index alignment turned one
    dropped line into 8 fields x 21 lines of phantom mismatches and
    destroyed every field-level number for the document.

    Three line-level outcomes, each reported once per affected line -- not
    once per field on that line:

      dropped_line         gold has this line_number, extraction does not.
      hallucinated_line    extraction has this line_number, gold does not.
                            Not yet observed on the real corpus, but the
                            asymmetry with dropped_line is real (a model can
                            invent a row same as it can invent a value) and
                            worth encoding rather than assuming away.
      line_number_mismatch line_number itself is wrong on one or more
                            extracted lines, so alignment by it cannot work
                            -- but the line *count* still matches gold, so
                            this is not really a drop-plus-hallucination.
                            Reported once for the document, and that
                            document's lines are aligned positionally
                            instead (order as extracted vs gold sorted by
                            line_number), flagged so a reviewer does not
                            mistake the fallback for a clean line_number
                            match.

    Lines that align cleanly by line_number get full per-field comparison,
    the same as compare()'s header fields, tagged field_name=f"lines.{name}"
    (not lines[i].name -- by_field() aggregates by name across documents,
    and baking the index into the name would fragment that into one row per
    line position instead of one row per field).

    All three line-level outcomes are scored at record.rollup()'s confidence,
    not at 0.0 or some derived per-line figure. This matters specifically for
    dropped_line: a dropped line has no ExtractedField at all -- there is no
    per-field confidence to read, and nothing about it is ever individually
    routed to review, so rollup() (the weakest populated field across the
    whole record) is the only signal that could plausibly have caught it.
    Using anything else -- e.g. always 0.0 -- would make a dropped line
    structurally unable to ever appear in false_negatives() at a real
    threshold, hiding exactly the outcome that check exists to surface.
    """
    extracted_raw = raw_predicted.get("lines", [])
    extracted_wrapped = list(getattr(record, "lines", []))
    gold_raw = {
        ln["line_number"]: ln for ln in gold["raw"].get("lines", [])
        if ln.get("line_number") is not None
    }
    gold_norm = {
        ln["line_number"]: ln for ln in gold["normalised"].get("lines", [])
        if ln.get("line_number") is not None
    }
    if not extracted_raw and not gold_raw:
        return []

    extracted = list(zip(extracted_raw, extracted_wrapped, strict=True))
    extracted_keys = [rline.get("line_number") for rline, _ in extracted]
    gold_keys = set(gold_raw)
    dropped = gold_keys - set(extracted_keys)
    hallucinated = set(extracted_keys) - gold_keys
    # What routing actually saw for this record as a whole -- see the
    # docstring above. Computed once; every line-level outcome below shares
    # it rather than each line inventing its own notion of confidence.
    line_confidence = record.rollup().score

    if len(extracted) == len(gold_keys) and (dropped or hallucinated):
        return _compare_lines_positionally(
            document_id, gold_raw, gold_norm, extracted, line_confidence,
        )

    out: list[FieldOutcome] = []
    for line_number in sorted(dropped):
        out.extend(_line_outcome(
            document_id, line_number, "dropped_line",
            gold_raw.get(line_number), gold_norm.get(line_number),
            predicted_raw_line=None, predicted_norm_line=None, confidence=line_confidence,
        ))
    for rline, wline in extracted:
        key = rline.get("line_number")
        if key in hallucinated:
            out.extend(_line_outcome(
                document_id, key, "hallucinated_line",
                gold_raw_line=None, gold_norm_line=None,
                predicted_raw_line=rline, predicted_norm_line=wline.model_dump(mode="json"),
                confidence=line_confidence,
            ))
        else:
            out.extend(_compare_one_line(
                document_id, rline, wline, gold_raw[key], gold_norm.get(key, {}),
            ))
    return out


def _compare_lines_positionally(
    document_id: str,
    gold_raw: dict[int, dict[str, Any]],
    gold_norm: dict[int, dict[str, Any]],
    extracted: list[tuple[dict[str, Any], Any]],
    line_confidence: float,
) -> list[FieldOutcome]:
    """Fallback for compare_lines() when line_number itself cannot be
    trusted to align with gold: pair extracted lines, in extraction order,
    against gold lines sorted by line_number. Scoped to the one document
    that failed line_number alignment -- other documents keep aligning by
    line_number as normal."""
    out = [
        FieldOutcome(document_id, "lines", "line_number_mismatch", axis,
                     None, None, line_confidence)
        for axis in ("extraction", "normalisation")
    ]
    gold_pairs = sorted(gold_raw.items())  # by line_number
    for (line_number, gold_raw_line), (rline, wline) in zip(gold_pairs, extracted, strict=True):
        out.extend(_compare_one_line(
            document_id, rline, wline, gold_raw_line, gold_norm.get(line_number, {}),
        ))
    return out


def _compare_one_line(
    document_id: str,
    raw_line: dict[str, Any],
    wrapped_line: Any,
    gold_raw_line: dict[str, Any],
    gold_norm_line: dict[str, Any],
) -> list[FieldOutcome]:
    out: list[FieldOutcome] = []
    for name, field_obj in _line_fields(wrapped_line).items():
        normalise = (
            normalise_date if name in _LINE_DATE_FIELDS else
            normalise_decimal if name in _LINE_DECIMAL_FIELDS else
            normalise_string
        )
        confidence = field_obj.confidence.score
        out.append(_score(
            document_id, f"lines.{name}", "extraction", confidence,
            gold_value=normalise(gold_raw_line.get(name)),
            predicted_value=normalise(raw_line.get(name)),
        ))
        out.append(_score(
            document_id, f"lines.{name}", "normalisation", confidence,
            gold_value=normalise(gold_norm_line.get(name)),
            predicted_value=normalise(field_obj.value),
        ))
    return out


def _line_outcome(
    document_id: str,
    line_number: int | None,
    outcome: str,
    gold_raw_line: dict[str, Any] | None,
    gold_norm_line: dict[str, Any] | None,
    *,
    predicted_raw_line: dict[str, Any] | None,
    predicted_norm_line: Any,
    confidence: float,
) -> list[FieldOutcome]:
    field_name = f"lines[{line_number}]" if line_number is not None else "lines[?]"
    return [
        FieldOutcome(document_id, field_name, outcome, "extraction",
                     predicted_raw_line, gold_raw_line, confidence),
        FieldOutcome(document_id, field_name, outcome, "normalisation",
                     predicted_norm_line, gold_norm_line, confidence),
    ]


def _line_fields(line: Any) -> dict[str, ExtractedField[Any]]:
    """Same shape as ExtractionRecord.fields(), for a POLine -- POLine is a
    plain BaseModel of ExtractedFields, not an ExtractionRecord itself."""
    return {name: value for name, value in line if isinstance(value, ExtractedField)}

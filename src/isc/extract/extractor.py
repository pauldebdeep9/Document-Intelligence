"""Stage 3: Document -> ExtractionRecord.

The mapping Raw -> wrapped is where confidence is *assembled*, and it is the most
important 200 lines in the repo. Seven signals folded in:

  1. Signal.MODEL       logprob from the completion
  2. Signal.SCHEMA      discounted per repair round (from structured.py)
  3. Signal.LAYOUT      from the parse artifact's parse_confidence
  4. Signal.PROVENANCE  whether the value's source text could be located at all
  5. Signal.LEXICAL     format check (PO number pattern, ISO date, currency code)
  6. Signal.MASTER_DATA value resolves in supplier/part master
  7. Signal.AGREEMENT   arithmetic cross-checks (line totals vs header total)

1-3 arrive pre-combined as `base_confidence` (extract()'s caller already ran
Confidence.independent(structured_conf, parse_confidence) -- see the WBS: that
combinator choice is upstream of this module). 4-7 are assembled per field
below, each with its own combinator choice and reasoning -- see _apply_check()
and _apply_span_outcome().

Anything routing to 'review' is enqueued in the HITL table with its weakest
factor attached, so the reviewer is told *why*, not just *that*. A field with
no span gets that named explicitly too: it is harder to review, because the
reviewer has to open the document and search it themselves rather than jump to
a page.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from isc.common.confidence import Check, CheckOutcome, Confidence, Signal, Thresholds
from isc.common.config import load_prompt
from isc.extract import masters, spans, validators
from isc.extract.spans import Located, SpanOutcome
from isc.llm.ports import ChatModel, LLMResult, Message
from isc.llm.structured import parse_structured
from isc.models.document import Document
from isc.models.records.base import ExtractedField, ExtractionRecord, registry
from isc.models.records.purchase_order import POLine, POLineRaw, PurchaseOrder, PurchaseOrderRaw
from isc.storage.sqlite_docstore import SqliteDocStore

# Score attached when a value's source text cannot be found anywhere in its
# search space at all (SpanOutcome.NOT_FOUND). Deliberately not near-zero: a
# parser gap (a wrapped description, say) can also produce this, so it is
# real negative evidence, not proof of hallucination. AMBIGUOUS never reaches
# this -- see _apply_span_outcome.
_NOT_FOUND_SCORE = 0.2


def extract(
    doc: Document,
    model: ChatModel,
    docs: SqliteDocStore,
    thresholds: Thresholds,
    parse_confidence: Confidence,
    masters_dir: Path,
) -> tuple[ExtractionRecord, BaseModel, LLMResult]:
    """Returns the wrapped record, the raw structured model output, and the
    raw LLMResult -- all three, because each is otherwise lost:

    * `raw` is discarded after _wrap() today; extract/pipeline.py needs it
      to persist alongside the record so eval/ can score the extraction axis
      against the exact bytes this run produced, instead of paying for a
      second inference pass that is not even guaranteed to reproduce it.
    * `llm_result` carries mean_logprob, finish_reason and usage, none of
      which survive on the record -- it only keeps the derived
      model_confidence() folded into Signal.MODEL, not the logprob itself.
    """
    record_cls = registry.get(doc.doc_type)
    prompt = load_prompt(f"extract/{doc.doc_type}.v1.md")
    messages = [Message.system(prompt), Message.user(doc.text())]

    raw, conf, llm_result = parse_structured(model, messages, record_cls.raw_model)
    base_confidence = Confidence.independent(conf, parse_confidence)
    record = _wrap(raw, record_cls, doc, base_confidence, masters_dir)

    for field_name, field in record.all_fields().items():
        if thresholds.route(field.confidence) in {"review", "low_confidence"}:
            weakest = field.confidence.weakest()
            detail = f"{weakest.signal}={weakest.value:.2f}" if weakest else ""
            if field.span is None:
                no_span = "no span -- open document manually"
                detail = f"{detail}; {no_span}" if detail else no_span
            docs.enqueue_review(doc.id, field_name, field.confidence.score, detail)
    return record, raw, llm_result


def _wrap(
    raw: BaseModel,
    record_cls: type[ExtractionRecord],
    doc: Document,
    base_confidence: Confidence,
    masters_dir: Path,
) -> ExtractionRecord:
    """Raw -> ExtractedField per attribute, attaching spans, running checks,
    and assembling confidence."""
    if record_cls is PurchaseOrder:
        assert isinstance(raw, PurchaseOrderRaw)
        return _wrap_purchase_order(raw, doc, base_confidence, masters_dir)
    raise NotImplementedError(f"_wrap: no mapping yet for {record_cls.__name__}")


# --- confidence combinators, applied uniformly everywhere below ------------

def _apply_span_outcome(field: ExtractedField[Any], located: Located) -> None:
    """A value whose source text cannot be found at all is genuinely separate
    negative evidence about the same claim (this value is correct) from what
    the model/schema/layout signals already say -- the cheapest hallucination
    signal available, and it should pull confidence down multiplicatively
    (flag_conflict(), independent() under the hood), not just get flagged.
    AMBIGUOUS is deliberately excluded: the value IS present, just not
    uniquely locatable, which says nothing about correctness, so it must not
    move confidence either way."""
    if located.outcome is not SpanOutcome.NOT_FOUND:
        return
    field.flag_conflict(Signal.PROVENANCE, "value not found in document text", _NOT_FOUND_SCORE)


def _apply_check(field: ExtractedField[Any], check: Check) -> None:
    """Fold a LEXICAL/MASTER_DATA/AGREEMENT check into a field, dispatching
    on the check's own stated CheckOutcome -- not a threshold reinterpreting
    its score. A check that could only return a float forced every score
    into pass-or-fail; that is what let parse_iso_date's "ambiguous, 0.6" --
    doubt, not support -- get read as a pass and corroborate a field's
    confidence upward instead of lowering it.

    check.confidence == Confidence.unknown() (no factors) means the check
    did not apply -- e.g. the field was None, or the value is a
    deliberately-unmastered part -- and contributes nothing: folding an
    unknown() into independent() would zero the whole score, and
    corroborate() would silently no-op while still polluting the factor
    list with a 0.0 entry that would wrongly look like the field's weakest
    signal.

    PASS corroborates (noisy-OR, can only raise). UNCERTAIN discounts
    (independent(), can only lower -- never logged as a conflict, since
    nothing actually disagreed, the check just could not confirm). FAIL
    conflicts (independent() via flag_conflict(), lowers, logged so a
    reviewer sees why)."""
    if not check.confidence.factors:
        return
    factor = check.confidence.factors[0]
    if check.outcome is CheckOutcome.PASS:
        field.corroborate_with(factor.signal, factor.value, factor.detail)
    elif check.outcome is CheckOutcome.UNCERTAIN:
        field.discount(factor.signal, factor.value, factor.detail)
    else:
        field.flag_conflict(factor.signal, factor.detail, factor.value)


# --- header fields: always unscoped, document-wide search ------------------

def _field(
    doc: Document,
    raw_value: str | None,
    base: Confidence,
    transform: Callable[[str], str] = lambda v: v,
    *,
    lexical: Callable[[str], Check] | None = None,
) -> ExtractedField[str]:
    if raw_value is None:
        return ExtractedField.missing()
    located = spans.locate(doc, raw_value)
    value = transform(raw_value)
    field = ExtractedField(value=value, span=located.span, confidence=base)
    _apply_span_outcome(field, located)
    if lexical is not None:
        _apply_check(field, lexical(value))
    return field


def _decimal_field(
    doc: Document, raw_value: float | None, base: Confidence
) -> ExtractedField[Decimal]:
    if raw_value is None:
        return ExtractedField.missing()
    located = spans.locate(doc, raw_value)
    field = ExtractedField(value=Decimal(str(raw_value)), span=located.span, confidence=base)
    _apply_span_outcome(field, located)
    return field


def _date_field(
    doc: Document, raw_value: str | None, base: Confidence, site_date_format: str | None = None,
) -> ExtractedField[date]:
    """Null in the raw model means "not in the document" -- missing(). A
    string that is present but does not parse as a date is a different claim:
    the value is there, our normalisation just failed on it. That gets
    value=None with a span (if locatable), missing()'s certain-absent would
    be the wrong claim entirely. parse_iso_date's own Check -- including its
    ambiguous-format discount -- is exactly the LEXICAL signal for dates.
    `site_date_format` (see _wrap_purchase_order) lets it try the shipping
    site's own convention before falling back to a guess."""
    if raw_value is None:
        return ExtractedField.missing()
    located = spans.locate(doc, raw_value)
    value, lexical = validators.parse_iso_date(raw_value, site_date_format)
    field = ExtractedField(value=value, span=located.span, confidence=base)
    _apply_span_outcome(field, located)
    _apply_check(field, lexical)
    return field


def _wrap_purchase_order(
    raw: PurchaseOrderRaw, doc: Document, base_confidence: Confidence, masters_dir: Path
) -> PurchaseOrder:
    scopes, detail = _row_scopes(doc, raw.lines)
    alignment_ok = detail == ""
    # Resolved once from the model's own raw ship_to_site read, and fed to
    # every date field on this record, header and lines alike -- see
    # validators.parse_iso_date() and masters.resolve_site_date_format().
    # An unresolved site (unknown or near-miss name) is None: every date
    # field falls back to the fixed-priority guess on its own, never to a
    # default convention.
    site_date_format = masters.resolve_site_date_format(raw.ship_to_site, masters_dir)

    record = PurchaseOrder(
        document_id=doc.id,
        record_confidence=base_confidence,  # placeholder; replaced by rollup() below
        po_number=_field(
            doc, raw.po_number, base_confidence, str.strip,
            lexical=lambda v: validators.check_pattern(v, validators.PO_NUMBER, "po_number"),
        ),
        po_date=_date_field(doc, raw.po_date, base_confidence, site_date_format),
        supplier_name=_field(doc, raw.supplier_name, base_confidence, str.strip),
        supplier_id=_field(doc, raw.supplier_id, base_confidence, str.strip),
        ship_to_site=_field(doc, raw.ship_to_site, base_confidence, str.strip),
        incoterms=_field(
            doc, raw.incoterms, base_confidence, str.strip,
            lexical=lambda v: validators.check_enum(v, validators.INCOTERM, "incoterms"),
        ),
        payment_terms=_field(doc, raw.payment_terms, base_confidence, str.strip),
        currency=_field(
            doc, raw.currency, base_confidence, str.strip,
            lexical=lambda v: validators.check_pattern(v, validators.CURRENCY, "currency"),
        ),
        total_amount=_decimal_field(doc, raw.total_amount, base_confidence),
        buyer_contact=_field(doc, raw.buyer_contact, base_confidence, str.strip),
        lines=[
            _wrap_line(doc, raw_line, scope, alignment_ok, base_confidence, masters_dir,
                       site_date_format)
            for raw_line, scope in zip(raw.lines, scopes, strict=True)
        ],
        row_alignment_ok=alignment_ok,
        row_alignment_detail=detail,
    )

    supplier_check = masters.resolve_supplier(
        record.supplier_id.value, record.supplier_name.value, masters_dir
    )
    _apply_check(record.supplier_id, supplier_check)
    _apply_check(record.supplier_name, supplier_check)

    _fold_agreement(record)

    record.record_confidence = record.rollup()
    return record


def _wrap_line(
    doc: Document,
    raw: POLineRaw,
    scope: str | None,
    alignment_ok: bool,
    base: Confidence,
    masters_dir: Path,
    site_date_format: str | None = None,
) -> POLine:
    """part_number, description and unit_of_measure are searched document-wide
    -- they are not in the scoped set (quantity, unit_price, extended_price,
    line_number, promised_date) because they are read to *identify* the row,
    not values prone to colliding with other digits on it. Everything else is
    scoped to the row, because a bare quantity or price is exactly the kind of
    short/generic value that collides constantly document-wide.

    When alignment_ok is False, every field on this line gets span=None and
    SpanOutcome is never even computed, scoped or not -- see _row_scopes. A
    silently misaligned span is worse than no span, and if the structural row
    correspondence cannot be trusted, neither can an "unscoped" search that
    happens to land somewhere on the right document.
    """

    def unscoped(
        raw_value: Any,
        transform: Callable[[Any], Any],
        lexical: Callable[[Any], Check] | None = None,
    ) -> ExtractedField[Any]:
        if raw_value is None:
            return ExtractedField.missing()
        value = transform(raw_value)
        if not alignment_ok:
            return ExtractedField(value=value, span=None, confidence=base)
        located = spans.locate(doc, raw_value)
        field = ExtractedField(value=value, span=located.span, confidence=base)
        _apply_span_outcome(field, located)
        if lexical is not None:
            _apply_check(field, lexical(value))
        return field

    def scoped(
        raw_value: Any,
        transform: Callable[[Any], Any],
        lexical: Callable[[Any], Check] | None = None,
    ) -> ExtractedField[Any]:
        if raw_value is None:
            return ExtractedField.missing()
        value = transform(raw_value)
        if not alignment_ok:
            return ExtractedField(value=value, span=None, confidence=base)
        located = spans.locate(doc, raw_value, scope=scope)
        field = ExtractedField(value=value, span=located.span, confidence=base)
        _apply_span_outcome(field, located)
        # lexical receives raw_value, not value -- unlike unscoped() above,
        # promised_date is the only scoped() caller that needs a check at
        # all, and parse_iso_date operates on the pre-transform string, not
        # the date object transform() already reduced it to.
        if lexical is not None:
            _apply_check(field, lexical(raw_value))
        return field

    line = POLine(
        line_number=scoped(raw.line_number, lambda v: v),
        part_number=unscoped(
            raw.part_number, str.strip,
            lexical=lambda v: validators.check_pattern(v, validators.PART_NUMBER, "part_number"),
        ),
        description=unscoped(raw.description, str.strip),
        quantity=scoped(raw.quantity, lambda v: Decimal(str(v))),
        unit_of_measure=unscoped(raw.unit_of_measure, str.strip),
        unit_price=scoped(raw.unit_price, lambda v: Decimal(str(v))),
        extended_price=scoped(raw.extended_price, lambda v: Decimal(str(v))),
        promised_date=scoped(
            raw.promised_date,
            lambda v: validators.parse_iso_date(v, site_date_format)[0],
            lexical=lambda v: validators.parse_iso_date(v, site_date_format)[1],
        ),
    )
    _apply_check(line.part_number, masters.resolve_part(line.part_number.value, masters_dir))
    _apply_check(line.description, masters.check_description(
        line.part_number.value, line.description.value, masters_dir,
    ))
    return line


def _fold_agreement(record: PurchaseOrder) -> None:
    """Signal.AGREEMENT: corroborate() when the arithmetic matches -- multiple
    independently-read values agreeing is exactly the "several signals
    support the same value" case corroborate() exists for. Disagreement is a
    real, specific defect (independent(), logged as a conflict), not just low
    certainty. Neither total_amount nor a line's extended_price gets touched
    when the corresponding check returns None (total absent, or the extended
    price was not printed): absent is not wrong, and must contribute nothing,
    not score as a conflict."""
    total_agrees = record.line_total_agrees()
    if total_agrees is not None:
        detail = "line sum (qty x price) agrees with total" if total_agrees else \
            "line sum (qty x price) disagrees with total"
        outcome = CheckOutcome.PASS if total_agrees else CheckOutcome.FAIL
        score = 0.95 if total_agrees else 0.1
        _apply_check(
            record.total_amount, Check(outcome, Confidence.of(Signal.AGREEMENT, score, detail)),
        )

    for line in record.lines:
        line_agrees = line.extended_price_agrees()
        if line_agrees is not None:
            detail = "extended price agrees with qty x unit_price" if line_agrees else \
                "extended price disagrees with qty x unit_price"
            outcome = CheckOutcome.PASS if line_agrees else CheckOutcome.FAIL
            score = 0.95 if line_agrees else 0.1
            _apply_check(
                line.extended_price, Check(outcome, Confidence.of(Signal.AGREEMENT, score, detail)),
            )


def _row_scopes(doc: Document, raw_lines: list[POLineRaw]) -> tuple[list[str | None], str]:
    """Positional row scopes, one per raw line, aligned by index -- or an
    all-None list plus a non-empty failure detail if the structural row
    extraction does not verifiably correspond to what the model returned.

    Never trust the zip on faith: positional matching fails silently when it
    fails, which is worse than a string search failing loudly. So every row's
    leading ordinal must equal the corresponding raw line's line_number, and
    the row count must match the line count, before any scope is used.
    """
    if not raw_lines:
        return [], ""
    rows = spans.extract_rows(doc)
    if len(rows) != len(raw_lines):
        return [None] * len(raw_lines), f"row count {len(rows)} != line count {len(raw_lines)}"
    for i, (ordinal, _) in enumerate(rows):
        if raw_lines[i].line_number != ordinal:
            got = raw_lines[i].line_number
            detail = f"row {i}: extracted ordinal {ordinal} != raw line_number {got}"
            return [None] * len(raw_lines), detail
    return [text for _, text in rows], ""

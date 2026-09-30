"""Select, gate and sum -- in code, with Decimal, over permitted records only.

Pure function of (plan, view, policy): no model, no I/O. Each P1-09
total-spend failure maps to a rule here:

  * partial sum (q_cd_01)        -> every permitted record is considered, not
                                    whichever eight chunks ranked highest.
  * over-inclusive sum (q_cd_02) -> selection is by resolved supplier_id,
                                    never by what happened to be in context.
  * fabricated figure (q_cd_03)  -> every number is a stored extracted value
                                    or a Decimal sum of them.
  * conflated entities (q_cd_04) -> one group per (supplier_id, currency);
                                    groups are never added together.

Confidence gate. A value's route is the WEAKEST of the value itself and
every field used to select it (supplier, currency, part number): a correct
total counted under the wrong supplier or currency is as wrong as a wrong
total. Which routes may contribute is policy (config aggregate.
contributing_routes), not a new threshold -- the bands themselves are the
existing, calibrated extraction thresholds. A value that does not
contribute is never dropped silently: it is returned in `excluded` with the
reason, and render.py names it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from isc.aggregate.plan import Operation, QueryPlan
from isc.aggregate.source import RecordView, VisibleRecord
from isc.common.confidence import Confidence, Thresholds
from isc.extract.masters import supplier_id_for_printed_name
from isc.models.chunk import Chunk
from isc.models.records.base import ExtractedField

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class ValueRef:
    """One value that was considered, with where it came from."""

    document_id: str
    po_number: str | None
    supplier_id: str | None
    currency: str | None
    value: Decimal | None
    route: str
    confidence: Confidence
    # (identity chunk, value chunk) -- deduplicated, ordinal order. The
    # identity chunk is what shows the supplier/PO the value belongs to;
    # the value chunk is where the number itself is printed. A total lives
    # in the footer, which never repeats the supplier name (ADR 0009).
    evidence: tuple[Chunk, ...] = ()
    line_number: int | None = None
    reason: str = ""          # set only on excluded values


@dataclass(frozen=True)
class Group:
    supplier_id: str
    currency: str
    included: tuple[ValueRef, ...]

    @property
    def total(self) -> Decimal:
        return sum((v.value for v in self.included if v.value is not None), Decimal("0"))


@dataclass(frozen=True)
class AggregateResult:
    plan: QueryPlan
    groups: tuple[Group, ...] = ()            # total_spend
    items: tuple[ValueRef, ...] = ()          # part_prices
    excluded: tuple[ValueRef, ...] = ()
    # total_spend with a currency in the plan: how many selected-supplier
    # orders were skipped because they are in a different currency. Counts
    # only, per currency -- never their amounts.
    other_currencies: dict[str, int] = field(default_factory=dict)
    unextracted: int = 0

    @property
    def included(self) -> tuple[ValueRef, ...]:
        if self.plan.operation is Operation.TOTAL_SPEND:
            return tuple(v for g in self.groups for v in g.included)
        return self.items

    @property
    def empty(self) -> bool:
        return not self.included


def _record_supplier_id(vr: VisibleRecord, masters_dir: Path) -> str | None:
    rec = vr.record
    if rec.supplier_id.value:
        return str(rec.supplier_id.value)
    # 9/20 documents print no vendor code -- resolve the printed name instead.
    return supplier_id_for_printed_name(rec.supplier_name.value, masters_dir)


def _selection_fields(vr: VisibleRecord) -> list[ExtractedField[Any]]:
    rec = vr.record
    supplier = rec.supplier_id if rec.supplier_id.value else rec.supplier_name
    return [supplier, rec.currency]


def _weakest(*fields: ExtractedField[Any]) -> Confidence:
    return Confidence.weakest_link(*[f.confidence for f in fields])


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _identity_chunk(vr: VisibleRecord) -> Chunk | None:
    """The chunk that shows which supplier/PO this is: the first non-table
    chunk printing the supplier name, else the document's first chunk."""
    name = (vr.record.supplier_name.value or "").casefold()
    prose = [c for c in vr.chunks if not c.is_table]
    for c in prose:
        if name and name in c.text.casefold():
            return c
    return vr.chunks[0] if vr.chunks else None


def _chunk_printing(chunks: tuple[Chunk, ...], value: Decimal) -> Chunk | None:
    """First chunk whose text prints this exact amount (formatting-insensitive:
    "392,589.57" == Decimal("392589.57")). Span location by string search,
    same approach as ADR 0005 -- bboxes are not available. Prose chunks are
    searched before table chunks: an order total is printed in the footer,
    and a line's extended price could coincidentally equal it."""
    for c in sorted(chunks, key=lambda c: (c.is_table, c.ordinal)):
        for tok in _NUMBER.findall(c.text):
            if _decimal(tok.replace(",", "")) == value:
                return c
    return None


def _line_chunk(chunks: tuple[Chunk, ...], line_number: int) -> Chunk | None:
    for c in chunks:
        if c.line_range is not None and c.line_range[0] <= line_number <= c.line_range[1]:
            return c
    return None


def _evidence(*chunks: Chunk | None) -> tuple[Chunk, ...]:
    seen: dict[str, Chunk] = {}
    for c in chunks:
        if c is not None:
            seen.setdefault(c.id, c)
    return tuple(sorted(seen.values(), key=lambda c: c.ordinal))


def execute(
    plan: QueryPlan, view: RecordView, thresholds: Thresholds,
    contributing_routes: frozenset[str], masters_dir: Path,
) -> AggregateResult:
    if plan.operation is Operation.TOTAL_SPEND:
        return _total_spend(plan, view, thresholds, contributing_routes, masters_dir)
    if plan.operation is Operation.PART_PRICES:
        return _part_prices(plan, view, thresholds, contributing_routes, masters_dir)
    raise ValueError(f"execute() called with operation {plan.operation}")


def _total_spend(
    plan: QueryPlan, view: RecordView, thresholds: Thresholds,
    contributing: frozenset[str], masters_dir: Path,
) -> AggregateResult:
    wanted = set(plan.supplier_ids)
    included: dict[tuple[str, str], list[ValueRef]] = {}
    excluded: list[ValueRef] = []
    other_currencies: dict[str, int] = {}

    for vr in view.records:
        sid = _record_supplier_id(vr, masters_dir)
        if sid is None or sid not in wanted:
            continue
        rec = vr.record
        currency = rec.currency.value
        if plan.currency is not None and currency != plan.currency:
            key = currency or "unknown currency"
            other_currencies[key] = other_currencies.get(key, 0) + 1
            continue

        value = _decimal(rec.total_amount.value)
        confidence = _weakest(rec.total_amount, *_selection_fields(vr))
        route = thresholds.route(confidence)
        identity = _identity_chunk(vr)
        base = ValueRef(document_id=vr.document.id, po_number=rec.po_number.value,
                        supplier_id=sid, currency=currency, value=value,
                        route=route, confidence=confidence)

        if currency is None:
            excluded.append(replace(base, evidence=_evidence(identity),
                                    reason="no currency is printed on the order"))
            continue
        if value is None:
            excluded.append(replace(base, evidence=_evidence(identity),
                                    reason="no order total is printed on the document"))
            continue
        value_chunk = _chunk_printing(vr.chunks, value)
        if value_chunk is None:
            excluded.append(replace(base, evidence=_evidence(identity),
                                    reason="its total could not be located in the indexed text"))
            continue
        if route not in contributing:
            excluded.append(replace(base, evidence=_evidence(identity, value_chunk),
                                    reason=f"its extracted total is below the review "
                                           f"threshold (confidence {confidence.score:.2f})"))
            continue
        included.setdefault((sid, currency), []).append(
            replace(base, evidence=_evidence(identity, value_chunk)))

    order = {sid: i for i, sid in enumerate(plan.supplier_ids)}
    groups = tuple(
        Group(supplier_id=sid, currency=cur,
              included=tuple(sorted(refs, key=lambda r: r.po_number or "")))
        for (sid, cur), refs in sorted(included.items(), key=lambda kv: (order[kv[0][0]], kv[0][1]))
    )
    return AggregateResult(plan=plan, groups=groups, excluded=tuple(excluded),
                           other_currencies=other_currencies,
                           unextracted=len(view.unextracted))


def _part_prices(
    plan: QueryPlan, view: RecordView, thresholds: Thresholds,
    contributing: frozenset[str], masters_dir: Path,
) -> AggregateResult:
    wanted = set(plan.supplier_ids)
    items: list[ValueRef] = []
    excluded: list[ValueRef] = []

    for vr in view.records:
        rec = vr.record
        sid = _record_supplier_id(vr, masters_dir)
        if wanted and (sid is None or sid not in wanted):
            continue
        currency = rec.currency.value
        if plan.currency is not None and currency != plan.currency:
            continue
        identity = _identity_chunk(vr)
        # A supplier-scoped question selected this record by its supplier, so
        # that field enters every line's route (D5); unscoped, it selected
        # nothing and stays out.
        scoped = [_selection_fields(vr)[0]] if wanted else []
        for line in rec.lines:
            part = line.part_number.value
            if part is None or str(part).upper() != plan.part_number:
                continue
            ln = line.line_number.value
            value = _decimal(line.unit_price.value)
            confidence = _weakest(line.unit_price, line.part_number, rec.currency, *scoped)
            route = thresholds.route(confidence)
            line_chunk = _line_chunk(vr.chunks, ln) if ln is not None else None
            ref = ValueRef(document_id=vr.document.id, po_number=rec.po_number.value,
                           supplier_id=sid, currency=currency, value=value, route=route,
                           confidence=confidence, line_number=ln)
            if value is None:
                excluded.append(replace(ref, evidence=_evidence(identity, line_chunk),
                                        reason="no unit price is printed on that line"))
            elif line_chunk is None:
                excluded.append(replace(ref, evidence=_evidence(identity),
                                        reason="that line could not be located in the "
                                               "indexed text"))
            elif route not in contributing:
                excluded.append(replace(ref, evidence=_evidence(identity, line_chunk),
                                        reason=f"its extracted unit price is below the review "
                                               f"threshold (confidence {confidence.score:.2f})"))
            else:
                items.append(replace(ref, evidence=_evidence(identity, line_chunk)))

    items.sort(key=lambda r: (r.po_number or "", r.line_number or 0))
    return AggregateResult(plan=plan, items=tuple(items), excluded=tuple(excluded),
                           unextracted=len(view.unextracted))

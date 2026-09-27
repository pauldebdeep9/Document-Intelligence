"""AggregateResult -> Answer, deterministically. No model writes this text.

Every number in the output is either a stored extracted value or a Decimal
sum of them, formatted with thousands separators and two decimals. Every
sentence that names a supplier or PO number carries [n] markers over
`supporting`, in the same 1-indexed convention the chunk path uses, so the
result goes through the SAME bind_citations() and verify_attribution() a
generated draft does (answerer.py). That check should never fire on text
built here; if it ever does, the renderer has a bug and the answer is
discarded exactly as a generated draft's would be.

Sentence layout is deliberate: one sentence per line, each starting with a
capital letter and ending with a period, because verify_attribution() is
per-sentence and its splitter only breaks on [.!?] followed by a capital or
"[" -- a "- " bullet list would merge every line into one sentence and let
one PO's citation vouch for another PO's number.

Wording is for a finance reader (the D7 review): a headline says how many of
the matching orders or lines its figure covers, an excluded or
not-yet-confirmed value sits under the supplier whose total it affects, and
review status is stated in plain words -- the confidence score stays in
Answer.confidence, not the text.

What is never rendered: anything about records the principal cannot read.
The view render sees was filtered before execute() ran, so "visible to you"
is literally true and a hidden order cannot even be counted.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from isc.aggregate.execute import AggregateResult, Group, ValueRef
from isc.aggregate.plan import Operation
from isc.aggregate.resolve import supplier_names_by_id
from isc.models.chunk import Chunk, ScoredChunk


def money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


class _Evidence:
    """Chunk -> citation number, first-seen order. `supporting` is built
    from this, so marker [n] always resolves to supporting[n-1]."""

    def __init__(self) -> None:
        self._order: list[Chunk] = []
        self._index: dict[str, int] = {}

    def markers(self, chunks: tuple[Chunk, ...] | list[Chunk]) -> str:
        nums = []
        for c in chunks:
            if c.id not in self._index:
                self._order.append(c)
                self._index[c.id] = len(self._order)
            nums.append(self._index[c.id])
        return "".join(f"[{n}]" for n in sorted(set(nums)))

    def supporting(self) -> list[ScoredChunk]:
        # score/rank carry no retrieval meaning on this path; score=1.0 marks
        # "selected deterministically", not a similarity.
        return [ScoredChunk(chunk=c, score=1.0, rank=i) for i, c in enumerate(self._order)]


def _all_evidence(refs: tuple[ValueRef, ...] | list[ValueRef]) -> list[Chunk]:
    return [c for r in refs for c in r.evidence]


def _po(ref: ValueRef) -> str:
    return f"PO {ref.po_number}" if ref.po_number else "An order with no PO number"


def _review_lines(refs: list[ValueRef], ev: _Evidence) -> list[str]:
    lines = []
    for r in refs:
        if r.route == "accept" or r.value is None:
            continue
        what = f"line {r.line_number} unit price" if r.line_number is not None else "total"
        lines.append(f"{_po(r)} {what} ({money(r.value)} {r.currency}) is included but has "
                     f"not yet been confirmed by a reviewer {ev.markers(r.evidence)}.")
    return lines


def _excluded_lines(refs: list[ValueRef], ev: _Evidence) -> list[str]:
    # Never names the supplier: placement under its section says whose it is,
    # and an excluded ref with no evidence would be an uncited supplier name.
    lines = []
    for r in refs:
        where = f"{_po(r)} line {r.line_number}" if r.line_number is not None else _po(r)
        marks = ev.markers(r.evidence)
        tail = f" {marks}." if marks else "."
        lines.append(f"Not included: {where}, because {r.reason}{tail}")
    return lines


def render_text(result: AggregateResult, masters_dir: Path) -> tuple[str, list[ScoredChunk]]:
    names = supplier_names_by_id(masters_dir)
    ev = _Evidence()
    plan = result.plan

    if plan.operation is Operation.TOTAL_SPEND:
        lines = _total_spend_lines(result, names, ev)
    else:
        lines = _part_price_lines(result, names, ev)

    if result.other_currencies:
        parts = ", ".join(f"{_plural(n, 'order')} in {cur}"
                          for cur, n in sorted(result.other_currencies.items()))
        lines.append(f"Also visible to you but not included, because the question asks for "
                     f"{plan.currency}: {parts}.")
    if result.unextracted:
        lines.append(f"Not checked: {_plural(result.unextracted, 'purchase order')} visible to "
                     f"you {'has' if result.unextracted == 1 else 'have'} no extracted record "
                     f"yet, so this may be incomplete.")
    return "\n".join(lines), ev.supporting()


def _total_spend_lines(result: AggregateResult, names: dict[str, str], ev: _Evidence) -> list[str]:
    plan = result.plan
    groups = result.groups
    lines: list[str] = []
    suppliers = {g.supplier_id for g in groups}

    def po_lines(g: Group) -> list[str]:
        return [f"{_po(r)}: {money(r.value)} {r.currency} {ev.markers(r.evidence)}."
                for r in g.included if r.value is not None]

    def group_line(g: Group, head: str) -> str:
        k = len(g.included)
        m = k + sum(1 for r in result.excluded
                    if r.supplier_id == g.supplier_id and r.currency == g.currency)
        if m == k:
            cover = f"from the {_plural(k, 'matching purchase order')} visible to you"
            suffix = ""
        else:
            cover = f"from {k} of the {m} matching purchase orders visible to you"
            suffix = f" ({m - k} not included, see below)"
        return (f"{head}: {money(g.total)} {g.currency}, {cover}{suffix} "
                f"{ev.markers(_all_evidence(g.included))}.")

    single = len(groups) == 1 and len(plan.supplier_ids) == 1
    if len(plan.supplier_ids) > 1:
        lines.append(
            f"\"{plan.supplier_mention}\" matches {len(plan.supplier_ids)} suppliers in the "
            f"supplier master, so each is shown separately and not added together.")
    if len({g.currency for g in groups}) > 1:
        lines.append("Totals are shown per currency and are not converted.")
    if len(plan.supplier_ids) > 1 and len(suppliers) < len(plan.supplier_ids):
        # Names neither supplier: an uncited sentence naming one would fail
        # verify_attribution(), and the master list is not access-controlled,
        # so "visible to you" discloses nothing about hidden orders.
        lines.append(f"Only {len(suppliers)} of the {len(plan.supplier_ids)} "
                     f"{'has' if len(suppliers) == 1 else 'have'} orders visible to you.")
    # One section per supplier, in the order the plan resolved them: its
    # group line(s) and PO lines, then its unconfirmed values, then what was
    # left out of its total.
    for sid in plan.supplier_ids:
        name = names.get(sid, sid)
        mine = [g for g in groups if g.supplier_id == sid]
        for g in mine:
            head = (f"Total spend with {name} in {g.currency}" if single
                    else f"{name} ({sid}), {g.currency}")
            lines.append(group_line(g, head))
            lines += po_lines(g)
        lines += _review_lines([r for g in mine for r in g.included], ev)
        lines += _excluded_lines([r for r in result.excluded if r.supplier_id == sid], ev)
    # execute() only excludes selected suppliers' orders; never drop one silently.
    lines += _excluded_lines(
        [r for r in result.excluded if r.supplier_id not in plan.supplier_ids], ev)
    return lines


def _part_price_lines(result: AggregateResult, names: dict[str, str], ev: _Evidence) -> list[str]:
    plan = result.plan
    items = result.items
    pos = {r.document_id for r in items}
    scope = ""
    if plan.supplier_ids:
        scope = " from " + " or ".join(names.get(s, s) for s in plan.supplier_ids)
    excluded = len(result.excluded)
    suffix = (f" ({_plural(excluded, 'matching line')} not included, see below)"
              if excluded else "")
    lines = [
        f"Part {plan.part_number}{scope}: {_plural(len(items), 'priced line')} "
        f"across {_plural(len(pos), 'purchase order')} visible to you{suffix} "
        f"{ev.markers(_all_evidence(items))}."
    ]
    for r in items:
        if r.value is None:   # execute() never includes one; keeps the type honest
            continue
        lines.append(f"{_po(r)} line {r.line_number}: unit price {money(r.value)} {r.currency} "
                     f"{ev.markers(r.evidence)}.")
    lines += _review_lines(list(items), ev)
    lines += _excluded_lines(list(result.excluded), ev)
    return lines

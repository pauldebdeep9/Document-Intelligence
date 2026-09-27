"""The typed query plan, and the checks that decide whether to trust it.

Two-model pattern, same as <Type>Raw / <Type> in models/records/:

  * QueryPlanRaw -- what the planner model is asked for. Flat, nullable,
                    strings copied from the question.
  * QueryPlan    -- what execute.py runs. Every identifier has been checked
                    against the question text and the master data.

Why a typed plan and not generated SQL: there is no SQL to audit, the ACL
filter lives in code the model cannot write around (source.py), every plan
is a small value that can be logged and unit-tested, and a bad plan fails
validation here instead of returning a plausible number.

The one rule that matters most: every identifier in the plan must appear
verbatim in the question. A planner that "helpfully" completes "Kestrel
Industrial" to "Kestrel Industrial AG", invents a part number, or infers
SGD from a Singapore supplier has changed the question, and the answer
would be precise and wrong. Dropping words changes it too: when the
question's next word would extend the mention toward a longer master name
("Kestrel Industrial" taken from "Kestrel Industrial AG"), the planner
truncated it, and the plan is rejected. Validation failure is never an
abstention -- it returns None and the question falls through to the chunk
path unchanged.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from isc.aggregate.resolve import resolve_mention
from isc.extract.masters import supplier_ids_by_name
from isc.extract.validators import PART_NUMBER

_CURRENCY = re.compile(r"^[A-Z]{3}$")


class Operation(StrEnum):
    TOTAL_SPEND = "total_spend"   # sum order totals for one supplier (per currency)
    PART_PRICES = "part_prices"   # every unit price paid for one part number
    NONE = "none"                 # not an aggregate question -- use the chunk path


class QueryPlanRaw(BaseModel):
    operation: Literal["total_spend", "part_prices", "none"]
    supplier: str | None = None
    part_number: str | None = None
    currency: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    operation: Operation
    supplier_mention: str | None = None
    # Every master supplier the mention could mean -- more than one is the
    # ambiguous case (both Kestrel entities), and each is reported
    # separately, never summed together.
    supplier_ids: tuple[str, ...] = ()
    part_number: str | None = None
    currency: str | None = None


def _collapse(s: str) -> str:
    return " ".join(s.split()).casefold()


def _tokens(s: str) -> list[str]:
    return [t.rstrip(string.punctuation) for t in _collapse(s).split()]


def _appears_truncated(mention: str, rest: str, masters_dir: Path) -> bool:
    """True when the question word right after the mention would extend it
    toward a longer master name (legal suffix included): the planner kept
    only part of what the user wrote."""
    following = rest.split(maxsplit=1)
    if not following:
        return False
    extended = [*_tokens(mention), *_tokens(following[0])[:1]]
    return any(
        _tokens(name)[: len(extended)] == extended
        for name in supplier_ids_by_name(masters_dir)
    )


def validate_plan(
    raw: QueryPlanRaw, question: str, masters_dir: Path,
) -> tuple[QueryPlan | None, str]:
    """(plan, "") when the plan is usable; (None, reason) otherwise. The
    reason is for the trace, not the user."""
    op = Operation(raw.operation)
    if op is Operation.NONE:
        return None, "planner: not an aggregate question"

    q = _collapse(question)
    q_upper = question.upper()

    supplier_mention: str | None = None
    supplier_ids: tuple[str, ...] = ()
    if raw.supplier is not None and raw.supplier.strip():
        supplier_mention = " ".join(raw.supplier.split())
        hit = re.search(rf"(?<!\w){re.escape(_collapse(supplier_mention))}(?!\w)", q)
        if hit is None:
            return None, f"supplier {raw.supplier!r} is not in the question verbatim"
        if _appears_truncated(supplier_mention, q[hit.end():], masters_dir):
            return None, f"supplier mention {raw.supplier!r} appears truncated"
        supplier_ids = resolve_mention(supplier_mention, masters_dir)
        if not supplier_ids:
            return None, f"supplier {raw.supplier!r} matches nothing in the supplier master"

    part_number: str | None = None
    if raw.part_number is not None and raw.part_number.strip():
        part_number = raw.part_number.strip().upper()
        if not PART_NUMBER.match(part_number):
            return None, f"part number {raw.part_number!r} is not part-number shaped"
        if not re.search(rf"(?<![A-Z0-9\-]){re.escape(part_number)}(?![A-Z0-9\-])", q_upper):
            return None, f"part number {raw.part_number!r} is not in the question verbatim"

    currency: str | None = None
    if raw.currency is not None and raw.currency.strip():
        currency = raw.currency.strip().upper()
        if not _CURRENCY.match(currency):
            return None, f"currency {raw.currency!r} is not a 3-letter code"
        if not re.search(rf"(?<![A-Za-z]){currency}(?![A-Za-z])", question):
            return None, f"currency {currency} is not named in the question"

    if op is Operation.TOTAL_SPEND and not supplier_ids:
        return None, "total_spend needs a supplier"
    if op is Operation.PART_PRICES and part_number is None:
        return None, "part_prices needs a part number"
    if op is Operation.TOTAL_SPEND and part_number is not None:
        return None, "total_spend does not take a part number"

    return QueryPlan(
        operation=op,
        supplier_mention=supplier_mention,
        supplier_ids=supplier_ids,
        part_number=part_number,
        currency=currency,
    ), ""

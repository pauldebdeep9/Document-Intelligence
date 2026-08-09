"""Tests for isc.llm.schema.to_strict_schema -- the reshaping that satisfies
OpenAI strict mode's additionalProperties/required requirements.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from isc.llm.schema import to_strict_schema
from isc.models.records.invoice import InvoiceRaw
from isc.models.records.purchase_order import PurchaseOrderRaw


def _violations(node: Any, path: str = "$") -> list[str]:
    """Every object node (anything with a `properties` dict) must have
    additionalProperties: False and every property name present in required."""
    out: list[str] = []
    if isinstance(node, dict):
        if "properties" in node:
            props = set(node["properties"])
            if node.get("additionalProperties") is not False:
                out.append(f"{path}: additionalProperties is not False")
            missing = props - set(node.get("required", []))
            if missing:
                out.append(f"{path}: required is missing {sorted(missing)}")
        for key, value in node.items():
            out.extend(_violations(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            out.extend(_violations(item, f"{path}[{i}]"))
    return out


class _Flat(BaseModel):
    name: str
    age: int | None = None


def test_flat_model_has_no_violations():
    assert _violations(to_strict_schema(_Flat)) == []


def test_purchase_order_raw_has_no_violations():
    """11 fields, all optional, nested line items in $defs -- exactly the
    shape that produced the 400 before this fix."""
    assert _violations(to_strict_schema(PurchaseOrderRaw)) == []


def test_invoice_raw_has_no_violations():
    assert _violations(to_strict_schema(InvoiceRaw)) == []


def test_optional_field_is_required_with_null_in_anyof():
    schema = to_strict_schema(_Flat)
    assert "age" in schema["required"]
    types = {branch.get("type") for branch in schema["properties"]["age"]["anyOf"]}
    assert "null" in types


def test_does_not_mutate_the_underlying_schema():
    """to_strict_schema must deep-copy before tightening: model_json_schema()
    is not guaranteed fresh-and-isolated on every call, and a shared or cached
    dict mutated in place would corrupt every later caller, not just this one."""
    before = _Flat.model_json_schema()
    to_strict_schema(_Flat)
    after = _Flat.model_json_schema()
    assert after == before
    assert "additionalProperties" not in after
    assert "age" not in after.get("required", [])

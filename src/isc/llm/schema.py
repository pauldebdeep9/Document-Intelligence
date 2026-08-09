"""Reshape Pydantic's model_json_schema() into what OpenAI's strict structured
output mode requires.

Two requirements Pydantic does not satisfy on its own:

  1. `additionalProperties: false` on every object node, at every level,
     including inside `$defs`. Pydantic never emits it.
  2. Every property must appear in `required`, even ones with a default.
     Pydantic omits fields that have defaults -- which for our Raw models is
     every field, since they are all optional by design.

Requirement 2 is not just an OpenAI quirk to satisfy; it is a real improvement
for us. Without it, the model can simply omit a field it found nothing for,
and "absent from the response" is a different claim from "present with a null
value" -- the second says the model considered the field and found nothing,
the first says nothing at all. Only the second is auditable by the eval
harness. Requiring every optional field to appear as an explicit
`"field": null` closes that gap for free.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Deep-copy before mutating: this is a structure we did not construct, on
    loan from Pydantic. Whether model_json_schema() happens to return a fresh
    dict or a shared one is a caching detail that can change between versions
    without notice -- the deep-copy makes that detail irrelevant rather than
    assuming today's behaviour."""
    schema = copy.deepcopy(model.model_json_schema())
    _tighten(schema)
    return schema


def _tighten(node: Any) -> None:
    """Object nodes are any dict with a `properties` key -- top-level, nested
    inline, or sitting in `$defs`. `anyOf` branches are lists, so lists need
    walking too, not just dicts."""
    if isinstance(node, dict):
        if "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for item in node:
            _tighten(item)

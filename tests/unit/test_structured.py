from __future__ import annotations

import pytest
from pydantic import BaseModel

from isc.common.errors import SchemaRepairExhausted
from isc.llm.ports import LLMResult, Message, Usage
from isc.llm.structured import parse_structured, strip_fences


class Target(BaseModel):
    po_number: str
    total: float


class ScriptedModel:
    """Fake ChatModel returning a fixed sequence. No network in unit tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        self.calls += 1
        return LLMResult(text=self._responses.pop(0), model="fake", usage=Usage(10, 5),
                         mean_logprob=-0.1)


def test_strip_fences():
    assert strip_fences('```json\n{"a":1}\n```') == '{"a":1}'


def test_first_pass_success_keeps_full_schema_confidence():
    m = ScriptedModel(['{"po_number":"4500123","total":99.5}'])
    obj, conf, _ = parse_structured(m, [Message.user("go")], Target)
    assert obj.po_number == "4500123"
    assert m.calls == 1
    schema_factor = next(f for f in conf.factors if f.signal == "schema")
    assert schema_factor.value == 1.0


def test_repair_round_discounts_confidence():
    m = ScriptedModel(['{"po_number":"4500123"}', '{"po_number":"4500123","total":99.5}'])
    _, conf, _ = parse_structured(m, [Message.user("go")], Target)
    assert m.calls == 2
    schema_factor = next(f for f in conf.factors if f.signal == "schema")
    assert schema_factor.value < 1.0


def test_exhausted_repairs_raises():
    m = ScriptedModel(["not json", "still not json", "nope"])
    with pytest.raises(SchemaRepairExhausted):
        parse_structured(m, [Message.user("go")], Target, max_repairs=2)

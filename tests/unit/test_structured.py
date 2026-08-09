from __future__ import annotations

import pytest
from pydantic import BaseModel

from isc.common.errors import OutputTruncated, SchemaRepairExhausted
from isc.llm.ports import LLMResult, Message, Usage
from isc.llm.structured import parse_structured, strip_fences


class Target(BaseModel):
    po_number: str
    total: float


class ScriptedModel:
    """Fake ChatModel returning a fixed sequence. No network in unit tests."""

    def __init__(self, responses: list[str], finish_reason: str = "stop") -> None:
        self._responses = list(responses)
        self.calls = 0
        self._finish_reason = finish_reason

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        self.calls += 1
        return LLMResult(text=self._responses.pop(0), model="fake", usage=Usage(10, 5),
                         mean_logprob=-0.1, finish_reason=self._finish_reason)


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


def test_truncated_output_raises_immediately_without_repair_loop():
    """finish_reason == 'length' must fail fast, not enter the repair loop:
    a retry gets the same max_tokens cap and truncates at the identical
    point, so entering the loop here would just pay for three failures
    instead of one. Regression for a real corpus failure that looked like a
    generic 'Invalid JSON' SchemaRepairExhausted after three wasted calls."""
    m = ScriptedModel(['{"po_number":"45001'], finish_reason="length")
    with pytest.raises(OutputTruncated):
        parse_structured(m, [Message.user("go")], Target, max_repairs=2)
    assert m.calls == 1


def test_llm_result_cache_round_trip():
    """Regression for the from_cache field/method collision: LLMResult is a
    slots=True dataclass, so a field named from_cache shadowed the classmethod
    of the same name, and every cache hit raised "'member_descriptor' object
    is not callable". 200 tests passed while this was broken because
    ScriptedModel above has no cache -- no unit test ever exercised the hit
    path, only OpenAIChatModel did, and only against a live provider."""
    result = LLMResult(text="hi", model="fake", usage=Usage(10, 5), mean_logprob=-0.1)
    restored = LLMResult.from_cache_payload(result.to_cache_payload())
    assert restored.text == result.text
    assert restored.model == result.model
    assert restored.mean_logprob == result.mean_logprob
    assert restored.from_cache is True

"""Constrained JSON extraction with a bounded repair loop.

Provider-side strict schemas are necessary but not sufficient: they constrain
shape, not semantics, and not every provider or API version honours them. So the
contract here is that the caller always gets a validated model instance or an
exception — never a dict that "looks right".

The repair loop is bounded and every attempt is traced, because silent retries
are how extraction pipelines end up with unexplained cost and latency.
"""

from __future__ import annotations

import json
import re
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from isc.common.confidence import Confidence, Signal
from isc.common.errors import OutputTruncated, SchemaRepairExhausted
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.llm.ports import ChatModel, LLMResult, Message

log = get_logger("llm.structured")

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_REPAIR_TEMPLATE = """Your previous response did not validate against the required schema.

Validation errors:
{errors}

Return the corrected JSON object only. No prose, no code fences.
Do not invent values to satisfy the schema: use null for anything not present
in the source document."""


def strip_fences(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def parse_structured(
    model: ChatModel,
    messages: Sequence[Message],
    schema: type[T],
    *,
    max_repairs: int = 2,
) -> tuple[T, Confidence, LLMResult]:
    """Return (instance, confidence, final raw result).

    Confidence combines the model's own certainty with a schema factor that is
    discounted per repair round: output that needed fixing is less trustworthy
    than output that validated first time, and that fact must survive downstream.
    """
    convo = list(messages)
    last_error = ""

    for attempt in range(max_repairs + 1):
        with span("structured.attempt", schema=schema.__name__, attempt=attempt):
            result = model.complete(convo, schema=schema)
            if result.finish_reason == "length":
                # Retrying cannot succeed: the next attempt gets the same
                # max_tokens cap and truncates at the identical point, so
                # entering the repair loop here just pays for three failures
                # instead of one. Fail fast and name the real cause instead
                # of letting this surface three attempts later as a
                # misleading "Invalid JSON" from the eventual parse failure.
                raise OutputTruncated(result.usage.completion_tokens)
            payload = strip_fences(result.text)
            try:
                instance = schema.model_validate_json(payload)
            except ValidationError as exc:
                last_error = _format_errors(exc)
                log.warning("schema validation failed (attempt %d): %s", attempt + 1, last_error)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc}"
                log.warning("JSON decode failed (attempt %d): %s", attempt + 1, last_error)
            else:
                schema_factor = 1.0 if attempt == 0 else 0.85 ** attempt
                confidence = Confidence.independent(
                    Confidence.of(Signal.MODEL, result.model_confidence(), result.model),
                    Confidence.of(
                        Signal.SCHEMA, schema_factor,
                        "first pass" if attempt == 0 else f"{attempt} repair(s)",
                    ),
                )
                return instance, confidence, result

            if attempt < max_repairs:
                convo = [
                    *convo,
                    Message("assistant", result.text),
                    Message.user(_REPAIR_TEMPLATE.format(errors=last_error)),
                ]

    raise SchemaRepairExhausted(max_repairs + 1, last_error)


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)

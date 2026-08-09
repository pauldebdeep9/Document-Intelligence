"""Per-stage token and dollar accounting, written into the trace.

Prices drift; they live in one dict so a stale number is a one-line fix rather
than a hunt. Unknown models cost 0 and log a warning — never silently guess.
"""

from __future__ import annotations

from isc.common.logging import get_logger
from isc.llm.ports import Usage

log = get_logger("llm.cost")

# USD per 1M tokens: (input, output). Update deliberately.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

_warned: set[str] = set()


def estimate_usd(model: str, usage: Usage) -> float:
    key = model.split("/")[-1]
    if key not in PRICES:
        if key not in _warned:
            log.warning("no price for model %r; cost recorded as 0", key)
            _warned.add(key)
        return 0.0
    inp, out = PRICES[key]
    return (usage.prompt_tokens * inp + usage.completion_tokens * out) / 1_000_000

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


def _match_price_key(model: str) -> str | None:
    """Longest matching prefix, not exact key: the API returns dated
    snapshots ("gpt-4o-mini-2024-07-18"), never the bare model name in
    PRICES. Exact-only matching means every real call misses and reports
    zero. The boundary check (require '-' right after the candidate key, not
    a raw substring) matters because "gpt-4o-mini-2024-07-18" is a prefix
    match for BOTH "gpt-4o-mini" and "gpt-4o" -- picking the longest ensures
    the more specific, correct price wins rather than whichever key happens
    to be shorter."""
    candidates = [k for k in PRICES if model == k or model.startswith(k + "-")]
    return max(candidates, key=len) if candidates else None


def estimate_usd(model: str, usage: Usage) -> float:
    key = model.split("/")[-1]
    price_key = _match_price_key(key)
    if price_key is None:
        if key not in _warned:
            log.warning("no price for model %r; cost recorded as 0", key)
            _warned.add(key)
        return 0.0
    inp, out = PRICES[price_key]
    return (usage.prompt_tokens * inp + usage.completion_tokens * out) / 1_000_000

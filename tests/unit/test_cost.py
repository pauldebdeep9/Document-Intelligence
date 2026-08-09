from __future__ import annotations

from isc.llm.cost import estimate_usd
from isc.llm.ports import Usage


def test_dated_snapshot_resolves_to_base_model_price():
    """The API returns dated snapshots ("gpt-4o-mini-2024-07-18"), never the
    bare name in PRICES -- exact-only matching means every real call misses."""
    assert estimate_usd("gpt-4o-mini-2024-07-18", Usage(1_000_000, 0)) == 0.15


def test_longest_prefix_wins_over_a_shorter_also_matching_key():
    """"gpt-4o-mini-2024-07-18" is a valid prefix match for both "gpt-4o-mini"
    and "gpt-4o" -- the longer, more specific key must win, not whichever is
    shorter or happens to be inserted first."""
    cost = estimate_usd("gpt-4o-mini-2024-07-18", Usage(1_000_000, 1_000_000))
    assert cost == 0.15 + 0.60  # gpt-4o-mini pricing, not gpt-4o's 2.50+10.00


def test_unknown_model_still_reports_zero_not_a_guess():
    assert estimate_usd("some-future-model-nobody-priced-yet", Usage(1000, 1000)) == 0.0

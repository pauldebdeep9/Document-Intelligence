"""Unit tests for common/tracing.py's Run -- specifically that a run id
shared across multiple PROCESSES (what `make slice` does: one id, five
separate `isc <stage>` invocations) accumulates correctly rather than each
new Run silently starting from zero.

Each test constructs Run directly rather than going through start_run(), to
simulate "a second process touching the same run_id" without needing to
actually spawn one.
"""

from __future__ import annotations

import json

from isc.common.tracing import Run, Span


def test_fresh_run_id_starts_with_no_totals_and_no_spans(tmp_path):
    run = Run("run_fresh", tmp_path)
    assert run.totals == {}
    assert run._counter == 0


def test_totals_persist_across_two_runs_sharing_one_id(tmp_path):
    """The exact bug `make slice` would hit: ingest's Run adds tokens and
    calls summarise(); a later stage's Run, constructed fresh for the SAME
    run_id, must not silently discard that and start summary.json over from
    zero."""
    first = Run("shared_id", tmp_path)
    first.add_total("usd", 0.01)
    first.add_total("tokens.prompt", 100)
    first.summarise()

    second = Run("shared_id", tmp_path)
    assert second.totals == {"usd": 0.01, "tokens.prompt": 100}

    second.add_total("usd", 0.02)
    second.summarise()

    payload = json.loads((tmp_path / "shared_id" / "summary.json").read_text())
    assert payload["totals"]["usd"] == 0.03  # accumulated, not overwritten
    assert payload["totals"]["tokens.prompt"] == 100  # untouched key survives


def test_a_run_id_never_seen_before_has_no_totals_to_load(tmp_path):
    run = Run("brand_new_id", tmp_path)
    assert run.totals == {}


def test_span_ids_do_not_collide_across_two_runs_sharing_one_id(tmp_path):
    """Without seeding _counter from what's already in trace.jsonl, a
    second process's Run restarts at s000001 -- exactly what the first
    process's first span was also called, corrupting the shared trace."""
    first = Run("shared_id", tmp_path)
    first_id = first._next_span_id()
    first.emit(Span(span_id=first_id, name="stage.one", parent_id=None, start_ms=0.0, end_ms=1.0))

    second = Run("shared_id", tmp_path)
    second_id = second._next_span_id()
    second.emit(Span(span_id=second_id, name="stage.two", parent_id=None, start_ms=2.0, end_ms=3.0))

    assert first_id != second_id

    lines = (tmp_path / "shared_id" / "trace.jsonl").read_text().splitlines()
    span_ids = [json.loads(line)["span_id"] for line in lines]
    assert len(span_ids) == len(set(span_ids)), "span ids collided across processes"


def test_span_counter_seeds_from_existing_trace_line_count(tmp_path):
    first = Run("shared_id", tmp_path)
    for _ in range(3):
        first.emit(Span(span_id=first._next_span_id(), name="s", parent_id=None,
                         start_ms=0.0, end_ms=1.0))

    second = Run("shared_id", tmp_path)
    assert second._counter == 3
    assert second._next_span_id() == "s000004"

"""Append-only span tracing to runs/<run_id>/trace.jsonl.

Why home-grown rather than OTel on day one: the trace here is also the eval
artifact and the audit record, so it needs stable, inspectable, diffable JSONL
that survives without a collector. The Span shape is deliberately OTel-compatible
(trace/span/parent ids, attributes, status) so an exporter is an afternoon's work
when Azure Monitor or Langfuse becomes the target.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from isc.common.ids import new_run_id

_current_run: ContextVar[Run | None] = ContextVar("isc_current_run", default=None)
_current_span: ContextVar[str | None] = ContextVar("isc_current_span", default=None)


@dataclass
class Span:
    span_id: str
    name: str
    parent_id: str | None
    start_ms: float
    end_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        return None if self.end_ms is None else self.end_ms - self.start_ms


class Run:
    """One execution of one or more stages. Owns the trace file.

    A run id is not always first touched by this process: `make slice`
    passes ONE id to five separate `isc <stage>` invocations (five separate
    processes) so `runs/<id>/` holds the whole pipeline's trace together,
    not five scattered directories. Both `_counter` and `totals` are seeded
    from whatever is already on disk for this id, not zero, so a later
    stage sharing an id with an earlier one does not collide span ids in
    the same trace.jsonl (each process restarting at s000001 would produce
    duplicate ids the moment two stages share a run) and does not silently
    discard the earlier stage's cost the moment its own summarise() call
    overwrites summary.json with only what THIS process itself spent.
    """

    def __init__(self, run_id: str, root: Path) -> None:
        self.run_id = run_id
        self.dir = root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "trace.jsonl"
        self._lock = threading.Lock()
        self._counter = self._count_existing_spans()
        self.totals: dict[str, float] = self._load_existing_totals()

    def _count_existing_spans(self) -> int:
        if not self._path.exists():
            return 0
        with self._path.open() as fh:
            return sum(1 for _ in fh)

    def _load_existing_totals(self) -> dict[str, float]:
        summary_path = self.dir / "summary.json"
        if not summary_path.exists():
            return {}
        return json.loads(summary_path.read_text()).get("totals", {})

    def _next_span_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"s{self._counter:06d}"

    def emit(self, span: Span) -> None:
        record = asdict(span)
        record["run_id"] = self.run_id
        record["duration_ms"] = span.duration_ms
        line = json.dumps(record, default=str, separators=(",", ":"))
        with self._lock, self._path.open("a") as fh:
            fh.write(line + "\n")

    def add_total(self, key: str, value: float) -> None:
        """Roll up cost/tokens across the run; written by summarise()."""
        with self._lock:
            self.totals[key] = self.totals.get(key, 0.0) + value

    def summarise(self) -> Path:
        out = self.dir / "summary.json"
        out.write_text(json.dumps({"run_id": self.run_id, "totals": self.totals}, indent=2))
        return out

    def artifact_dir(self, stage: str) -> Path:
        """Stages are file-to-file; each writes into its own directory."""
        d = self.dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d


def start_run(root: Path, run_id: str | None = None) -> Run:
    run = Run(run_id or os.environ.get("ISC_RUN_ID") or new_run_id(), root)
    _current_run.set(run)
    return run


def current_run() -> Run | None:
    return _current_run.get()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Records duration, parentage, and exceptions. A no-op if no run is active."""
    run = _current_run.get()
    if run is None:
        yield Span("s0", name, None, time.time() * 1000, attributes=attributes)
        return

    sp = Span(
        span_id=run._next_span_id(),
        name=name,
        parent_id=_current_span.get(),
        start_ms=time.time() * 1000,
        attributes=dict(attributes),
    )
    token = _current_span.set(sp.span_id)
    try:
        yield sp
    except Exception as exc:
        sp.status = "error"
        sp.attributes["error.type"] = type(exc).__name__
        sp.attributes["error.message"] = str(exc)[:500]
        raise
    finally:
        _current_span.reset(token)
        sp.end_ms = time.time() * 1000
        run.emit(sp)

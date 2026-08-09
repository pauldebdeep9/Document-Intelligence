"""Content-addressed response cache.

Eval harnesses re-run the same prompts against the same documents dozens of times.
Without this the iteration loop is priced in dollars and minutes rather than
milliseconds. Keyed on the full request, so any prompt or parameter change misses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from isc.common.ids import cache_key
from isc.common.logging import get_logger

log = get_logger("llm.cache")


class ResponseCache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        namespace, digest = key.split(":", 1)
        d = self.root / namespace / digest[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.json"

    def key_for(self, namespace: str, request: dict[str, Any]) -> str:
        return cache_key(namespace, request)

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("corrupt cache entry %s; ignoring", p)
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._path(key).write_text(json.dumps(value, default=str))

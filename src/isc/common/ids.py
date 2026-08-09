"""Deterministic identifiers.

Every id is a pure function of content or of a stable path. Re-running a stage on
unchanged input must produce identical ids, or delta sync and eval diffing break.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

_HASH_LEN = 16


def _h(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return digest[:_HASH_LEN]


def content_hash(data: bytes) -> str:
    """Full-fidelity hash used for dedupe and cache keys."""
    return hashlib.sha256(data).hexdigest()


def doc_id(source_uri: str, content_sha256: str) -> str:
    """Stable across re-ingest of the same bytes from the same location.

    Same bytes at a different URI is a distinct document: in SharePoint the location
    carries the permissions, so it is not safe to collapse them.
    """
    return f"doc_{_h(source_uri, content_sha256)}"


def chunk_id(document_id: str, ordinal: int, text: str) -> str:
    """Includes text so a re-chunk with different settings does not silently alias."""
    return f"chk_{_h(document_id, str(ordinal), text)}"


def span_id(document_id: str, page: int, bbox: tuple[float, float, float, float]) -> str:
    return f"spn_{_h(document_id, str(page), ','.join(f'{v:.2f}' for v in bbox))}"


def cache_key(namespace: str, payload: Any) -> str:
    """Canonical JSON so key ordering never changes the key."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{namespace}:{_h(blob)}"


def new_run_id(prefix: str = "run") -> str:
    """Not deterministic by design: identifies one execution, not one input."""
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

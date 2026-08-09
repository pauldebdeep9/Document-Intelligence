from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from isc.models.document import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_documents_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS ix_documents_sha ON documents(content_sha256);

CREATE TABLE IF NOT EXISTS records (
    document_id TEXT PRIMARY KEY,
    doc_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- HITL queue. Rows land here from extract/ when a field routes to review.
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    weakest_signal TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    resolved_value TEXT,
    reviewer TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_review_status ON review_queue(status, confidence);
"""


class SqliteDocStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert_document(self, doc: Document) -> None:
        self._conn.execute(
            "INSERT INTO documents (id, source_uri, content_sha256, doc_type, payload) "
            "VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "payload=excluded.payload, doc_type=excluded.doc_type, "
            "updated_at=CURRENT_TIMESTAMP",
            (doc.id, doc.source_uri, doc.content_sha256, doc.doc_type, doc.model_dump_json()),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT payload FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
        return Document.model_validate_json(row["payload"]) if row else None

    def list_documents(self, doc_type: str | None = None) -> list[str]:
        if doc_type:
            rows = self._conn.execute(
                "SELECT id FROM documents WHERE doc_type=? ORDER BY id", (doc_type,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT id FROM documents ORDER BY id").fetchall()
        return [r["id"] for r in rows]

    def seen_hash(self, sha: str) -> bool:
        """Dedupe support for ingest/ delta sync."""
        return self._conn.execute(
            "SELECT 1 FROM documents WHERE content_sha256=? LIMIT 1", (sha,)
        ).fetchone() is not None

    def upsert_record(self, doc_id: str, doc_type: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO records (document_id, doc_type, payload) VALUES (?,?,?) "
            "ON CONFLICT(document_id) DO UPDATE SET payload=excluded.payload, "
            "updated_at=CURRENT_TIMESTAMP",
            (doc_id, doc_type, json.dumps(payload, default=str)),
        )
        self._conn.commit()

    def get_record(self, doc_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload FROM records WHERE document_id=?", (doc_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def enqueue_review(
        self, doc_id: str, field_name: str, confidence: float, weakest_signal: str = ""
    ) -> None:
        self._conn.execute(
            "INSERT INTO review_queue (document_id, field_name, confidence, weakest_signal) "
            "VALUES (?,?,?,?)",
            (doc_id, field_name, confidence, weakest_signal),
        )
        self._conn.commit()

    def open_reviews(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM review_queue WHERE status='open' ORDER BY confidence ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

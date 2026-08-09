from __future__ import annotations

from pathlib import Path


class LocalBlobStore:
    """Filesystem-backed BlobStore. Keys are relative paths under `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes blob root: {key}")
        return p

    def put(self, key: str, data: bytes) -> str:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    def get(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()
        )

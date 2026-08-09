"""The provider port is only real if nothing bypasses it.

This test is what turns a naming convention into an enforced boundary, and it is
what makes the eventual Azure OpenAI migration a config change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "isc"
ALLOWED = {"llm/openai_client.py", "llm/azure_openai_client.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.name))
def test_no_direct_provider_imports(path: Path):
    rel = path.relative_to(SRC).as_posix()
    if rel in ALLOWED:
        return
    assert "openai" not in _imports(path), (
        f"{rel} imports openai directly; go through isc.llm.ports instead"
    )


def test_models_do_not_import_storage():
    """Domain models stay storage-agnostic so they can be reused in P2 and P3."""
    for path in (SRC / "models").rglob("*.py"):
        assert "isc.storage" not in path.read_text(), f"{path} imports storage"

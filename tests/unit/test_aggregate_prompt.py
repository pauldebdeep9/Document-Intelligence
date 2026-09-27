"""The planner prompt must not name anything from the corpus. Its examples
use invented suppliers and part numbers, so a model that parrots an example
cannot land on a real entity, and the gold questions cannot leak into the
prompt that is later scored against them (ADR 0011, Consequences)."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPT = REPO / "config" / "prompts" / "aggregate" / "query_plan.v1.md"
MASTERS = REPO / "data" / "masters"
GOLD_EXTRACTION = REPO / "data" / "gold" / "extraction"


def _corpus_entities() -> dict[str, str]:
    """Every corpus string the prompt must not contain -> where it comes from."""
    found: dict[str, str] = {}
    for s in json.loads((MASTERS / "suppliers.json").read_text()):
        found[s["name"]] = "supplier name"
        first = s["name"].split()[0]
        if len(first) >= 4:
            found.setdefault(first, "supplier name first token")
        found[s["supplier_id"]] = "supplier_id"
    for master in ("parts.json", "unmastered_parts.json"):
        for p in json.loads((MASTERS / master).read_text()):
            found[p["part_number"]] = f"part_number ({master})"
    for path in sorted(GOLD_EXTRACTION.glob("*.json")):
        po_number = json.loads(path.read_text())["raw"].get("po_number")
        if po_number:
            found[str(po_number)] = f"po_number ({path.name})"
    return found


def test_planner_prompt_names_no_corpus_entity():
    text = PROMPT.read_text().casefold()
    entities = _corpus_entities()
    assert len(entities) > 20, "corpus masters/gold not found -- the check would be vacuous"
    offenders = [
        f"{term!r} ({source})" for term, source in sorted(entities.items())
        if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", text)
    ]
    assert offenders == [], f"{PROMPT.name} names corpus entities: {offenders}"

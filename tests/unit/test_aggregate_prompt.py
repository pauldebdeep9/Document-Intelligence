"""The planner prompt must not name anything from the corpus. Its examples
use invented suppliers and part numbers, so a model that parrots an example
cannot land on a real entity, and the gold questions cannot leak into the
prompt that is later scored against them (ADR 0011, Consequences)."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO / "config" / "prompts" / "aggregate"
# Every planner prompt version, loaded or not (NOTES files are prose, never
# sent to a model).
PROMPTS = sorted(p for p in PROMPT_DIR.glob("query_plan.v*.md") if "NOTES" not in p.name)
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
    assert len(PROMPTS) >= 2, f"planner prompt versions not found in {PROMPT_DIR}"
    entities = _corpus_entities()
    assert len(entities) > 20, "corpus masters/gold not found -- the check would be vacuous"
    offenders = [
        f"{prompt.name}: {term!r} ({source})"
        for prompt in PROMPTS
        for term, source in sorted(entities.items())
        if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", prompt.read_text().casefold())
    ]
    assert offenders == [], f"planner prompts name corpus entities: {offenders}"

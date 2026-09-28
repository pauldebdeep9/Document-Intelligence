"""Build the planner gold: the plan each retrieval-gold question should get.

THE INVARIANT, same as gen_gold.py's (the P1-08 rule): every expected plan
is derived from gold STRUCTURE -- subtype, gold_answer, source_documents and
the gold extraction records -- never by parsing the question text and never
from a model. A planner scored against plans a planner wrote, or against
identifiers scraped from the very wording it is asked to copy from, would be
graded on its own reading. The question text is used only to CHECK that each
derived identifier appears in it verbatim, which is what the planner prompt
requires.

Derivation:
  cross_document, gold_answer {"total", "currency", ...}  -> total_spend
      supplier = the gold supplier_name all source documents share
  cross_document, gold_answer [ {document, line_number, ...} ]  -> part_prices
      part_number = the part on every referenced gold line
  ambiguous, every gold_answer entry has total_amount  -> total_spend
      supplier = the longest common whole-token prefix of the entries'
      supplier_names ("Kestrel Industrial"); no currency -- the gold spans two
      entities, and the question does not name one
  everything else -> none

Every aggregate case is then run through the real validate_plan(): an
expected plan the validator would reject is a bug in this file, not a
planner miss.

Output is deterministic (sorted keys, gold order, trailing newline), so
"generate twice, diff bytes" holds and a gold change shows up as a diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from isc.aggregate.plan import QueryPlanRaw, validate_plan
from isc.aggregate.resolve import resolve_mention

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "gold" / "retrieval" / "questions.json"
EXTRACTION = ROOT / "data" / "gold" / "extraction"
MASTERS = ROOT / "data" / "masters"
OUT = ROOT / "data" / "gold" / "planner" / "cases.json"

EXPECTED_AGGREGATE, EXPECTED_NONE = 9, 47


class GoldError(Exception):
    """The gold is not shaped the way this derivation assumes -- fatal."""


def _none_plan() -> dict[str, Any]:
    return {"operation": "none", "supplier": None, "part_number": None, "currency": None}


def load_extraction(extraction_dir: Path = EXTRACTION) -> dict[str, dict[str, Any]]:
    """document file name ("po_000.pdf") -> its gold raw extraction."""
    out = {}
    for path in sorted(extraction_dir.glob("*.json")):
        gold = json.loads(path.read_text())
        out[gold["document"]] = gold["raw"]
    return out


def _common_token_prefix(names: list[str]) -> str:
    split = [n.split() for n in names]
    prefix: list[str] = []
    for tokens in zip(*split, strict=False):
        if len({t.casefold() for t in tokens}) != 1:
            break
        prefix.append(tokens[0])
    return " ".join(prefix)


def derive_expected(q: dict[str, Any], extraction: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The expected QueryPlanRaw fields for one gold question, from structure."""
    gold = q["gold_answer"]
    plan = _none_plan()

    if q["subtype"] == "cross_document" and isinstance(gold, dict) and "total" in gold:
        names = {extraction[doc]["supplier_name"] for doc in q["source_documents"]}
        if len(names) != 1:
            raise GoldError(f"{q['id']}: source documents disagree on supplier: {sorted(names)}")
        plan.update(operation="total_spend", supplier=names.pop(), currency=gold["currency"])

    elif (q["subtype"] == "cross_document" and isinstance(gold, list)
          and all("line_number" in e for e in gold)):
        parts = set()
        for entry in gold:
            lines = [ln for ln in extraction[entry["document"]]["lines"]
                     if ln["line_number"] == entry["line_number"]]
            if len(lines) != 1:
                raise GoldError(f"{q['id']}: {entry['document']} line {entry['line_number']} "
                                f"matches {len(lines)} gold lines")
            parts.add(lines[0]["part_number"])
        if len(parts) != 1:
            raise GoldError(f"{q['id']}: referenced lines carry different parts: {sorted(parts)}")
        plan.update(operation="part_prices", part_number=parts.pop())

    elif (q["subtype"] == "ambiguous" and isinstance(gold, list) and gold
          and all("total_amount" in e for e in gold)):
        prefix = _common_token_prefix([e["supplier_name"] for e in gold])
        if not prefix:
            raise GoldError(f"{q['id']}: gold supplier names share no leading token")
        plan.update(operation="total_spend", supplier=prefix)

    return plan


def build(questions_path: Path = QUESTIONS, extraction_dir: Path = EXTRACTION,
          masters_dir: Path = MASTERS) -> dict[str, Any]:
    raw_bytes = questions_path.read_bytes()
    questions = json.loads(raw_bytes)["questions"]
    extraction = load_extraction(extraction_dir)
    cases = []
    for q in questions:
        expected = derive_expected(q, extraction)
        supplier_ids = (list(resolve_mention(expected["supplier"], masters_dir))
                        if expected["operation"] == "total_spend" else [])
        cases.append({"id": q["id"], "subtype": q["subtype"], "text": q["text"],
                      "expected": expected, "expected_supplier_ids": supplier_ids})
    return {
        "provenance": {
            "questions_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "generator": "scripts/gen_planner_gold.py",
        },
        "cases": cases,
    }


def check(out: dict[str, Any], masters_dir: Path = MASTERS) -> list[str]:
    """Every problem with a built planner gold; [] when it is usable."""
    problems = []
    for case in out["cases"]:
        exp, text, cid = case["expected"], case["text"], case["id"]
        if exp["supplier"] is not None and exp["supplier"].casefold() not in text.casefold():
            problems.append(f"{cid}: supplier {exp['supplier']!r} is not in the question")
        for field in ("part_number", "currency"):
            if exp[field] is not None and exp[field] not in text:
                problems.append(f"{cid}: {field} {exp[field]!r} is not in the question")
        if exp["operation"] == "none":
            continue
        plan, reason = validate_plan(QueryPlanRaw(**exp), text, masters_dir)
        if plan is None:
            problems.append(f"{cid}: validate_plan rejects the expected plan: {reason}")
        elif list(plan.supplier_ids) != case["expected_supplier_ids"]:
            problems.append(f"{cid}: validate_plan resolves {list(plan.supplier_ids)}, "
                            f"expected {case['expected_supplier_ids']}")
    ops = [c["expected"]["operation"] for c in out["cases"]]
    n_none = ops.count("none")
    if (len(ops) - n_none, n_none) != (EXPECTED_AGGREGATE, EXPECTED_NONE):
        problems.append(f"expected {EXPECTED_AGGREGATE} aggregate / {EXPECTED_NONE} none, "
                        f"got {len(ops) - n_none} / {n_none}")
    return problems


def render(out: dict[str, Any]) -> str:
    return json.dumps(out, indent=2, sort_keys=True) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--extraction-gold", type=Path, default=EXTRACTION)
    ap.add_argument("--masters", type=Path, default=MASTERS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    try:
        out = build(args.questions, args.extraction_gold, args.masters)
    except GoldError as exc:
        sys.exit(f"gen_planner_gold: {exc}")
    problems = check(out, args.masters)
    if problems:
        sys.exit("gen_planner_gold: inconsistent planner gold:\n  " + "\n  ".join(problems))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(out))

    by_op: dict[str, int] = {}
    for c in out["cases"]:
        by_op[c["expected"]["operation"]] = by_op.get(c["expected"]["operation"], 0) + 1
    print(f"wrote {len(out['cases'])} cases -> {args.out.relative_to(ROOT)}")
    print("  by operation:", dict(sorted(by_op.items())))


if __name__ == "__main__":
    main()

"""Planner routing eval: does the planner produce the expected plan for every
question, and route everything else to "none"?

Measures plan_question() -- the exact code path RecordAnswerer runs -- over
the expected-plan cases (data/gold/planner/cases.json, gold-derived;
handwritten.json, paraphrases and near-misses). No retrieval, no answering.

Outcomes (E = expected operation, P = the validated plan):

  correct_none   E none, P None
  misroute_in    E none, P not None -- the expensive direction: a working
                 single-PO answer replaced by a records answer to a different
                 question. Reported as its own headline, never folded into
                 an accuracy figure.
  misroute_out   E aggregate, P None -- the planner said none, or a guard in
                 validate_plan() rejected its plan (the reason is kept).
                 Reproduces P1 chunk-path behaviour; cheap by comparison.
  exact          E aggregate, P matches on operation, supplier_ids (as a
                 set), part_number and currency
  wrong_plan     anything else
  error          the planner call raised something plan_question() does not
                 handle; recorded, never aborts the run

`widened` flags a plan that resolves to MORE suppliers than expected -- the
truncation failure (both Kestrel entities for a question about one).

Every rate is a "k/n" string (the EV-02 rule): no bare percentages.

Token counts and cost are the FINAL attempt's only: when parse_structured()
needed repair rounds, the earlier calls' usage is not included here (the
run's trace totals still carry it).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from isc.aggregate.plan import QueryPlan
from isc.aggregate.planner import PlanAttempt, plan_question
from isc.llm.cost import estimate_usd
from isc.llm.ports import ChatModel

OUTCOMES = ("exact", "correct_none", "wrong_plan", "misroute_in", "misroute_out", "error")
CORRECT = frozenset({"exact", "correct_none"})
_PLAN_FIELDS = ("operation", "supplier_ids", "part_number", "currency")


class CaseFileError(ValueError):
    """A planner case file is not shaped the way the eval expects."""


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Cases from a planner case file, each tagged with its source ("gold"
    when it carries a gold subtype, "handwritten" when it carries a class)
    and group (that subtype or class)."""
    cases = json.loads(path.read_text())["cases"]
    out = []
    for c in cases:
        missing = {"id", "text", "expected", "expected_supplier_ids"} - c.keys()
        if missing:
            raise CaseFileError(f"{path.name}: case {c.get('id')!r} lacks {sorted(missing)}")
        if {"operation", "supplier", "part_number", "currency"} - c["expected"].keys():
            raise CaseFileError(f"{path.name}: case {c['id']!r} has an incomplete expected plan")
        if "subtype" in c:
            source, group = "gold", c["subtype"]
        elif "class" in c:
            source, group = "handwritten", c["class"]
        else:
            raise CaseFileError(f"{path.name}: case {c['id']!r} has neither subtype nor class")
        out.append({**c, "source": source, "group": group})
    return out


@dataclass(frozen=True)
class PlannerOutcome:
    id: str
    source: str
    group: str
    expected_operation: str
    got_operation: str
    raw_operation: str | None
    outcome: str
    reason: str
    widened: bool
    prompt_tokens: int
    completion_tokens: int
    usd: float
    expected_plan: dict[str, Any]
    got_plan: dict[str, Any] | None


def _expected_plan(case: dict[str, Any]) -> dict[str, Any]:
    e = case["expected"]
    return {"operation": e["operation"], "supplier_ids": sorted(case["expected_supplier_ids"]),
            "part_number": e["part_number"], "currency": e["currency"]}


def _got_plan(plan: QueryPlan) -> dict[str, Any]:
    return {"operation": plan.operation.value, "supplier_ids": sorted(plan.supplier_ids),
            "part_number": plan.part_number, "currency": plan.currency}


def classify(case: dict[str, Any], attempt: PlanAttempt) -> PlannerOutcome:
    expected = _expected_plan(case)
    plan = attempt.plan
    got = _got_plan(plan) if plan is not None else None
    if expected["operation"] == "none":
        outcome = "correct_none" if plan is None else "misroute_in"
    elif got is None:
        outcome = "misroute_out"
    else:
        outcome = "exact" if got == expected else "wrong_plan"
    usage = attempt.result.usage if attempt.result is not None else None
    return PlannerOutcome(
        id=case["id"], source=case["source"], group=case["group"],
        expected_operation=expected["operation"],
        got_operation=got["operation"] if got is not None else "none",
        raw_operation=attempt.raw.operation if attempt.raw is not None else None,
        outcome=outcome, reason=attempt.reason,
        widened=plan is not None and set(plan.supplier_ids) > set(expected["supplier_ids"]),
        prompt_tokens=usage.prompt_tokens if usage is not None else 0,
        completion_tokens=usage.completion_tokens if usage is not None else 0,
        usd=(estimate_usd(attempt.result.model, attempt.result.usage)
             if attempt.result is not None else 0.0),
        expected_plan=expected, got_plan=got,
    )


def error_outcome(case: dict[str, Any], exc: BaseException) -> PlannerOutcome:
    return PlannerOutcome(
        id=case["id"], source=case["source"], group=case["group"],
        expected_operation=case["expected"]["operation"], got_operation="none",
        raw_operation=None, outcome="error", reason=f"{type(exc).__name__}: {exc}",
        widened=False, prompt_tokens=0, completion_tokens=0, usd=0.0,
        expected_plan=_expected_plan(case), got_plan=None,
    )


def evaluate(chat: ChatModel, cases: list[dict[str, Any]],
             masters_dir: Path) -> list[PlannerOutcome]:
    """One planner call per case, sequentially. An exception on one case is
    that case's outcome; the rest still run."""
    outcomes = []
    for case in cases:
        try:
            outcomes.append(classify(case, plan_question(chat, case["text"], masters_dir)))
        except Exception as exc:  # noqa: BLE001 -- recorded per case, by design
            outcomes.append(error_outcome(case, exc))
    return outcomes


def _kn(k: int, n: int) -> str:
    return f"{k}/{n}"


def _reason_pattern(reason: str) -> str:
    """Guard reasons with the planner's own values blanked, so they tally."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", reason)


def summarise(outcomes: list[PlannerOutcome]) -> dict[str, Any]:
    n = len(outcomes)
    by_outcome = Counter(o.outcome for o in outcomes)

    def cell(rows: list[PlannerOutcome]) -> dict[str, str]:
        counts = Counter(o.outcome for o in rows)
        out = {"correct": _kn(sum(counts[x] for x in CORRECT), len(rows))}
        out.update({k: _kn(counts[k], len(rows)) for k in OUTCOMES if counts[k]})
        return out

    gold = [o for o in outcomes if o.source == "gold"]
    headline_rows = {
        "gold aggregate": [o for o in gold if o.expected_operation != "none"],
        "gold none": [o for o in gold if o.expected_operation == "none"],
    }
    for group in sorted({o.group for o in outcomes if o.source == "handwritten"}):
        headline_rows[f"handwritten {group}"] = [
            o for o in outcomes if o.source == "handwritten" and o.group == group]

    groups: dict[str, list[PlannerOutcome]] = {}
    for o in outcomes:
        groups.setdefault(f"{o.source}/{o.group}", []).append(o)

    called = [o for o in outcomes if o.outcome != "error"]
    prompt_tokens = sum(o.prompt_tokens for o in outcomes)
    completion_tokens = sum(o.completion_tokens for o in outcomes)
    usd = sum(o.usd for o in outcomes)
    rejected = [o for o in outcomes
                if o.got_operation == "none" and o.raw_operation not in (None, "none")]

    return {
        "n": n,
        "misroute_in": {"rate": _kn(by_outcome["misroute_in"], n),
                        "ids": [o.id for o in outcomes if o.outcome == "misroute_in"]},
        "correct": _kn(sum(by_outcome[x] for x in CORRECT), n),
        "by_outcome": {k: _kn(by_outcome[k], n) for k in OUTCOMES},
        "headline": {name: cell(rows) for name, rows in headline_rows.items() if rows},
        "by_group": {name: cell(rows) for name, rows in sorted(groups.items())},
        "wrong_plan": [
            {"id": o.id, "diffs": {
                f: {"expected": o.expected_plan[f], "got": (o.got_plan or {}).get(f)}
                for f in _PLAN_FIELDS if o.expected_plan[f] != (o.got_plan or {}).get(f)}}
            for o in outcomes if o.outcome == "wrong_plan"],
        "misroute_out": [{"id": o.id, "planner_said": o.raw_operation, "reason": o.reason}
                         for o in outcomes if o.outcome == "misroute_out"],
        "guard_rejections": dict(sorted(Counter(_reason_pattern(o.reason)
                                                for o in rejected).items())),
        "widened": [o.id for o in outcomes if o.widened],
        "errors": [{"id": o.id, "reason": o.reason} for o in outcomes if o.outcome == "error"],
        "cost": {
            "calls": len(called),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd": usd,
            "mean_prompt_tokens_per_call": prompt_tokens / len(called) if called else 0.0,
            "mean_completion_tokens_per_call": completion_tokens / len(called) if called else 0.0,
            "mean_usd_per_call": usd / len(called) if called else 0.0,
        },
    }


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    misroute_ids = summary["misroute_in"]["ids"]
    lines = [
        "# Planner routing eval",
        "",
        f"- run: `{meta.get('run_id', '')}` · model: `{meta.get('model', '')}` · "
        f"cache enabled: {meta.get('cache_enabled')} · {meta.get('timestamp', '')}",
        f"- prompt: `{meta.get('prompt', {}).get('path', '')}` "
        f"(sha256 {str(meta.get('prompt', {}).get('sha256', ''))[:12]})",
        "",
        f"**Misroute-in: {summary['misroute_in']['rate']}**"
        + (f" — {', '.join(misroute_ids)}" if misroute_ids else ""),
        "",
        f"Correct (exact or correct_none): {summary['correct']}",
        "",
        "| set | correct | outcomes |",
        "|---|---|---|",
    ]
    for name, c in summary["headline"].items():
        rest = ", ".join(f"{k} {v}" for k, v in c.items() if k != "correct")
        lines.append(f"| {name} | {c['correct']} | {rest} |")
    lines += ["", "## By group", "", "| group | correct | outcomes |", "|---|---|---|"]
    for name, c in summary["by_group"].items():
        rest = ", ".join(f"{k} {v}" for k, v in c.items() if k != "correct")
        lines.append(f"| {name} | {c['correct']} | {rest} |")
    lines += ["", "## Wrong plans", ""]
    lines += [f"- {w['id']}: " + "; ".join(f"{f} expected {d['expected']!r}, got {d['got']!r}"
                                           for f, d in w["diffs"].items())
              for w in summary["wrong_plan"]] or ["- none"]
    lines += ["", "## Misroute-out", ""]
    lines += [f"- {m['id']}: planner said {m['planner_said']!r}; {m['reason']}"
              for m in summary["misroute_out"]] or ["- none"]
    lines += ["", "## Guard rejections (plan proposed, validate_plan refused)", ""]
    lines += [f"- {k}: {v}" for k, v in summary["guard_rejections"].items()] or ["- none"]
    lines += ["", f"Widened (resolved to more suppliers than expected): "
                  f"{', '.join(summary['widened']) or 'none'}"]
    if summary["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e['id']}: {e['reason']}" for e in summary["errors"]]
    cost = summary["cost"]
    lines += ["", "## Cost (final attempt per case; repair rounds not included)", "",
              f"- calls: {cost['calls']} · prompt tokens {cost['prompt_tokens']} · "
              f"completion tokens {cost['completion_tokens']} · ${cost['usd']:.6f}",
              f"- per call: {cost['mean_prompt_tokens_per_call']:.1f} prompt · "
              f"{cost['mean_completion_tokens_per_call']:.1f} completion · "
              f"${cost['mean_usd_per_call']:.6f}", ""]
    return "\n".join(lines)


def outcomes_as_json(outcomes: list[PlannerOutcome]) -> list[dict[str, Any]]:
    return [asdict(o) for o in outcomes]

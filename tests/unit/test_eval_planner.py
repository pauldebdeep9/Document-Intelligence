"""eval/planner.py and scripts/eval_planner.py, offline: a scripted planner
maps question text to a canned plan and a canned Usage. No registry, no
network -- the live run is AG-08 prompt 4."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import isc.llm.registry as registry
from isc.eval.planner import evaluate, load_cases, summarise
from isc.llm.cost import estimate_usd
from isc.llm.ports import LLMResult, Usage

ROOT = Path(__file__).resolve().parents[2]
MASTERS = ROOT / "data" / "masters"
GOLD = ROOT / "data" / "gold" / "planner" / "cases.json"
HANDWRITTEN = ROOT / "data" / "gold" / "planner" / "handwritten.json"

MODEL = "gpt-4o-mini"
USAGE = Usage(prompt_tokens=600, completion_tokens=20)
AG, PNEU, OMRON = "V100781", "V100782", "V102337"
NONE = {"operation": "none", "supplier": None, "part_number": None, "currency": None}


def _raw(operation="none", supplier=None, part_number=None, currency=None):
    return {"operation": operation, "supplier": supplier, "part_number": part_number,
            "currency": currency}


class ScriptedPlanner:
    """question text -> raw plan dict, or an exception to raise."""

    def __init__(self, script: dict[str, dict | Exception]) -> None:
        self.script = script
        self.calls = 0

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        self.calls += 1
        answer = self.script[messages[-1].content]
        if isinstance(answer, Exception):
            raise answer
        return LLMResult(text=json.dumps(answer), model=MODEL, usage=USAGE)


def _case(cid, text, expected, ids=(), group="paraphrase"):
    return {"id": cid, "text": text, "source": "handwritten", "group": group,
            "expected": expected, "expected_supplier_ids": list(ids)}


Q_OMRON = "What did we spend with Omron Electronics Asia in total, in SGD?"
Q_PARTS = "Which parts did we buy from Omron Electronics Asia?"
Q_AG = "What did we spend with Kestrel Industrial AG in total, in USD?"
Q_TWICE = ("What did we spend with Kestrel Industrial in total — "
           "I mean Kestrel Industrial AG — in USD?")
Q_PO = "What is the order total on PO 4522345741?"

OMRON_SGD = _raw("total_spend", "Omron Electronics Asia", currency="SGD")
AG_USD = _raw("total_spend", "Kestrel Industrial AG", currency="USD")


def _six():
    """One hand-built case per outcome class (misroute_out twice: said none,
    and guard-rejected)."""
    cases = [
        _case("exact", Q_OMRON, OMRON_SGD, [OMRON]),
        _case("wrong", Q_OMRON + " ", OMRON_SGD, [OMRON]),
        _case("in", Q_PARTS, NONE, group="near_miss"),
        _case("out_none", Q_OMRON + "  ", OMRON_SGD, [OMRON]),
        _case("out_guard", Q_AG, AG_USD, [AG]),
        _case("none_ok", Q_PO, NONE, group="near_miss"),
    ]
    script = {
        Q_OMRON: OMRON_SGD,
        Q_OMRON + " ": _raw("total_spend", "Omron Electronics Asia"),    # currency dropped
        Q_PARTS: _raw("total_spend", "Omron Electronics Asia"),          # valid, but not asked
        Q_OMRON + "  ": _raw(),                                          # planner said none
        Q_AG: _raw("total_spend", "Kestrel Industrial", currency="USD"),  # truncated: guard
        Q_PO: _raw(),
    }
    return cases, ScriptedPlanner(script)


def test_each_outcome_class():
    cases, chat = _six()
    got = {o.id: o for o in evaluate(chat, cases, MASTERS)}
    assert {k: o.outcome for k, o in got.items()} == {
        "exact": "exact", "wrong": "wrong_plan", "in": "misroute_in",
        "out_none": "misroute_out", "out_guard": "misroute_out", "none_ok": "correct_none"}
    assert got["out_none"].raw_operation == "none"
    assert got["out_guard"].raw_operation == "total_spend"
    assert "appears truncated" in got["out_guard"].reason


def test_guard_rejected_case_keeps_the_models_raw_plan():
    """planner.json must show what the model actually said, not only the
    validated plan: a rejected plan has none, and its reason may not name
    every field."""
    cases, chat = _six()
    got = {o.id: o for o in evaluate(chat, cases, MASTERS)}
    assert got["out_guard"].raw_plan == {"operation": "total_spend",
                                         "supplier": "Kestrel Industrial",
                                         "part_number": None, "currency": "USD"}
    assert got["out_guard"].got_plan is None
    summary = summarise(list(got.values()))
    [row] = [c for c in summary["non_pass"] if c["id"] == "out_guard"]
    assert row["raw_plan"]["supplier"] == "Kestrel Industrial"
    assert row["raw_plan"]["currency"] == "USD"


def test_widened_plan_is_flagged():
    """Validates (the first mention is followed by "in", not a suffix), but
    resolves both Kestrel entities for a question about one."""
    case = _case("twice", Q_TWICE, AG_USD, [AG], group="truncation")
    chat = ScriptedPlanner({Q_TWICE: _raw("total_spend", "Kestrel Industrial", currency="USD")})
    [o] = evaluate(chat, [case], MASTERS)
    assert o.outcome == "wrong_plan" and o.widened
    assert o.got_plan["supplier_ids"] == [AG, PNEU]


def test_summary_cells_are_k_of_n():
    cases, chat = _six()
    summary = summarise(evaluate(chat, cases, MASTERS))
    assert summary["misroute_in"] == {"rate": "1/6", "ids": ["in"]}
    assert summary["correct"] == "2/6"
    assert summary["by_outcome"] == {"exact": "1/6", "correct_none": "1/6", "wrong_plan": "1/6",
                                     "misroute_in": "1/6", "misroute_out": "2/6", "error": "0/6"}
    assert summary["headline"]["handwritten paraphrase"]["correct"] == "1/4"
    assert summary["headline"]["handwritten near_miss"]["correct"] == "1/2"
    assert summary["wrong_plan"] == [
        {"id": "wrong", "diffs": {"currency": {"expected": "SGD", "got": None}}}]
    assert [m["id"] for m in summary["misroute_out"]] == ["out_none", "out_guard"]
    assert summary["guard_rejections"] == {"supplier mention '…' appears truncated": 1}


def test_cost_is_estimated_per_call_and_totals_add_up():
    cases, chat = _six()
    outcomes = evaluate(chat, cases, MASTERS)
    per_call = estimate_usd(MODEL, USAGE)
    assert per_call > 0
    assert all(o.usd == per_call and o.prompt_tokens == 600 for o in outcomes)
    cost = summarise(outcomes)["cost"]
    assert cost["calls"] == 6
    assert cost["prompt_tokens"] == 3600 and cost["completion_tokens"] == 120
    assert cost["usd"] == pytest.approx(6 * per_call)
    assert cost["mean_usd_per_call"] == pytest.approx(per_call)


def test_an_exception_on_one_case_does_not_abort_the_run():
    cases, chat = _six()
    chat.script[Q_PO] = ConnectionError("provider down")
    outcomes = {o.id: o for o in evaluate(chat, cases, MASTERS)}
    assert outcomes["none_ok"].outcome == "error"
    assert "ConnectionError: provider down" in outcomes["none_ok"].reason
    assert outcomes["exact"].outcome == "exact" and len(outcomes) == 6
    assert chat.calls == 6
    assert summarise(list(outcomes.values()))["cost"]["calls"] == 5


def test_script_refuses_to_run_with_the_cache_on(monkeypatch, capsys):
    from isc.common.config import get_settings
    from scripts import eval_planner

    def no_model(*_a, **_k):
        raise AssertionError("planner model built despite the cache guard")

    monkeypatch.setattr(registry, "get_chat_model", no_model)
    monkeypatch.setenv("ISC_CACHE__ENABLED", "true")
    get_settings.cache_clear()
    assert eval_planner.main([]) != 0
    assert "cache is enabled" in capsys.readouterr().err


def test_both_case_files_agree_with_the_classifier():
    """A planner that returns each case's own expected plan scores 81/81: the
    case files, validate_plan() and classify() agree end to end."""
    cases = load_cases(GOLD) + load_cases(HANDWRITTEN)
    script = {c["text"]: c["expected"] for c in cases}
    assert len(cases) == 81
    summary = summarise(evaluate(ScriptedPlanner(script), cases, MASTERS))
    assert summary["correct"] == "81/81"
    assert summary["misroute_in"]["rate"] == "0/81"
    assert summary["wrong_plan"] == [] and summary["misroute_out"] == []

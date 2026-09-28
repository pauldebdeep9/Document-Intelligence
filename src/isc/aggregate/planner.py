"""Question -> planner call -> validated plan. The one planning code path.

RecordAnswerer.plan() and the planner eval (eval/planner.py) both call
plan_question(), so the eval measures exactly what the answerer runs: the
same prompt file, the same structured-output parsing and repair, the same
validate_plan() guards and reason strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isc.aggregate.plan import QueryPlan, QueryPlanRaw, validate_plan
from isc.common.config import load_prompt
from isc.common.errors import OutputTruncated, SchemaRepairExhausted
from isc.llm.ports import ChatModel, LLMResult, Message
from isc.llm.structured import parse_structured

PROMPT = "aggregate/query_plan.v1.md"


@dataclass(frozen=True)
class PlanAttempt:
    raw: QueryPlanRaw | None      # what the model returned (None if the call failed)
    plan: QueryPlan | None        # after validate_plan()
    reason: str                   # "" when plan is not None
    result: LLMResult | None      # the final LLM call (usage, model)


def plan_question(chat: ChatModel, question: str, masters_dir: Path) -> PlanAttempt:
    prompt = load_prompt(PROMPT)
    try:
        raw, _conf, result = parse_structured(
            chat, [Message.system(prompt), Message.user(question)], QueryPlanRaw)
    except (SchemaRepairExhausted, OutputTruncated) as exc:
        return PlanAttempt(raw=None, plan=None, reason=f"planner failed: {exc}", result=None)
    plan, reason = validate_plan(raw, question, masters_dir)
    return PlanAttempt(raw=raw, plan=plan, reason=reason, result=result)

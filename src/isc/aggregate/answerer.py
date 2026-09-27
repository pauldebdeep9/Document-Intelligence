"""plan -> permitted records -> execute -> render -> the same grounding checks.

`try_answer()` returns None whenever this path should not handle the
question -- not an aggregate question, a plan that fails validation, or a
planner call that fails outright -- and AnswerOrchestrator then runs the
chunk path exactly as it did before this module existed. Once a valid plan
exists, though, this path owns the answer: an empty result abstains here
rather than falling through, because handing the same question to the chunk
path is how q_cd_01..04 produced a partial sum, an over-inclusive sum and a
fabricated total in the first place.
"""

from __future__ import annotations

from isc.aggregate.execute import AggregateResult, execute
from isc.aggregate.plan import QueryPlan, QueryPlanRaw, validate_plan
from isc.aggregate.render import render_text
from isc.aggregate.source import visible_records
from isc.answer.citations import bind_citations, verify_attribution
from isc.common.confidence import Confidence, Thresholds
from isc.common.config import Settings, load_prompt
from isc.common.errors import OutputTruncated, SchemaRepairExhausted
from isc.common.logging import get_logger
from isc.common.tracing import span
from isc.extract.masters import supplier_ids_by_name
from isc.llm.ports import ChatModel, Message
from isc.llm.structured import parse_structured
from isc.models.acl import Principal
from isc.models.answer import AbstentionReason, Answer
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("aggregate")

ROUTE = "records"


class RecordAnswerer:
    def __init__(self, chat: ChatModel, docs: SqliteDocStore, store: LocalVectorStore,
                 settings: Settings) -> None:
        self._chat = chat
        self._docs = docs
        self._store = store
        self._masters = settings.paths.data / "masters"
        self._thresholds = Thresholds(
            auto_accept=settings.thresholds.auto_accept,
            review=settings.thresholds.review,
            reject=settings.thresholds.reject,
        )
        self._contributing = frozenset(settings.aggregate.contributing_routes)
        self._supplier_ids = supplier_ids_by_name(self._masters)

    def plan(self, question: str) -> tuple[QueryPlan | None, str]:
        prompt = load_prompt("aggregate/query_plan.v1.md")
        try:
            raw, _conf, _result = parse_structured(
                self._chat, [Message.system(prompt), Message.user(question)], QueryPlanRaw)
        except (SchemaRepairExhausted, OutputTruncated) as exc:
            return None, f"planner failed: {exc}"
        return validate_plan(raw, question, self._masters)

    def try_answer(self, question: str, principal: Principal) -> Answer | None:
        with span("aggregate.plan") as s:
            plan, reason = self.plan(question)
            # Recorded on every question, so a misroute (a single_hop
            # question answered from records, or an aggregate one that fell
            # through) is visible in the trace without re-running anything.
            s.attributes["route"] = ROUTE if plan is not None else "chunks"
            s.attributes["reason"] = reason
            if plan is None:
                log.info("aggregate: falling through to chunk path (%s)", reason)
                return None
        with span("aggregate.answer", principal=principal.id,
                  operation=plan.operation.value, suppliers=",".join(plan.supplier_ids),
                  part=plan.part_number or "", currency=plan.currency or ""):
            view = visible_records(principal, self._docs, self._store)
            result = execute(plan, view, self._thresholds, self._contributing, self._masters)
            return self._answer(question, result)

    def _answer(self, question: str, result: AggregateResult) -> Answer:
        if result.empty:
            # Nothing this principal may read contributes. Deliberately the
            # same user-facing abstention as the chunk path's NO_RESULTS:
            # there is no way to tell "no such orders" from "orders you
            # cannot see" here, because the hidden ones were never loaded.
            ans = Answer.abstain(question, AbstentionReason.NO_RESULTS)
            if result.excluded:
                # Something visible matched but nothing could be summed
                # (e.g. every matching order prints no total). Keep the
                # evidence for eval, same as every other abstention path.
                _text, supporting = render_text(result, self._masters)
                ans = Answer.abstain(question, AbstentionReason.INSUFFICIENT_CONTEXT,
                                     supporting=supporting)
            return ans.model_copy(update={"route": ROUTE})

        text, supporting = render_text(result, self._masters)
        citations = bind_citations(text, supporting)
        if not citations:
            return Answer.abstain(question, AbstentionReason.UNGROUNDED_DRAFT,
                                  supporting=supporting).model_copy(update={"route": ROUTE})
        attribution = verify_attribution(text, supporting, self._supplier_ids)
        if attribution.mismatches:
            log.error("aggregate renderer produced unattributable text: %s",
                      attribution.mismatches)
            return Answer.abstain(question, AbstentionReason.ATTRIBUTION_MISMATCH,
                                  supporting=supporting).model_copy(update={"route": ROUTE})

        return Answer(
            question=question,
            text=text,
            citations=citations,
            supporting=supporting,
            # The weakest value the figure depends on -- not the planner's
            # confidence: the plan is checked by validate_plan(), the
            # numbers are what the reader acts on.
            confidence=Confidence.weakest_link(*[r.confidence for r in result.included]),
            route=ROUTE,
        )

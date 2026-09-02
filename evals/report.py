"""Turns a saved RunRecord into a plain-text report using evals.metrics. Touches no API.

Every count is rendered with metrics.format_kn ("k/n") — no aggregate headline number and no
percentage anywhere. Failure slices are the point: an overall pass rate would hide exactly the
per-class and per-field differences this corpus was built to expose. Sliced numbers stay
sliced; nothing here rolls them back up into one score.
"""

from collections import Counter, defaultdict

from evals.gold.schema import GoldSet, QuestionClass
from evals.metrics import (
    FieldVerdict,
    first_hit_rank,
    format_kn,
    hit_at_k,
    is_insufficiency_response,
    reciprocal_rank,
    score_line_items,
    score_purchase_order,
)
from evals.runner import ItemRecord, RunRecord
from isc.models import SourceEvidence

_QUESTION_CLASSES: tuple[QuestionClass, ...] = (
    "header_field",
    "line_item",
    "cross_page",
    "absent",
    "near_duplicate",
)


def _reconstruct_sources(item: ItemRecord) -> list[SourceEvidence]:
    return [
        SourceEvidence(doc_id=doc_id, chunk_id=chunk_id, page_number=page_number, text=text,
                        score=score)
        for doc_id, chunk_id, page_number, text, score in zip(
            item.retrieved_doc_ids,
            item.retrieved_chunk_ids,
            item.retrieved_page_numbers,
            item.retrieved_texts,
            item.retrieved_scores,
            strict=True,
        )
    ]


def build_report(run_record: RunRecord, goldset: GoldSet) -> str:
    """Build a plain-text report from a saved run record and the goldset it was run against.

    absent questions are scored by string compliance (is_insufficiency_response), not
    retrieval — they have no anchors, so including them in the retrieval hit@k/MRR slices
    would silently inflate or deflate those numbers depending on how an empty anchor list is
    treated. They are excluded from the retrieval section entirely and reported in their own
    "Insufficiency compliance" section instead; the excluded count is stated in the header.

    Items with a recorded error are excluded from every scoring slice (retrieval, extraction,
    compliance) and listed only in "Errors" — counting an infrastructure failure as a
    retrieval miss or a non-compliant answer would conflate two different failure modes in
    the same number.
    """
    retrieval_gold_by_id = {
        retrieval.question_id: retrieval
        for item in goldset.items
        for retrieval in item.retrieval
    }
    expected_purchase_order_by_doc = {
        item.doc_id: item.extraction.expected for item in goldset.items
    }

    ok_items = [item for item in run_record.items if item.error is None]
    errored_items = [item for item in run_record.items if item.error is not None]
    absent_items = [item for item in ok_items if item.question_class == "absent"]
    retrieval_items = [item for item in ok_items if item.question_class != "absent"]

    lines: list[str] = []
    lines.append(f"Run {run_record.run_id} ({run_record.config.split.value})")
    lines.append(
        f"{len(absent_items)} absent questions excluded from retrieval slices "
        "(scored separately by string compliance, not retrieval)."
    )
    lines.append("")

    # --- Retrieval, sliced by question_class -------------------------------------------------
    lines.append("=== Retrieval ===")
    hits_by_class: dict[QuestionClass, int] = defaultdict(int)
    total_by_class: dict[QuestionClass, int] = defaultdict(int)
    reciprocal_ranks_by_class: dict[QuestionClass, list[float]] = defaultdict(list)

    for item in retrieval_items:
        gold = retrieval_gold_by_id[item.question_id]
        sources = _reconstruct_sources(item)
        total_by_class[item.question_class] += 1
        if hit_at_k(sources, gold.doc_id, gold.anchors):
            hits_by_class[item.question_class] += 1
        rank = first_hit_rank(sources, gold.doc_id, gold.anchors)
        reciprocal_ranks_by_class[item.question_class].append(reciprocal_rank(rank))

    for question_class in _QUESTION_CLASSES:
        if question_class == "absent":
            continue
        total = total_by_class[question_class]
        if total == 0:
            lines.append(f"{question_class}: no questions in this run")
            continue
        hits = hits_by_class[question_class]
        ranks = reciprocal_ranks_by_class[question_class]
        mrr = sum(ranks) / len(ranks)
        lines.append(f"{question_class}: hit@k {format_kn(hits, total)}, MRR {mrr:.3f}")
    lines.append("")

    # --- Extraction, sliced by field ----------------------------------------------------------
    lines.append("=== Extraction ===")
    document_count = len(run_record.documents)
    verdict_counts_by_field: dict[str, Counter[FieldVerdict]] = defaultdict(Counter)
    for document in run_record.documents:
        expected = expected_purchase_order_by_doc[document.doc_id]
        verdicts = score_purchase_order(expected, document.purchase_order)
        for field, verdict in verdicts.items():
            verdict_counts_by_field[field][verdict] += 1

    for field, counts in verdict_counts_by_field.items():
        rendered = ", ".join(
            f"{verdict.value} {format_kn(counts[verdict], document_count)}"
            for verdict in FieldVerdict
        )
        lines.append(f"{field}: {rendered}")
    lines.append("")

    # --- Line items, per document (no rollup across documents) -------------------------------
    lines.append("=== Line items ===")
    for document in run_record.documents:
        expected_items = expected_purchase_order_by_doc[document.doc_id].line_items
        actual_items = document.purchase_order.line_items
        line_item_score = score_line_items(expected_items, actual_items)
        lines.append(
            f"{document.doc_id}: "
            f"matched {format_kn(line_item_score.matched, len(expected_items))}, "
            f"missing {format_kn(line_item_score.missing, len(expected_items))}, "
            f"spurious {format_kn(line_item_score.spurious, len(actual_items))}"
        )
    lines.append("")

    # --- Insufficiency compliance (absent slice) ----------------------------------------------
    lines.append("=== Insufficiency compliance (absent questions) ===")
    compliant_items: list[ItemRecord] = []
    non_compliant_items: list[ItemRecord] = []
    for item in absent_items:
        is_compliant = item.answer is not None and is_insufficiency_response(item.answer)
        (compliant_items if is_compliant else non_compliant_items).append(item)
    lines.append(f"compliant: {format_kn(len(compliant_items), len(absent_items))}")
    if non_compliant_items:
        lines.append("Non-compliant items:")
        for item in non_compliant_items:
            lines.append(f"  {item.question_id} (doc {item.doc_id}): answer = {item.answer!r}")
    lines.append("")

    # --- Errors --------------------------------------------------------------------------------
    lines.append("=== Errors ===")
    lines.append(f"{len(errored_items)} item(s) errored" + (":" if errored_items else "."))
    for item in errored_items:
        lines.append(f"  {item.question_id} (doc {item.doc_id}): {item.error}")

    return "\n".join(lines) + "\n"

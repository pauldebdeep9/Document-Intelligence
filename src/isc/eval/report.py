"""Render both harnesses into runs/<run_id>/eval/report.{json,md}.

Markdown alongside JSON on purpose: the JSON is for regression diffing between
runs, the Markdown is what gets pasted into a design review.

EV-02: every rate (a float in [0,1] computed as a ratio over some population)
is emitted as its own {"n": <denominator>, ...: <rate>} object, never a bare
float -- see tests/unit/test_eval_report_rates.py, which walks report.json
and fails on any exception to that. n_answerable sitting far from
recall@5/recall@8/mrr/ndcg@8 in a flat dict, or abstention_precision's and
abstention_recall's denominators not appearing anywhere at all, both read
fine until someone reads recall@8=0.97 without noticing it is 34/35 rather
than 340/350 -- at this corpus's scale (35 answerable questions, subtypes as
small as 2) that difference is the whole story. Known, deliberately deferred
exception: extraction.by_field's PRF shape (precision/recall/f1) -- see the
allowlist in test_eval_report_rates.py for why.

report.json also carries a top-level schema_version (REPORT_SCHEMA_VERSION)
from this item on -- there is no runtime loader for report.json anywhere in
this codebase today (it is a terminal artifact for humans and for
`git diff`-style regression comparison between runs), so
validate_schema_version() exists for the day something does read it back,
and for this module's own tests to pin the shape against drift.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from isc.common.errors import IscError
from isc.eval.extraction import ExtractionReport
from isc.eval.retrieval import RetrievalReport

REPORT_SCHEMA_VERSION = 2


class ReportSchemaMismatch(IscError):
    """report.json's schema_version does not match this reader's
    REPORT_SCHEMA_VERSION. No code in this repo reads report.json back
    today, so nothing raises this yet in production -- it exists so a
    future reader (and this module's own tests) can reject a mismatched
    shape loudly rather than silently misreading it, the same discipline as
    eval/outcomes.py's OutcomesSchemaMismatch."""

    def __init__(self, found: object) -> None:
        super().__init__(
            f"report.json schema_version {found!r} does not match this reader's "
            f"REPORT_SCHEMA_VERSION={REPORT_SCHEMA_VERSION} -- this report was written by a "
            "different version of report.write(); regenerate it, or read it with a matching "
            "version of this module."
        )
        self.found = found


def validate_schema_version(payload: dict) -> None:
    found = payload.get("schema_version")
    if found != REPORT_SCHEMA_VERSION:
        raise ReportSchemaMismatch(found)


def write(out_dir: Path, extraction: ExtractionReport | None,
          retrieval: RetrievalReport | None, threshold: float = 0.9,
          review_threshold: float = 0.6) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"schema_version": REPORT_SCHEMA_VERSION}
    lines = ["# Evaluation report", ""]

    if extraction:
        # PRF is a slots=True dataclass -- it has no __dict__, so vars()
        # raises; asdict() works for any dataclass regardless of slots.
        by_field = {k: asdict(v) for k, v in extraction.by_field().items()}
        false_negatives_by_axis = {
            axis: [asdict(o) for o in extraction.false_negatives(threshold, axis=axis)]
            for axis in ("extraction", "normalisation")
        }
        line_outcomes = [asdict(o) for o in extraction.line_outcomes()]
        detected, total_errors = extraction.detection_rate(threshold)
        review_wrong, review_total = extraction.band_precision(review_threshold, threshold)
        auto_wrong, auto_total = extraction.auto_accept_band(threshold)
        payload["extraction"] = {
            "by_field": by_field,
            "calibration": extraction.calibration(),
            "auto_accept_error_rate": {
                "n": auto_total,
                "wrong": auto_wrong,
                "auto_accept_error_rate": extraction.auto_accept_error_rate(threshold),
            },
            "detection": {"detected": detected, "total_errors": total_errors},
            "review_band": {"wrong": review_wrong, "total": review_total},
            "false_negatives": false_negatives_by_axis,
            "line_outcomes": line_outcomes,
        }
        lines += ["## Extraction", "", "| field | P | R | F1 | n |", "|---|---|---|---|---|"]
        lines += [
            f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
            for k, v in by_field.items()
        ]
        lines += ["", f"Auto-accept error rate at {threshold}: "
                      f"{extraction.auto_accept_error_rate(threshold):.3%}", ""]

        # Detection rate: the two counts, not a percentage alone. A percent
        # by itself cannot say whether it moved because more errors got
        # caught (the numerator) or because fewer errors exist at all (the
        # denominator) -- state both every time, so "detection improved"
        # never gets misread as "detection got better" when instead an
        # error class was eliminated outright and the denominator shrank.
        not_detected = total_errors - detected
        lines += ["### Detection rate", ""]
        if total_errors:
            lines += [
                f"{detected} of {total_errors} real errors (normalisation axis) scored "
                f"low enough to avoid auto-accept at threshold {threshold}. "
                f"{not_detected} auto-accepted despite disagreeing with gold "
                f"-- see false negatives below.",
                "",
            ]
        else:
            lines += ["No errors on this axis.", ""]

        # Review-band precision: of everything routed to review, what
        # fraction is actually wrong. This is the cost side of the same
        # threshold: every item in this band is a human review, whether or
        # not it turns out to be an error.
        lines += ["### Review-band precision", ""]
        if review_total:
            lines += [
                f"{review_wrong} of {review_total} outcomes routed to review "
                f"({review_threshold}-{threshold}) are actually wrong "
                f"({review_wrong / review_total:.1%}).",
                "",
            ]
        else:
            lines += ["Nothing routed to review at this threshold.", ""]

        # False negatives: the headline number. Everything that disagrees
        # with gold (field or whole line) but scored high enough to
        # auto-accept -- see ExtractionReport.false_negatives()'s docstring.
        # Split by axis: an extraction-axis false negative is the model
        # misreading the document; a normalisation-axis one is our own code
        # turning a correct read into a wrong typed value -- conflating them
        # misattributes the fix.
        lines += [f"### False negatives (auto-accept threshold {threshold})", ""]
        for axis, false_negatives in false_negatives_by_axis.items():
            lines += [f"**{axis} axis**", ""]
            if false_negatives:
                lines += ["| document | field | outcome | extracted | gold | confidence |",
                          "|---|---|---|---|---|---|"]
                lines += [
                    f"| {o['document_id']} | {o['field_name']} | {o['outcome']} | "
                    f"{o['predicted']} | {o['gold']} | {o['confidence']:.3f} |"
                    for o in false_negatives
                ]
            else:
                lines += ["None."]
            lines += [""]

        # Line outcomes: record-level events (dropped/hallucinated/mismatch
        # lines) reported on their own, never folded into the field table --
        # see ExtractionReport.by_field()'s docstring for why.
        lines += ["### Line outcomes", ""]
        if line_outcomes:
            counts: dict[str, int] = {}
            for o in line_outcomes:
                counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
            lines += [f"- {k}: {v}" for k, v in sorted(counts.items())]
            lines += ["", "| document | line | outcome | confidence |", "|---|---|---|---|"]
            lines += [
                f"| {o['document_id']} | {o['field_name']} | {o['outcome']} | "
                f"{o['confidence']:.3f} |"
                for o in line_outcomes
            ]
        else:
            lines += ["None."]
        lines += [""]

    if retrieval:
        recall_by_subtype = retrieval.recall_by_subtype()
        answer_accuracy = retrieval.answer_accuracy()
        answerable_failures = retrieval.answerable_failures()
        abstention_by_subtype = retrieval.abstention_by_subtype()
        restricted_summary = retrieval.restricted_summary()
        no_reader_summary = retrieval.no_reader_summary()
        leaks_by_subtype = retrieval.leaks_by_subtype()
        n_answerable = sum(v["n"] for v in recall_by_subtype.values())
        expected_to_abstain, n_abstained = retrieval.abstention_precision_band()
        abstention_correct, n_unanswerable = retrieval.abstention_recall_band()

        # EV-02: every rate below is its own {"n": <denominator>, ...} object,
        # never a bare float -- see this module's docstring and
        # test_eval_report_rates.py. recall@5/8, mrr and ndcg@8 share
        # n_answerable as their denominator (all four average over the same
        # answerable-class outcomes, retrieval.py's recall_at()/mean_mrr()/
        # ndcg()); abstention_precision and abstention_recall do NOT share
        # that denominator (or each other's) -- see abstention_precision_band()/
        # abstention_recall_band()'s docstrings for the populations each one
        # actually counts over.
        restricted_payload = {}
        for subtype, row in restricted_summary.items():
            restricted_payload[subtype] = dict(row)
            restricted_payload[subtype]["primary_recall@8"] = {
                "n": row["n_pairs"], "primary_recall@8": row["primary_recall@8"],
            }

        payload["retrieval"] = {
            "recall@5": {"n": n_answerable, "recall@5": retrieval.recall_at(5)},
            "recall@8": {"n": n_answerable, "recall@8": retrieval.recall_at(8)},
            "mrr": {"n": n_answerable, "mrr": retrieval.mean_mrr()},
            "ndcg@8": {"n": n_answerable, "ndcg@8": retrieval.ndcg(8)},
            "n_answerable": n_answerable,
            "recall_by_subtype": recall_by_subtype,
            "answer_accuracy": answer_accuracy,
            "answerable_failures": answerable_failures,
            "abstention_precision": {
                "n": n_abstained,
                "expected_to_abstain": expected_to_abstain,
                "abstention_precision": retrieval.abstention_precision(),
            },
            "abstention_recall": {
                "n": n_unanswerable,
                "correct": abstention_correct,
                "abstention_recall": retrieval.abstention_recall(),
            },
            "abstention_by_subtype": abstention_by_subtype,
            "restricted": restricted_payload,
            "no_reader": no_reader_summary,
            "acl_leaks": len(retrieval.leaks()),
            "leaks_by_subtype": leaks_by_subtype,
            "passed": retrieval.passed(),
        }

        lines += ["## Retrieval", ""]
        lines += [
            f"n={n_answerable} answerable questions over 20 documents. At this sample size a "
            "single flipped outcome moves any per-subtype figure by several percentage points "
            "-- these numbers are indicative of where the system is weak, not a tight estimate "
            "of by how much. The extraction harness measures a corpus with almost no errors "
            "left (docs/adr/0007); retrieval is not that -- treat these as a first honest "
            "reading, not a converged benchmark.",
            "",
        ]

        lines += [
            f"- recall@5: {retrieval.recall_at(5):.3f}",
            f"- recall@8: {retrieval.recall_at(8):.3f}",
            f"- mrr: {retrieval.mean_mrr():.3f}",
            f"- ndcg@8: {retrieval.ndcg(8):.3f}",
            f"- answer accuracy: {answer_accuracy['correct']}/{answer_accuracy['n']} "
            f"({answer_accuracy['accuracy']:.1%})",
            f"- abstention precision: {retrieval.abstention_precision():.3f}",
            f"- abstention recall (reason-aware): {retrieval.abstention_recall():.3f}",
            f"- acl_leaks: {len(retrieval.leaks())}",
            f"- passed: {retrieval.passed()}",
            "",
        ]
        if not retrieval.passed():
            lines += ["**RUN FAILED: ACL leak detected.**", ""]

        # Per-subtype recall: cross_document and line_item called out
        # first and explicitly -- the two subtypes P1-08's gold set was
        # built to stress (cross_document needs chunks from 2+ documents
        # surviving one RRF fusion; line_item needs the right slice of a
        # split table), so a reader should not have to hunt for them in an
        # alphabetical table.
        lines += ["### Recall by subtype (answerable)", ""]
        for headline in ("cross_document", "line_item"):
            if headline in recall_by_subtype:
                v = recall_by_subtype[headline]
                lines += [
                    f"**{headline}** (n={v['n']}): recall@5={v['recall@5']:.3f}, "
                    f"recall@8={v['recall@8']:.3f}, mrr={v['mrr']:.3f}, ndcg@8={v['ndcg@8']:.3f}",
                    "",
                ]
        lines += ["| subtype | n | recall@5 | recall@8 | mrr | ndcg@8 |", "|---|---|---|---|---|---|"]
        lines += [
            f"| {subtype} | {v['n']} | {v['recall@5']:.3f} | {v['recall@8']:.3f} | "
            f"{v['mrr']:.3f} | {v['ndcg@8']:.3f} |"
            for subtype, v in sorted(recall_by_subtype.items())
        ]
        lines += [""]

        # Answer accuracy failures: every one tagged with which stage the
        # failure implicates, so "answering is weak" is never reported
        # when the real finding is retrieval never surfaced the evidence.
        lines += ["### Answerable questions that failed answer accuracy", ""]
        if answerable_failures:
            lines += ["| question | subtype | gold chunks retrieved | diagnosis | abstained | reason |",
                      "|---|---|---|---|---|---|"]
            lines += [
                f"| {f['question_id']} | {f['subtype']} | {f['gold_chunks_retrieved']} | "
                f"{f['diagnosis']} | {f['abstained']} | {f['abstention_reason'] or ''} |"
                for f in answerable_failures
            ]
        else:
            lines += ["None."]
        lines += [""]

        # Abstention by subtype: absent/out_of_scope (plain "abstain") vs
        # underspecified ("abstain_with_clarification", which nothing in
        # this system can currently produce) -- reported separately so the
        # one known gap does not drag down the two subtypes that work.
        lines += ["### Abstention by subtype (unanswerable)", ""]
        lines += ["| subtype | n | correct | accuracy |", "|---|---|---|---|"]
        lines += [
            f"| {subtype} | {v['n']} | {v['correct']} | {v['accuracy']:.1%} |"
            for subtype, v in sorted(abstention_by_subtype.items())
        ]
        lines += [""]

        # Restricted: filtered vs unfiltered, never folded into one number.
        lines += ["### Restricted (filtered vs unfiltered)", ""]
        for subtype, v in restricted_summary.items():
            lines += [
                f"**{subtype}**: {v['n_pairs']} pairs. Gold-principal side: "
                f"recall@8={v['primary_recall@8']:.3f}, "
                f"answer correct {v['primary_answer_correct']}/{v['n_pairs']}. "
                f"Non-gold-principal side: {v['secondary_correctly_empty_and_abstained']}/"
                f"{v['secondary_n']} correctly empty-and-abstained, {v['leaks']} leak(s).",
                "",
            ]
        lines += [
            f"no_reader: {no_reader_summary['n_principals_checked']} principals checked, "
            f"all empty: {no_reader_summary['all_empty']}"
            + (f", principals with results: {no_reader_summary['principals_with_results']}"
               if no_reader_summary["principals_with_results"] else ""),
            "",
        ]

        if leaks_by_subtype:
            lines += ["### Leaks by subtype", ""]
            lines += [f"- {subtype}: {n}" for subtype, n in sorted(leaks_by_subtype.items())]
            lines += [""]

    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    md = out_dir / "report.md"
    md.write_text("\n".join(lines))
    return md

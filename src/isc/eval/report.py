"""Render both harnesses into runs/<run_id>/eval/report.{json,md}.

Markdown alongside JSON on purpose: the JSON is for regression diffing between
runs, the Markdown is what gets pasted into a design review.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from isc.eval.extraction import ExtractionReport
from isc.eval.retrieval import RetrievalReport


def write(out_dir: Path, extraction: ExtractionReport | None,
          retrieval: RetrievalReport | None, threshold: float = 0.9,
          review_threshold: float = 0.6) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
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
        payload["extraction"] = {
            "by_field": by_field,
            "calibration": extraction.calibration(),
            "auto_accept_error_rate": extraction.auto_accept_error_rate(threshold),
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
        payload["retrieval"] = {
            "recall@5": retrieval.recall_at(5),
            "recall@8": retrieval.recall_at(8),
            "mrr": retrieval.mean_mrr(),
            "ndcg@8": retrieval.ndcg(8),
            "abstention_precision": retrieval.abstention_precision(),
            "abstention_recall": retrieval.abstention_recall(),
            "acl_leaks": len(retrieval.leaks()),
            "passed": retrieval.passed(),
        }
        r = payload["retrieval"]
        lines += ["## Retrieval", ""]
        lines += [f"- {k}: {v}" for k, v in r.items()]
        if not retrieval.passed():
            lines += ["", "**RUN FAILED: ACL leak detected.**"]

    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    md = out_dir / "report.md"
    md.write_text("\n".join(lines))
    return md

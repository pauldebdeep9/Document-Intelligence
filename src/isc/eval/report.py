"""Render both harnesses into runs/<run_id>/eval/report.{json,md}.

Markdown alongside JSON on purpose: the JSON is for regression diffing between
runs, the Markdown is what gets pasted into a design review.
"""

from __future__ import annotations

import json
from pathlib import Path

from isc.eval.extraction import ExtractionReport
from isc.eval.retrieval import RetrievalReport


def write(out_dir: Path, extraction: ExtractionReport | None,
          retrieval: RetrievalReport | None, threshold: float = 0.9) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    lines = ["# Evaluation report", ""]

    if extraction:
        by_field = {k: vars(v) for k, v in extraction.by_field().items()}
        payload["extraction"] = {
            "by_field": by_field,
            "calibration": extraction.calibration(),
            "auto_accept_error_rate": extraction.auto_accept_error_rate(threshold),
        }
        lines += ["## Extraction", "", "| field | P | R | F1 | n |", "|---|---|---|---|---|"]
        lines += [
            f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
            for k, v in by_field.items()
        ]
        lines += ["", f"Auto-accept error rate at {threshold}: "
                      f"{extraction.auto_accept_error_rate(threshold):.3%}", ""]

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

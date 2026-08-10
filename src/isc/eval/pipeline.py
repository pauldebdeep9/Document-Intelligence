"""Score a completed extract run against gold.

Reads runs/<run_id>/extract/*.json (written by extract/pipeline.py) and
data/gold/extraction/*.json side by side -- no LLM calls. extract/pipeline.py
persists the model's raw structured output next to the wrapped record
specifically so this stage never needs a second inference pass: an eval that
measures a different run than the one it reports on is not an eval.

Runs per document, isolated -- same reasoning as extract/pipeline.py: one
malformed artifact or missing gold file must not abort scoring the other
nineteen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from isc.common.logging import get_logger
from isc.common.tracing import Run
from isc.eval.extraction import ExtractionReport, compare, compare_lines, load_artifact

# load_artifact()'s registry.get(doc_type) only finds a record class once
# that class's module has been imported -- @register is a decorator, its
# side effect runs at import time. extract/extractor.py imports these
# directly for the same reason; this stage runs standalone (a separate `isc
# eval` process, not one that already ran `isc extract`), so it must import
# them too, or every artifact is unscoreable with "no record class
# registered" regardless of how correct load_artifact() itself is.
from isc.models.records import invoice, purchase_order  # noqa: F401
from isc.storage.sqlite_docstore import SqliteDocStore

log = get_logger("eval.extraction")


@dataclass
class ExtractionEvalResult:
    report: ExtractionReport
    scored: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (artifact, reason)


def run(run_obj: Run, docs: SqliteDocStore, gold_dir: Path) -> ExtractionEvalResult:
    extract_dir = run_obj.dir / "extract"
    result = ExtractionEvalResult(report=ExtractionReport())

    for path in sorted(extract_dir.glob("*.json")):
        try:
            record, raw_predicted = load_artifact(path)
            doc = docs.get_document(record.document_id)
            if doc is None:
                raise RuntimeError(f"{record.document_id} not in docstore")
            gold_path = gold_dir / f"{Path(doc.source_uri).stem}.json"
            if not gold_path.exists():
                raise RuntimeError(f"no gold at {gold_path}")
            gold = json.loads(gold_path.read_text())
            result.report.outcomes += compare(record.document_id, gold, raw_predicted, record)
            result.report.outcomes += compare_lines(record.document_id, gold, raw_predicted, record)
        except Exception as exc:  # noqa: BLE001 - batch isolation boundary, by design
            log.warning("eval skipped %s: %s", path.name, exc)
            result.skipped.append((path.name, str(exc)))
            continue
        result.scored.append(record.document_id)

    log.info("scored %d documents against gold, %d skipped",
              len(result.scored), len(result.skipped))
    return result

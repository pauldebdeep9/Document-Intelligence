"""Unit test for eval/report.py's write().

Regression for a real bug: `vars(v)` on a PRF raises TypeError, because PRF
is a slots=True dataclass (no __dict__) -- never caught by any test because
`isc eval` was a stub (`raise typer.Exit`) until it was wired up for real
here. Nothing exercised write() end to end before that.
"""

from __future__ import annotations

import json

from isc.eval.extraction import ExtractionReport, FieldOutcome
from isc.eval.report import write


def test_write_handles_prf_and_line_outcomes_without_raising(tmp_path):
    outcomes = [
        FieldOutcome("doc_1", "po_number", "correct", "normalisation", "123", "123", 0.95),
        FieldOutcome("doc_1", "supplier_name", "wrong", "normalisation", "Acme", "Acmee", 0.95),
        FieldOutcome("doc_1", "lines[20]", "dropped_line", "normalisation",
                     None, {"line_number": 20}, 0.92),
    ]
    report = ExtractionReport(outcomes)

    md_path = write(tmp_path, report, None, threshold=0.90)

    assert md_path.exists()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["extraction"]["by_field"]["po_number"]["support"] == 1
    assert payload["extraction"]["line_outcomes"][0]["outcome"] == "dropped_line"
    text = md_path.read_text()
    assert "po_number" in text
    assert "dropped_line" in text

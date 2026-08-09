"""Extraction harness: field-level precision/recall against gold.

Four outcomes per field, and collapsing them is how extraction evals lie:

  correct        value matches gold
  wrong          value present, differs from gold          <- the dangerous one
  missed         gold has a value, extraction returned null
  correct_absent gold is null and extraction returned null <- must not count as
                 a win in the same bucket as `correct`, or a model that returns
                 null for everything scores well on sparse documents

Reported per field name and per document type, never as one aggregate number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isc.eval.metrics import PRF, calibration_bins


@dataclass
class FieldOutcome:
    document_id: str
    field_name: str
    outcome: str
    predicted: Any = None
    gold: Any = None
    confidence: float = 0.0


@dataclass
class ExtractionReport:
    outcomes: list[FieldOutcome] = field(default_factory=list)

    def by_field(self) -> dict[str, PRF]:
        names = {o.field_name for o in self.outcomes}
        out = {}
        for name in sorted(names):
            rows = [o for o in self.outcomes if o.field_name == name]
            tp = sum(o.outcome == "correct" for o in rows)
            fp = sum(o.outcome == "wrong" for o in rows)
            fn = sum(o.outcome == "missed" for o in rows)
            out[name] = PRF.from_counts(tp, fp, fn)
        return out

    def calibration(self) -> list[dict[str, float]]:
        return calibration_bins(
            [(o.confidence, o.outcome in {"correct", "correct_absent"}) for o in self.outcomes]
        )

    def auto_accept_error_rate(self, threshold: float) -> float:
        """The number a process owner actually asks for: of everything we would
        push through without review, what fraction is wrong?"""
        auto = [o for o in self.outcomes if o.confidence >= threshold]
        if not auto:
            return 0.0
        return sum(o.outcome == "wrong" for o in auto) / len(auto)


def compare(predicted: dict[str, Any], gold: dict[str, Any],
            confidences: dict[str, float], document_id: str) -> list[FieldOutcome]:
    raise NotImplementedError(
        "first slice: exact match with type-aware normalisation "
        "(dates to ISO, decimals to 2dp, strings casefold+strip)"
    )

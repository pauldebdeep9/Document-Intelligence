"""Metric primitives, kept separate from the harnesses so they are unit-testable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PRF:
    precision: float
    recall: float
    f1: float
    support: int

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> PRF:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return cls(p, r, f, tp + fn)


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    import math
    dcg = sum(
        1 / math.log2(i + 1)
        for i, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in gold_ids
    )
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal else 0.0


def calibration_bins(pairs: list[tuple[float, bool]], bins: int = 10) -> list[dict[str, float]]:
    """Is a 0.9-confidence field right 90% of the time?

    Without this, confidence is decoration. It is the single number that decides
    whether the auto-accept threshold can be defended to a process owner.
    """
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        window = [c for conf, c in pairs if lo <= conf < hi or (b == bins - 1 and conf == 1.0)]
        if not window:
            continue
        out.append({
            "bin_low": lo, "bin_high": hi, "n": len(window),
            "accuracy": sum(window) / len(window),
            "gap": sum(window) / len(window) - (lo + hi) / 2,
        })
    return out

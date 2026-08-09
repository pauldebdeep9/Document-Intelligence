"""Confidence as a first-class propagated signal.

The design commitment: confidence is never a bare float on a dict. It carries the
signals that produced it, so that when a field is rejected at HITL review you can
tell whether the OCR was bad, the layout model mis-grouped a table, or the model
guessed. Losing that decomposition is what makes extraction pipelines unfixable.

Combination rules, and when each is right:

  independent()  product of factors. Use when signals are genuinely separate
                 evidence about the same claim (OCR legibility x model certainty).
                 Pessimistic by construction; that is intended.

  weakest_link() minimum. Use along a *chain* where any one step failing invalidates
                 the result (parse -> locate -> read). A perfect model reading a
                 garbled span is not a confident field.

  corroborate()  noisy-OR. Use when several signals independently *support* the same
                 value (regex match AND model output AND master-data lookup agree).
                 The only combinator that can raise confidence.

  weighted()     calibrated blend. Use only where you have gold data to fit weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

_EPS = 1e-9


class Signal(StrEnum):
    """Where a confidence factor came from. Keep this closed; add deliberately."""

    OCR = "ocr"                    # character-level recogniser score
    LAYOUT = "layout"              # block/table detection score
    MODEL = "model"                # LLM logprob or self-reported certainty
    SCHEMA = "schema"              # passed validation without repair
    LEXICAL = "lexical"            # regex / format check (e.g. PO number pattern)
    MASTER_DATA = "master_data"    # value resolves against supplier/part master
    RETRIEVAL = "retrieval"        # similarity or rerank score
    AGREEMENT = "agreement"        # multiple extractors returned the same value
    HUMAN = "human"                # reviewed in the HITL queue
    PROVENANCE = "provenance"      # value's source text could (not) be located in the document


@dataclass(frozen=True, slots=True)
class Factor:
    signal: Signal
    value: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"factor {self.signal} out of range: {self.value}")


@dataclass(frozen=True, slots=True)
class Confidence:
    """A score plus the factors that produced it."""

    score: float
    factors: tuple[Factor, ...] = field(default=())

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"confidence out of range: {self.score}")

    # -- constructors ------------------------------------------------------
    @classmethod
    def certain(cls) -> Self:
        """Only for values that cannot be wrong: constants, computed totals."""
        return cls(1.0)

    @classmethod
    def unknown(cls) -> Self:
        return cls(0.0)

    @classmethod
    def of(cls, signal: Signal, value: float, detail: str = "") -> Self:
        return cls(value, (Factor(signal, value, detail),))

    @classmethod
    def human_verified(cls, reviewer: str) -> Self:
        return cls(1.0, (Factor(Signal.HUMAN, 1.0, reviewer),))

    # -- combinators -------------------------------------------------------
    @classmethod
    def independent(cls, *parts: Confidence) -> Self:
        if not parts:
            return cls.unknown()
        score = math.prod(p.score for p in parts)
        return cls(score, _merge(parts))

    @classmethod
    def weakest_link(cls, *parts: Confidence) -> Self:
        if not parts:
            return cls.unknown()
        return cls(min(p.score for p in parts), _merge(parts))

    @classmethod
    def corroborate(cls, *parts: Confidence) -> Self:
        """Noisy-OR: 1 - prod(1 - p). Independent supporting evidence."""
        if not parts:
            return cls.unknown()
        score = 1.0 - math.prod(1.0 - p.score for p in parts)
        return cls(min(score, 1.0), _merge(parts))

    @classmethod
    def weighted(cls, pairs: list[tuple[Confidence, float]]) -> Self:
        total = sum(w for _, w in pairs)
        if total <= _EPS:
            return cls.unknown()
        score = sum(c.score * w for c, w in pairs) / total
        return cls(score, _merge([c for c, _ in pairs]))

    def penalise(self, signal: Signal, factor: float, reason: str) -> Confidence:
        """Apply a known-defect discount, e.g. a parser fallback was used.
        `signal` is the source of the defect, not assumed -- a MASTER_DATA
        conflict and a parser fallback are not the same signal, and tagging
        both as LAYOUT would mislabel the weakest factor shown to a reviewer."""
        return Confidence(
            self.score * factor,
            (*self.factors, Factor(signal, factor, reason)),
        )

    # -- interpretation ----------------------------------------------------
    def explain(self) -> str:
        if not self.factors:
            return f"{self.score:.3f} (no factors recorded)"
        parts = ", ".join(
            f"{f.signal}={f.value:.2f}" + (f" [{f.detail}]" if f.detail else "")
            for f in self.factors
        )
        return f"{self.score:.3f} <- {parts}"

    def weakest(self) -> Factor | None:
        """The factor to show a human reviewer first."""
        return min(self.factors, key=lambda f: f.value) if self.factors else None


def _merge(parts: tuple[Confidence, ...] | list[Confidence]) -> tuple[Factor, ...]:
    return tuple(f for p in parts for f in p.factors)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Routing policy. Lives in config, not scattered as literals in the code."""

    auto_accept: float = 0.90    # straight through
    review: float = 0.60         # below auto_accept -> HITL queue
    reject: float = 0.30         # below this -> do not surface the value at all

    def route(self, c: Confidence) -> str:
        if c.score >= self.auto_accept:
            return "accept"
        if c.score >= self.review:
            return "review"
        if c.score >= self.reject:
            return "low_confidence"
        return "reject"

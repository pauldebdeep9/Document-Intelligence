"""Extracted records: every field carries value + confidence + provenance.

The generic `ExtractedField[T]` is the reason confidence had to exist before the
first extractor was written. Retrofitting it means touching every schema, every
prompt, every gold comparison and every HITL screen.

Two-model pattern per document type:

  * `<Type>Raw`   — what the LLM is asked for. Flat, plain types, nullable.
                    Simple schemas raise strict-mode compliance and reduce repair
                    rounds; nested confidence objects invite the model to invent
                    its own certainty, which is not a signal worth having.
  * `<Type>`      — what the pipeline stores. Same fields wrapped in
                    ExtractedField, with confidence computed by *us* from OCR,
                    layout, logprobs, lexical checks and master-data lookups.

extract/ maps Raw -> wrapped. The model never sets its own confidence.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from isc.common.confidence import Confidence, Signal, Thresholds
from isc.models.document import DocType, Span

T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    """One field with its evidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: T | None = None
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    span: Span | None = None
    # Set when a lexical/master-data check disagreed with the model.
    conflicts: list[str] = Field(default_factory=list)
    # Set only when the field was affirmatively determined to be absent.
    absent_reason: str | None = None

    @classmethod
    def missing(cls, reason: str = "not present in document") -> "ExtractedField[T]":
        """Absent is a real answer, distinct from low confidence. The eval harness
        scores 'correctly identified as absent' separately from a wrong value."""
        return cls(value=None, confidence=Confidence.certain(), absent_reason=reason)

    @property
    def present(self) -> bool:
        return self.value is not None

    def route(self, thresholds: Thresholds) -> str:
        return thresholds.route(self.confidence)

    def corroborate_with(self, signal: Signal, score: float, detail: str = "") -> None:
        """Fold in an independent check (regex match, supplier master hit)."""
        object.__setattr__(
            self, "confidence",
            Confidence.corroborate(self.confidence, Confidence.of(signal, score, detail)),
        )

    def flag_conflict(self, message: str, penalty: float = 0.5) -> None:
        self.conflicts.append(message)
        object.__setattr__(self, "confidence", self.confidence.penalise(penalty, message))


class ExtractionRecord(BaseModel):
    """Base for every document-type record. Subclasses declare doc_type and fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_type: ClassVar[DocType]
    raw_model: ClassVar[type[BaseModel]]

    document_id: str
    extracted_at: date | None = None
    record_confidence: Confidence = Field(default_factory=Confidence.unknown)

    def fields(self) -> dict[str, ExtractedField[Any]]:
        return {
            name: value
            for name, value in self
            if isinstance(value, ExtractedField)
        }

    def rollup(self) -> Confidence:
        """Record-level confidence is the weakest required field, not the mean.

        Averaging hides the one wrong field that makes the whole record unusable,
        which is exactly the failure the HITL queue exists to catch.
        """
        required = [f for name, f in self.fields().items() if name in self.required_fields()]
        if not required:
            return Confidence.unknown()
        return Confidence.weakest_link(*[f.confidence for f in required])

    @classmethod
    def required_fields(cls) -> set[str]:
        """Fields whose absence makes the record not worth storing. Override."""
        return set()

    def needs_review(self, thresholds: Thresholds) -> list[str]:
        return [
            name for name, f in self.fields().items()
            if thresholds.route(f.confidence) in {"review", "low_confidence"}
        ]


# --- registry -------------------------------------------------------------

class _Registry:
    """DocType -> record class. Lets extract/ and eval/ dispatch without imports
    scattered through the pipeline, and lets scripts/export_schemas.py enumerate."""

    def __init__(self) -> None:
        self._by_type: dict[DocType, type[ExtractionRecord]] = {}

    def register(self, cls: type[ExtractionRecord]) -> type[ExtractionRecord]:
        self._by_type[cls.doc_type] = cls
        return cls

    def get(self, doc_type: DocType) -> type[ExtractionRecord]:
        if doc_type not in self._by_type:
            raise KeyError(f"no record class registered for {doc_type}")
        return self._by_type[doc_type]

    def all(self) -> dict[DocType, type[ExtractionRecord]]:
        return dict(self._by_type)


registry = _Registry()


def register(cls: type[ExtractionRecord]) -> type[ExtractionRecord]:
    return registry.register(cls)

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

    def discount(self, signal: Signal, score: float, detail: str = "") -> None:
        """Fold in a check that could not confirm the value either way --
        genuine uncertainty (e.g. an ambiguous date format resolved only
        partway), not a conflict. independent() so it can only lower
        confidence, never raise it the way corroborate_with() would -- the
        bug this exists to prevent is an expression of doubt getting read as
        support. Not appended to `conflicts`: nothing actually disagreed,
        the check just could not fully confirm, and a reviewer reading
        `conflicts` should see checks that failed, not ones that merely
        couldn't."""
        object.__setattr__(
            self, "confidence",
            Confidence.independent(self.confidence, Confidence.of(signal, score, detail)),
        )

    def flag_conflict(self, signal: Signal, message: str, penalty: float = 0.5) -> None:
        """Fold in a check that disagreed with the value (failed regex,
        master-data miss, arithmetic disagreement). `signal` must name the
        check that actually failed -- see Confidence.penalise()."""
        self.conflicts.append(message)
        object.__setattr__(self, "confidence", self.confidence.penalise(signal, penalty, message))


class ExtractionRecord(BaseModel):
    """Base for every document-type record. Subclasses declare doc_type and fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_type: ClassVar[DocType]
    raw_model: ClassVar[type[BaseModel]]

    document_id: str
    extracted_at: date | None = None
    record_confidence: Confidence = Field(default_factory=Confidence.unknown)
    # Line-item spans are located positionally (row text scoped from a
    # structural row-extraction pass), not by content match. True unless that
    # structural pass failed to verifiably line up with the record -- see
    # extract/extractor.py's row-alignment self-check. False means every
    # line-item field on this record has span=None regardless of whether an
    # unscoped search would otherwise have found something: a silently
    # misaligned span points a reviewer at the wrong row while looking
    # healthy, which is worse than no span at all.
    row_alignment_ok: bool = True
    row_alignment_detail: str = ""

    def fields(self) -> dict[str, ExtractedField[Any]]:
        """Top-level ExtractedField attributes only. Does not see into a
        `lines: list[POLine]`-shaped attribute -- use all_fields() for that."""
        return {
            name: value
            for name, value in self
            if isinstance(value, ExtractedField)
        }

    def all_fields(self) -> dict[str, ExtractedField[Any]]:
        """Every ExtractedField on this record, including line items --
        keyed lines[i].name for anything nested inside a list of sub-records.
        fields() alone only sees top-level attributes: a record with a
        `lines: list[POLine]` attribute has no other way for a per-line
        problem to ever reach rollup() or a reviewer."""
        out = dict(self.fields())
        for name, value in self:
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, BaseModel):
                        out.update({
                            f"{name}[{i}].{sub_name}": sub_value
                            for sub_name, sub_value in item
                            if isinstance(sub_value, ExtractedField)
                        })
        return out

    def rollup(self) -> Confidence:
        """Record-level confidence is the weakest *populated* field across
        the whole record -- header and line items alike -- not the mean, and
        not scoped to required_fields(). A record can have a sound
        po_number/supplier_name/po_date and still contain a materially wrong
        total_amount or a wrong line price; this is what is supposed to
        catch that. See required_confidence() for the narrower "can we
        proceed at all" question that required_fields() alone used to answer
        here, which is worth asking separately, not instead.

        Fields that are correctly absent are excluded explicitly. missing()
        already scores them certain() (1.0), so they would never be the
        minimum regardless -- excluded anyway so the intent reads directly
        rather than depending on that staying true.
        """
        populated = [f for f in self.all_fields().values() if f.absent_reason is None]
        if not populated:
            return Confidence.unknown()
        return Confidence.weakest_link(*[f.confidence for f in populated])

    def required_confidence(self) -> Confidence:
        """The narrower question rollup() used to answer alone: are the
        fields we cannot proceed without sound? "Is this record usable at
        all" and "is anything in this record wrong" are different questions
        with different consumers -- this is the first one."""
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
            name for name, f in self.all_fields().items()
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

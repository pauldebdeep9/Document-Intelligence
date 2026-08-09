# ADR 0003: Confidence is a propagated signal, not a float

**Status:** accepted · **Date:** 2026-08-09

## Context
Extraction quality is not binary. A field can be right, wrong, absent, or read
off a page the OCR mangled. Downstream — the HITL queue, the auto-accept
threshold, the abstention rule — all need to distinguish these. A single float
attached late cannot, because by then the signals that produced it are gone.

## Decision
- `Confidence` carries a score plus the `Factor`s that produced it
  (OCR, layout, model logprob, schema validity, lexical check, master-data hit,
  arithmetic agreement, human review).
- Four combinators with distinct semantics: `independent` (product),
  `weakest_link` (min), `corroborate` (noisy-OR), `weighted`.
- `ExtractedField[T]` wraps every extracted value with confidence and a source
  `Span`. A value without provenance cannot be reviewed, so it is not trusted.
- The model never reports its own confidence. We compute it from signals we can
  audit.
- Record-level confidence is the weakest *required* field, not the mean.
  Averaging hides the one wrong field that makes a record unusable.

## Consequences
- Every extractor must assemble factors — more work per document type.
- `eval/metrics.py::calibration_bins` can answer whether a 0.9 field is right
  90% of the time. Without that, confidence is decoration.
- The auto-accept threshold becomes defensible to a process owner with a number:
  the auto-accept error rate.

## Revisit if
Calibration shows the combinators are systematically mis-shaped; the fix is
`weighted()` with fitted weights, not abandoning decomposition.

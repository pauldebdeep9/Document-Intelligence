# ADR 0006: Checks report an outcome, not just a score; dates use site context

**Status:** accepted · **Date:** 2026-08-10

## Context

The first P1-03 eval run against the real 20-document corpus measured
detection rate — of every field that actually disagreed with gold, what
fraction scored low enough to *not* auto-accept — at 3.45% (normalisation
axis). 56 of 58 real errors sailed through at ~0.999 confidence. 42 of
those 56 were the same root cause: `validators.parse_iso_date()` correctly
flags an ambiguous slash date (`03/04/2025` is a different date depending on
whether it is `%d/%m/%Y` or `%m/%d/%Y`) and returns a reduced score, 0.6, to
say so.

`extract/extractor.py`'s `_apply_check()` had exactly two paths: a score at
or above `_PASS_THRESHOLD` (0.5) corroborates (noisy-OR, can only raise
confidence); below it, conflicts (multiplicative, can only lower). 0.6
clears 0.5. An explicit expression of doubt was read as support and raised
the field's confidence — backwards, and not a tuning problem: no value of
`_PASS_THRESHOLD` fixes it, because 0.6 needs to *lower* confidence, and any
threshold below 0.6 that achieves that also reclassifies genuinely
ambiguous-but-correctly-flagged dates as outright failures, which
misrepresents them just as badly in the other direction.

A second, independent gap compounded it for line-item dates specifically:
`_wrap_line()`'s `scoped()` helper computed `promised_date`'s value via
`parse_iso_date(v)[0]` but never applied `_apply_check()` to `[1]` at all —
no LEXICAL factor reached a line date's confidence, ambiguous or not. Found
by direct inspection of a real artifact's confidence factors (only
`model`/`schema`/`layout` present on `promised_date`, `lexical` entirely
absent) while diagnosing why the numbers matched neither hypothesis cleanly.

Separately: `03/04/2025` is genuinely two different dates from the string
alone. No amount of format-priority reordering recovers that — the
information is not in the input. But the document *does* carry it:
`ship_to_site` resolves against `data/masters/sites.json`, which records
each site's actual date convention (`site_us42` → `%m/%d/%Y`, the corpus's
one US-format site and the origin of all 42 date errors in the P1-03 run).
`parse_iso_date()` was never given that context.

## Decision

### `Check`: outcome plus evidence, not a bare score

`common/confidence.py` gains `CheckOutcome` (`PASS` / `UNCERTAIN` / `FAIL`)
and `Check` (`outcome: CheckOutcome, confidence: Confidence`). Every
validator (`extract/validators.py`) and resolver (`extract/masters.py`) now
returns a `Check`, not a `Confidence` a caller has to reinterpret by
threshold. `_apply_check()` dispatches on `check.outcome` directly:

- **PASS** → `field.corroborate_with()` (noisy-OR, can only raise).
- **UNCERTAIN** → `field.discount()` (new on `ExtractedField`; `independent()`,
  can only lower, **not** appended to `conflicts` — nothing actually
  disagreed, the check just could not confirm).
- **FAIL** → `field.flag_conflict()` (unchanged: `independent()`, lowers,
  logged).

`_PASS_THRESHOLD` is deleted. A check that cannot express uncertainty will
always have it rounded to agreement or to failure; the fix is a state the
threshold could never reach, not a different threshold.

`resolve_supplier()`/`resolve_part()`/the inline `AGREEMENT` checks in
`_fold_agreement()` have no genuine third state today (master-data
resolution and arithmetic agreement are binary in the current design) and
map their existing pass/fail branches onto `PASS`/`FAIL` unchanged — the
"ambiguous in master" case stays `FAIL`, per ADR 0005: an ambiguous
near-match resolving to a miss with a conflict is deliberate, documented
behaviour, not something this ADR revisits.

### `parse_iso_date()`: site convention tried first, ambiguity kept even when resolved

`parse_iso_date(value, site_date_format=None)` tries the caller-supplied
site format first. `extract/masters.py` gains `resolve_site_date_format()`
(`ship_to_site` → exact name match in `sites.json` → `date_format`, `None`
if unresolved) — exact match only, same discipline as `resolve_supplier()`:
a near-miss site name must not silently borrow another site's convention.
`_wrap_purchase_order()` resolves it once, from the model's own raw
`ship_to_site` read, and threads it to every date field on the record,
header and lines alike. Unresolved falls back to the original fixed
priority order (`%Y-%m-%d`, `%d/%m/%Y`, `%m/%d/%Y`, `%d.%m.%Y`,
`%d-%b-%Y`) — never to a default convention.

A date that resolves through site context but would have been ambiguous
without it stays **UNCERTAIN**, not PASS — at 0.85, higher than a blind
guess (0.6) but below a value that was never ambiguous at all (0.95).
`ship_to_site` is itself an extracted field and could be wrong; site
resolution is real evidence, not proof.

`_wrap_line()`'s `scoped()` gained the same optional `lexical` parameter
`unscoped()` already had, wired for `promised_date` — closing the gap where
the line-date LEXICAL signal was silently dropped regardless of what
`_apply_check()` did with it.

## Consequences

- Files touched: `common/confidence.py` (`CheckOutcome`, `Check`),
  `models/records/base.py` (`ExtractedField.discount()`),
  `extract/validators.py`, `extract/masters.py`, `extract/extractor.py`.
- Every validator/resolver signature changes return type from `Confidence`
  to `Check`. All are internal to `extract/`; no external caller.
- `data/gold/extraction/*.json`'s `meta.ambiguous_date_fields` now also
  covers line-item `promised_date` (previously `po_date` only) — a
  metadata correction to `scripts/gen_corpus.py`, applied to the existing
  20 files in place, not a corpus regeneration.
- `docs/LIMITATIONS.md` opened to hold the one P1-03 finding this ADR does
  **not** address: 14 wrapped-table-cell description truncations, a parse/
  defect (P1-04 territory), unrelated to checks or dates.

## Revisit if

- A validator or resolver develops a genuine third state beyond
  PASS/UNCERTAIN/FAIL (e.g. master-data "found but stale") — extend
  `CheckOutcome` deliberately, per Signal's existing "closed, add
  deliberately" discipline, rather than overloading one of the three.
- Calibration (still not meaningfully measurable per the P1-03 report — 99.8%
  of scores land in one bin) eventually has enough spread to check whether
  0.85 (site-resolved) and 0.6 (blind guess) are themselves well-calibrated,
  not just correctly *ordered* relative to 0.95 and each other.
- A site's shipping address or date convention changes and `sites.json`
  is not updated — `resolve_site_date_format()` fails closed (falls back to
  the blind guess) rather than silently applying a stale convention, but a
  stale-but-present entry would silently misresolve; no staleness check
  exists.

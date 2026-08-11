# ADR 0005: Span location by string search, master-data matching without fuzzy fallback

**Status:** accepted · **Date:** 2026-08-09

## Context

`extract/` maps model output back onto the source document for two reasons:
provenance (a reviewer needs to jump to where a value came from) and, indirectly,
correctness (a value the source text does not contain is the cheapest
hallucination signal available). pypdf gives no bounding boxes, so the only
mechanism available is string search over the parsed block text. String search
is imprecise in ways that matter: the same short value can appear more than
once, and a value can be genuinely present but unlocatable because of how the
parser flattened the page. Both had to be designed for deliberately rather than
discovered as bugs later.

Master-data resolution (supplier, part) has a parallel problem: the corpus
deliberately contains confusable pairs (two Fastenal entities, two Kestrel
Industrial entities) to test whether resolution can tell them apart, and 9/20
documents omit `supplier_id` entirely, so ID-only lookup was never viable.

## Decision

### Span location: normalised string search, three outcomes not two

`spans.locate()` normalises whitespace on both sides before matching (layout-mode
text pads columns with runs of spaces) and, for numeric values, generates
thousands-grouped and alternate-decimal-precision search forms (`_precision_forms`)
keyed off the numeric value itself, not its string form -- `str(float)` drops
trailing zeros a fixed-precision source never had reason to drop (`250.0` never
prints as `250`; `11,572.40` round-trips through float as `11572.4`).

A miss is not one outcome, it is two, and they mean opposite things:

- **NOT_FOUND** (zero matches): the value may not be in the document at all.
  Negative evidence about the value's correctness.
- **AMBIGUOUS** (2+ matches): the value IS present, just not uniquely
  locatable. Says nothing about correctness -- a comma is not a word
  character, so a bare quantity of `1` collides with the leading digit of
  `1,801.47` even *within its own row*, and that is not evidence the quantity
  is wrong.

Collapsing both into a bare `None` would force every caller to either treat
them identically (wrong: an ambiguous-but-present value is not hallucinated)
or reconstruct the distinction by re-running the search. `locate()` returns a
`Located(span, outcome: SpanOutcome)` instead, with no truthiness shortcut, so
a caller cannot silently drop the distinction back to "None means miss."

**A miss never approximates.** No nearest-block fallback, no fuzzy match. A
wrong span sends a human reviewer to the wrong place on the document, which is
worse than sending them nowhere -- they at least know to search manually. This
is why a field with `span=None` gets that fact named explicitly in the HITL
enqueue detail.

### Row location: positional, not content-anchored -- measured, not assumed

The first version scoped a line item's fields to its own printed row by
locating the line's `part_number` document-wide, on the premise that a part
number is distinctive within its row. Measured against the real corpus before
trusting it: distinctive in only 18% of line items (50/278). The parts master
has ~10 distinct part numbers; any document with more line items than that has
repetition by pigeonhole, and the documents with the most line items -- the
30-42-line ones -- are exactly the documents this item cared most about
scoping correctly.

Replaced with `extract_rows()`: a structural regex, `^\s*(\d+)\s+(\S+)\s`,
matching a printed row's shape (leading ordinal, then a token) rather than its
content. This does not depend on how many times a part number repeats, and
measured resolution jumped from 18% to 90% across all five scoped fields
(quantity, unit_price, extended_price, line_number, promised_date), with the
alignment self-check (below) passing on 20/20 real documents including all six
30+-line ones.

**Never trust the zip on faith.** Positional matching fails silently when it
fails -- a wrong row-to-line correspondence produces a span that points at a
real row, just the wrong one, which looks exactly as healthy as a correct
span. So before any row text is used as a scope, `_row_scopes()` verifies that
the extracted row count matches the raw line count *and* that every row's
leading ordinal equals the corresponding raw line's `line_number`. On any
mismatch, every field on every line of that document gets `span=None` --
scoped and unscoped fields alike, because if the structural correspondence
cannot be trusted, an "unscoped" search landing somewhere in the right
document cannot be trusted either. This fired for real on the actual run: the
model dropped one line item out of 42 on `po_008.pdf` (line 380 missing
entirely), the row count came up 42 vs 41, and every line field on that
document was correctly suppressed rather than silently misaligned by one
position from there on.

### MASTER_DATA: exact match only, ID then name then normalised name

`masters.resolve_supplier()` resolves on `supplier_id` when present, else
exact `supplier_name`, else name normalised (casefold, whitespace-collapsed,
common legal suffix stripped). Every step is an *exact* match on the
normalised string. Never fuzzy, never edit-distance, never "closest match
above a similarity threshold." Verified directly against the corpus's
deliberately confusable pairs: normalising "Fastenal Industrial Supply Pte
Ltd" and "Fastenal Industrial Services Pte Ltd" produces `fastenal industrial
supply` and `fastenal industrial services` -- still distinct, because
normalisation only strips a fixed, small suffix vocabulary, never touches the
part of the name that actually differs. An ambiguous near-match -- including a
normalised collision, checked explicitly -- resolves to a miss with a
conflict logged, never to whichever candidate happens to score higher. Picking
a plausible-but-wrong supplier is the one failure mode this refuses to create,
because it is the one a downstream consumer cannot detect from the record
alone.

Parts follow the same shape with one addition: `data/masters/unmastered_parts.json`
lists parts that appear on real documents but are absent from the master *by
design*, so that "correctly extracted, doesn't resolve" is a real, populated
outcome class and not a theoretical one. A miss there contributes nothing
(`Confidence.unknown()`) rather than a conflict -- it is not evidence the part
number is wrong, the master simply has nothing to say about it.

### AGREEMENT: quantity x unit_price, tolerance measured against the corpus

`PurchaseOrder.line_total_agrees()` sums `quantity x unit_price` per line and
compares to the extracted `total_amount`, tolerance `Decimal("0.01")`. It does
**not** sum the printed `extended_price`: 4/20 corpus documents omit the
extended-price column while still printing a true total (the prompt instructs
the model to return `null` rather than compute one), and summing `None` there
would read as zero and fire a false conflict on a fifth of the corpus. The
`0.01` tolerance was not guessed -- checked directly against all 20 gold
records first: `total_amount` equals `sum(quantity x unit_price)` with zero
rounding error corpus-wide, because the generator computes the total from the
unrounded per-line products rather than summing independently-rounded
extendeds. `POLine.extended_price_agrees()` runs the same check per line where
an extended price *is* printed, catching a different, real error: on
`po_010.pdf`, line 270's `unit_price` was extracted as `1276.85` against a
true `1,223.36`, and the wrong value showed up as three independent signals
firing on the same root cause -- `unit_price` NOT_FOUND (the wrong string
matches nothing on the page), that line's `extended_price` disagreeing with
`quantity x (wrong) unit_price`, and the header total disagreeing too, because
the header check is re-derived from the record's own quantity/unit_price, not
from the (coincidentally correct) extended price. Both checks return `None`,
not `False`, when the inputs are incomplete (`total_amount` absent, or any
line missing quantity/unit_price) -- a data gap must never be reported as an
arithmetic disagreement, those are different claims about the record.

### PROVENANCE: a new signal, and why only one of the two miss outcomes uses it

Signal.PROVENANCE was added (the enum is documented "closed; add
deliberately," so this is named, not implicit) for exactly the NOT_FOUND case.
It folds in as `independent()` -- a value whose source text cannot be found at
all is genuinely separate negative evidence about the same claim (this value
is correct) from what MODEL/SCHEMA/LAYOUT already say, and the combination
should be able to pull confidence down multiplicatively rather than just
attach a note. AMBIGUOUS deliberately never reaches this: it is not evidence
either way, and folding it in -- even as a small penalty -- would punish a
correctly-extracted value for being short and generic, which is exactly the
information AMBIGUOUS is supposed to represent.

### Raw numerics stay `float`, `_precision_forms` compensates

`PurchaseOrderRaw`'s numeric fields (`quantity`, `unit_price`, `extended_price`,
`total_amount`) are typed `float`, not `str`, and stay that way deliberately.
Retyping to `str` would sidestep the trailing-zero mismatch this ADR describes,
but at a real cost: floats are what `ResponseCache` hashes on for structured
calls, what `Decimal(str(v))` converts predictably, and what strict JSON
schema validation constrains to `type: number` rather than an unconstrained
string the model could format arbitrarily. A string-typed amount field
reopens exactly the "digits and decimal point only" formatting discipline the
prompt currently gets for free from the schema. `_precision_forms()` -- try
the value at its natural `str()` precision, at a forced two decimal places,
and (if the value is a true whole number) with no decimal point at all --
compensates for the float round-tripping loss without giving up any of that.

### max_tokens: measured requirement, ceiling not allocation, and truncation fails fast

The first full-corpus run failed on every document with 36+ line items:
`SchemaRepairExhausted` after three attempts, each with an identical `Invalid
JSON: EOF while parsing a string` error. Diagnosis (a single non-repairing
call, inspecting the raw result rather than the parse failure):
`finish_reason: "length"`, `completion_tokens: 2048` exactly equal to the
configured `max_tokens`, response truncated mid-value. Measured, not
estimated, from six real documents once the cap was raised: PurchaseOrderRaw
output costs roughly 45-65 completion tokens per line item (2687 tokens for
42 lines, 1901-1906 tokens for 30 lines), so the old 2048 cap was exceeded
above roughly 36 lines -- four of the corpus's twenty documents, all at
exactly the line count this item's Definition of Done named as the hard case.

Set to 8192, not 4096: 4096 gives ~1.8x headroom over the largest current
document (42 lines, ~2700 tokens); P1-11 adds invoices to the same setting,
and a cap that is merely adequate for today's corpus silently truncates again
the moment a document type or corpus revision needs more, which is the same
bug deferred rather than fixed. 8192 gives ~3.6x headroom (~150 line items)
and costs nothing when unused: `max_tokens` is a ceiling passed with every
request, not a reservation, and only tokens actually generated are billed.

The more important half of the fix is not the number, it is that hitting the
cap now fails immediately and legibly. `llm/structured.py` checks
`result.finish_reason == "length"` before attempting to parse and raises
`OutputTruncated` (new, `ExtractionError` subclass, distinct from
`SchemaRepairExhausted`) without entering the repair loop. Retrying a
truncated response cannot succeed -- the retry gets the same `max_tokens` cap
and truncates at the identical point -- so the old behaviour paid for three
API calls to fail identically and then raised a generic JSON-parse error that
named none of this. `OutputTruncated` vs `SchemaRepairExhausted` is a
deliberate distinction for callers: one means the model got the content
wrong (a prompt or model problem), the other means the response didn't fit (a
config problem). They need different responses and must not be the same
exception.

## Consequences

- Span misses are informative rather than uniform: NOT_FOUND is a signal,
  AMBIGUOUS is deliberately not. A future signal that wants "value located but
  not confidently" has a real, distinct outcome to hang off rather than
  needing a fourth bucket.
- The positional row scope depends on the printed "Item" ordinal column
  existing and being a bare leading integer. A source layout without that
  column (a different template, a future doc type) gets zero rows and clean,
  loud alignment failures -- not silent misscoping -- but does get zero
  line-item spans until a second row-shape pattern is added.
- `max_tokens=8192` is shared across every doc type using this LLM settings
  block. P1-11 (Invoice) inherits the headroom for free; a doc type with
  meaningfully more line items than 150 would need this measurement redone,
  not just the number bumped again.
- `OutputTruncated` is currently uncaught above `extract/pipeline.py`'s
  per-document isolation boundary, same as any other extraction failure --
  it is counted and logged per document, not treated as fatal to the batch.

## Revisit if

- A source layout is added where line items are not printed with a leading
  bare-integer ordinal column -- `extract_rows()`'s structural pattern needs a
  second shape, not a content-anchored fallback (rejected once already, see
  above).
- Calibration (P1-03) shows `_NOT_FOUND_SCORE` (0.2) or the master-data
  pass/fail scores are mis-shaped against real outcomes -- fix the scores,
  not the combinator choices; the combinator choice (independent for
  NOT_FOUND, corroborate for a passing check, independent for a failing one)
  is a separate decision from where its inputs are calibrated to.
- A document type needs more than ~150 line items in one structured call --
  remeasure tokens-per-item for that schema rather than reusing 8192.

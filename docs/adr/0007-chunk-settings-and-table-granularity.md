# ADR 0007: Chunk settings, settled before P1-08's gold exists

**Status:** accepted · **Date:** 2026-08-10

## Context

`Chunk.id` is a pure function of `(document_id, ordinal, chunk_text)`
(`common/ids.py`'s `chunk_id()`), and `chunk_text` is itself a function of
`ChunkSettings` — how blocks get windowed, whether and how a table splits.
P1-08 will build retrieval gold against a specific run's chunk ids. Changing
`target_tokens`, `overlap_tokens`, or `max_table_tokens` after that gold
exists does not raise an error anywhere — it just silently produces a
different set of chunk ids than the gold references, and every recall
number computed against it becomes meaningless without looking wrong. The
only point at which this is cheap to get right is before that gold exists,
which is now.

Two questions had to be answered with evidence, not left at the
`ChunkSettings` defaults on the assumption they were fine: whether
`target_tokens`/`overlap_tokens` (512/64) suit this corpus, and what
`max_table_tokens` should be, since P1-03/P1-04 already established that
this corpus's largest tables run to 40+ rows.

**Measured against the real 20-document corpus** (`tiktoken`, `cl100k_base`,
post P1-04 table reconstruction):

| | min | median | max |
|---|---|---|---|
| whole document (`Document.text()`) | 246 | 340 | 2099 |
| TITLE block | 3 | 3 | 3 |
| KEY_VALUE block | 6 | 15 | 28 |
| PARAGRAPH block | 8 | 18 | 41 |
| TABLE block (one page's portion) | 62 | 197 | 1883 |
| single table row | 32 | 43 | 51 |

12 of 20 documents are under 400 tokens *in total* — small enough that
their entire non-table prose (title + every KEY_VALUE + every PARAGRAPH
block combined) never approaches even a few hundred tokens, let alone 512.
The four 42-line, two-page documents are the outliers: their page-1 table
portion alone runs 1733–1883 tokens (38–41 rows), because reportlab packs
as many rows as fit before forcing a page break, leaving only 1–4 rows to
spill onto page 2.

## Decision

### `target_tokens=512`, `overlap_tokens=64` — kept

Every document's non-table prose, summed, stays under ~150 tokens even on
the largest documents (title ~3 + four KEY_VALUE blocks ~60 + two
PARAGRAPH blocks ~50). At 512, the windowing path in `chunk_document()`
essentially never closes a prose window mid-document for this corpus: the
header/footer text always fits in one chunk together, which is the
*correct* outcome for retrieval quality here — a query about "who is the
supplier" needs PO number, supplier name and dates in the same chunk, not
fragmented across three. There is no evidence in this corpus that 512 is
too large; there is also no evidence it is exercised much yet. It stays at
the default because nothing here argues for changing it, and because
other doc types on the roadmap (contracts, SOPs) have denser prose where
this budget will actually matter.

`overlap_tokens=64` follows the same reasoning: it has effectively no
observable effect on this corpus (prose never needs a second window to
overlap with), and is left at a conventional value for when it does.

### `max_table_tokens`: **1500 → 800**, changed with evidence

At 1500, a 38–41 row table splits into as few as **2** parts — e.g. rows
10–1180 and 1190–1420 (illustrative numbers at this token scale). A
citation naming "rows 10–1180" spans nearly the whole table and is barely
more specific than "the table" — exactly the failure this milestone exists
to fix (`docs/WBS-P1.md`'s own "Watch: a half table retrieves confidently
and answers wrongly" is about splitting *correctness*; this is the same
concern applied to split *granularity*).

At 800 (row cost ~43 tokens median, header ~20): the same 38–41 row table
splits into **3–4** parts of roughly 15–18 rows each — `docs/adr/0007`'s
own worked example from a live run was rows 10–170, 180–340, 350–380,
390–420 on a 42-line document (four parts, not two). This puts a table
chunk's *retrieval-unit scale* roughly in line with a prose chunk's
(512 tokens), rather than 3x larger, and makes `line_range` actually
useful as a citation: 15–18 line items is something a reviewer can scan,
40 is not meaningfully different from "the whole table."

The trade-off is more chunks (and therefore more embedding calls) for the
four largest documents specifically — 16 documents are entirely unaffected,
since their tables are 62–244 tokens and stay whole either way.

### Budget enforcement corrected: measure the text the chunk ships, not a cheaper stand-in

`_table_parts()`'s first version decided where to split by measuring
`estimate_tokens()` against the rows joined with a flat two spaces — an
internal representation never shown to anyone — while the chunk actually
ships `Table.to_markdown()` (see `_table_chunk()`). Markdown's `| cell |`
pipes and the `|---|` separator row are real tokens with a real cost that
scales with column count, and they were not in the number the split
decision was made against. Measured on the real corpus: **10 of 34 table
chunks shipped over `max_table_tokens=800` once rendered, by 28–93 tokens
(3.5–11.6%)** — all the largest parts of the four 8-column, 42-row tables.

This gets worse, not better, as a fixed absolute overhead: the next
document type is not guaranteed eight columns, and every additional column
adds another `| ` `| ` pair to every row. A budget enforced against a
narrower column count than the shipped table will overshoot by more, not
less.

**Fixed:** `_table_parts()` now builds each candidate part and measures
`estimate_tokens(candidate.to_markdown())` directly against
`max_table_tokens`, rather than reasoning about markdown overhead
separately from the plain-text row cost. `_rows_text()` (the internal
join) is gone — nothing needs it once the shipped text is the only text
measured.

**Re-verified, not assumed, after the fix:**
- **Zero table chunks over `max_table_tokens=800`** across all 34 (was 10).
  New table token range: 83–797 (was 83–893).
- **Chunk counts per document did not change** — this is worth stating
  precisely because the opposite was predicted before measuring. All 6
  documents that split before still split into the same number of parts
  (4 parts for the four 42-line documents, 2 for the two 30-line ones);
  what changed is the row boundary *within* that same part count (e.g.
  `po_008.pdf`'s first two parts moved from 17+17 rows to 16+16 — one
  fewer row each, not an additional part). The markdown overhead this
  fix accounts for was large enough to matter for staying under budget,
  but not large enough to push any part over a row-count threshold that
  would force a fifth or third part. Total table chunks: still 34.
- `chunk_id`s for every affected table chunk changed (text content
  changed with the row boundaries), confirmed by construction — nothing
  downstream depends on them yet, which is exactly why this was worth
  fixing now rather than after P1-08 exists.
- `line_range` coverage re-verified across all 20 documents: still exact,
  no gaps, no overlaps, no duplicates against gold's line_number sequence.
- Determinism re-verified: two independent `chunk_document()` runs over
  all 20 documents produce identical `chunk_id` sequences.

### `keep_tables_whole=True` — kept, given a real meaning

Rather than a dead boolean, `False` routes a TABLE block through ordinary
prose windowing instead of the dedicated table path — still never splits
a row (windowing never splits within a single block's text at all), but
loses per-row-boundary splitting, header repetition, and `line_range`.
This is an escape hatch for a future doc type whose "tables" are not worth
the structural treatment, not a way to defeat the one correctness rule.

## The flattening regression: structural verification was not enough

While measuring the extraction delta for P1-04 step 1 (before any of the
settings work above), `parse/chain.py`'s first version of `_table_text()`
rendered a reconstructed table by joining cells with a flat two spaces
(`"  ".join(row)`), discarding the original printed table's column
alignment. Verified structurally at the time — correct values, correct row
count, correct continuation folding, everything a diff against gold could
check — and shipped.

It broke a document anyway. `po_018.pdf`'s `extended_price` went from 8/8
correct (the pre-P1-04 run, reading the original wide-spaced text) to 8/8
missed, all at confidence 1.0, once real extraction was re-run against the
flattened text. Root cause, confirmed by diffing old vs. new table text
directly: `po_018.pdf`'s descriptions range from "Spare parts kit 4420" to
"ControlLogix processor module" — different lengths — so with a flat
2-space join, `Unit Price` and `Extended` land at a different horizontal
offset on nearly every row, with no consistent gap between them anywhere.
The values were there. The model could not tell where one column ended and
the next began.

**Why structural verification missed it.** Every check that was run before
shipping — row/ordinal alignment against gold, "does every gold description
appear complete in the text" — asks whether the *information* survived
reconstruction. None of them asks whether the *layout* the model reads did.
A flattened table and a column-aligned table contain identical
information and are, by every structural test in this codebase,
indistinguishable. They are not indistinguishable to the model: column
alignment is not decoration on top of the data, it is itself a signal the
model uses to segment the row into fields, and this corpus's own original
tables prove it — every one of them uses generous, consistent padding, not
because reportlab defaults to it, but because that is what makes a table
readable at all, by a human or a model. Removing it is not a neutral
formatting choice.

**Fixed:** `_table_text()` pads every column to its own widest value
(header included) before joining, restoring consistent vertical alignment
— structurally the same information as the flat-join version, formatted
the way the rest of the corpus's tables already print.

**Fixed and verified against a live model, not just structurally.**
Re-ran `isc extract` for real against the padded text ($0.0162, 20
documents, ~$0.0008/document, ~32k prompt + ~19k completion tokens — both
figures essentially identical to the flat-join run, since the fix pads
rather than lengthens content). `po_018.pdf`: 8/8 `extended_price` correct.
The fix holds against the thing that actually broke, not just against the
structural check that failed to catch the break in the first place.

The same live run also directly confirmed something only suspected before:
**both genuine model extraction errors P1-03 measured are gone.**
`po_008.pdf`'s dropped line 380 (`compare_lines()`'s `dropped_line`
outcome) — zero `line_outcomes()` anywhere in the corpus now. `po_010.pdf`
line 270's price misattribution (`unit_price` had been reading line 290's
`1276.85` instead of its own `1223.36`) — line 270 now reads `1223.36`
exactly, matching gold. Both were genuine reading errors *caused by* text
that was harder to parse than it needed to be (a wrapped continuation the
model never saw at all; two visually similar adjacent rows sharing a part
number in a less legible table); better table text was enough for the
model to read both correctly without any change to the extraction prompt
or logic. The corpus's error surface is now not "P1-03's findings, minus
14 description truncations" — it is close to empty. One error remains,
newly appeared: `po_010.pdf` line 20 `extended_price` read as `568040.00`
against a gold (and internally consistent — `50 × 1136.08 = 56804.00`)
`56804.00`, a 10x digit slip against unambiguous, well-formatted source
text this same run read correctly on every other line. Caught cleanly
(confidence 0.02, reject band) — not a false negative, and given the
source text here shows nothing structurally wrong, this reads as ordinary
model call variance (`temperature=0.0` bounds sampling, it does not
eliminate it) rather than anything caused by this fix.

Net across the whole corpus (2417 field/line outcomes, both axes):
**detection rate 1/1, false negatives 0, review-band precision 0/93 wrong
(down from 89 total / 15.7% wrong at the P1-03 baseline — that comparison
*is* meaningful, n=93 both times), auto-accept error rate 0.000%.** The
detection-rate ratio itself is reported for completeness, not as a result:
at n=1 real error corpus-wide, "1/1" is not a measurement of the routing
signals' ability to catch errors, because there is almost nothing left in
this corpus for them to be measured against. That is a different, better
problem than the one P1-03 was measuring, and worth being precise about
rather than quoting a ratio that happens to read as 100%.

**The general lesson, which outlives this corpus:** any normalisation of
document layout — collapsing whitespace, re-flowing a table, "cleaning up"
formatting before it reaches a model — has to be checked against what the
model can still read, not only against what information technically
survives. A diff against gold values is necessary and was not sufficient
here; the corpus's own printed formatting was quietly carrying structural
information (which digits belong to which column) that a values-only check
has no way to represent, let alone verify. The fix for *that* problem in
general is not "diff harder" — it is: when a change reformats text a model
will read, verify it against a live model call before trusting it, the
same way this session should have (and now has) for this specific case.

## FRAGMENTED (`extract/spans.py`'s `SpanOutcome`): not dead, but currently unused

P1-04's table reconstruction removed the one real source of `FRAGMENTED` in
this corpus — a wrapped continuation scattering a description's tokens
across two physical lines with the rest of the row's other columns sitting
between the pieces (see `docs/LIMITATIONS.md`'s first entry). Checked
directly, not assumed: re-ran `locate()` for every field actually extracted
by the model, against every one of the 20 post-fix parsed documents (2,322
locate() calls: 1,440 FOUND, 881 AMBIGUOUS, 1 NOT_FOUND, **0 FRAGMENTED**).

Left in place. A wrapped cell the column-boundary logic in
`_reconstruct_table()` cannot resolve — a table with an unusual layout this
corpus does not contain — could still produce it, and the mechanism (two
independent code paths in `locate()`, scoped and unscoped) existed for a
real reason before this fix. Deleting an outcome branch because the current
corpus does not exercise it is how it comes back as a silent regression the
next time a genuinely fragmented value shows up. Two synthetic tests
(`test_spans.py`) now exercise both the scoped and unscoped path directly
via a constructed `Document`, so the branch cannot rot into untested dead
code the way it would if it depended on the real corpus continuing to
produce it.

## The settings-fingerprint guard

`index/chunker.py`'s `settings_fingerprint(settings)` is a deterministic id
(`common/ids.cache_key`) over `ChunkSettings`' full field set.
`storage/local_vector.py`'s `LocalVectorStore.add()` takes an optional
`settings_fingerprint`: the first call records it, persisted through
`save()`/`load()`; any later call with a *different* value raises
`ChunkSettingsMismatch` (new, `common/errors.py`, same "never caught and
downgraded" discipline as `AclViolation`) rather than silently mixing chunk
boundaries built under two different configurations into one store.
`index/pipeline.py` computes the fingerprint once per run and threads it
through.

This is the chunker/index half of the guard. **The other half — P1-08's
retrieval gold recording which fingerprint it was built against, and
checking it at eval time — does not exist yet, because P1-08 does not
exist yet.** `settings_fingerprint()` is the function P1-08 is expected to
call and compare against; wiring that comparison in is P1-08's own work,
not retrofitted here.

## Which settings actually do anything on this corpus — stated plainly

Measured across all 74 chunks from all 20 documents: **prose chunks max out
at 94 tokens against a `target_tokens=512` budget.** Not one prose chunk
in this corpus gets within half the budget, let alone closes a window
because of it. `overlap_tokens=64` follows from the same fact — a second
window is never opened for prose here, so there is nothing for it to
overlap with.

**`target_tokens` and `overlap_tokens` are inert on this corpus.**
`max_table_tokens` is the only one of the three settings that does any
observable work: it is the sole reason 6 of 20 documents produce more than
3 chunks at all, and the only one a real change to (1500 vs. 800) produces
a measurably different corpus.

This narrows the actual risk the settings-fingerprint guard exists for. It
was framed as protecting against a change to any of `target_tokens`,
`overlap_tokens`, or `max_table_tokens` silently invalidating P1-08's gold
— true as stated, but on *this* corpus specifically, only a
`max_table_tokens` change would ever produce a different set of chunk
boundaries to invalidate against. The guard still earns its place: it is
cheap, it is general (it does not know or care which setting changed), and
"inert here" is a fact about this corpus's current documents, not a
property of the settings that a future document type is bound to share
(see "Revisit if" — denser prose is exactly the case that would make
`target_tokens` live). But the number worth carrying forward is narrower
than the original framing: **the freeze that actually matters today is on
`max_table_tokens`**, and it is the one setting this ADR changed on
evidence, not the two it left alone.

## Consequences

- `config/default.yaml`'s `chunk.max_table_tokens` changed 1500 → 800.
  Nothing downstream depends on the old value yet (P1-05/P1-08 unbuilt), so
  this costs nothing today and would cost a full gold regeneration after
  P1-08 ships.
- Measured, not assumed (see "Budget enforcement corrected" above): the
  `max_table_tokens` change does **not** change chunk counts per document
  on this corpus — the same 6 documents split into the same number of
  parts before and after. What changed is which rows land in which part,
  and, once the markdown-measurement fix above is included, whether every
  part actually respects the configured budget.
- `LocalVectorStore.add()`'s `settings_fingerprint` parameter is optional and
  defaults to `None` (no check) — existing/manual callers are not broken,
  but also not protected until they pass one.

## Revisit if

- A future doc type's prose genuinely approaches or exceeds 512 tokens per
  section (this corpus never does) — re-measure rather than assume the
  same conclusion holds.
- P1-08's gold generation needs the comparison side of the settings guard —
  implement it against `settings_fingerprint()`, do not reinvent the hash.
- A real document (any doc type) produces a genuinely unresolvable wrapped
  cell and `FRAGMENTED` starts firing for real — recalibrate whether
  `_apply_span_outcome()` treating it as neutral (vs. a small penalty, the
  way `NOT_FOUND` is penalised) is still the right call once there is a
  real distribution to look at instead of zero occurrences.
- `po_010.pdf` line 20's `extended_price` misread recurs on a later
  re-run against unchanged text — one occurrence against otherwise-clean
  source text reads as model call variance, not a pattern; a repeat would
  say otherwise and is worth investigating as its own finding, not folded
  into this one.
- Any future change reformats text between parse/ and the model (table
  rendering, block joining, whitespace normalisation) — verify it against
  a live model call before trusting it, per "The flattening regression"
  above. A structural diff against gold is not sufficient for a change
  that alters what the model reads, only for one that alters what
  information exists.

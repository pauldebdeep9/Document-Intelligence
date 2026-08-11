# ADR 0009: Binding resolves a citation; it does not verify what it supports

**Status:** accepted · **Date:** 2026-08-11

## Context

`answer/citations.py`'s `bind_citations()` parses `[n]` markers out of a
generated draft and resolves each to `hits[n-1]`. It answers exactly one
question: does this marker point at a chunk that exists in the retrieved
set. It cannot answer a different one: does the sentence next to the marker
describe what is actually in that chunk. A marker resolving to a real,
ACL-permitted chunk is not evidence the prose beside it is true — the
model can cite a real chunk and still misattribute the fact it contains.

**Found live, not hypothesised.** Running P1-07's 11-question sample,
`q_re_10` asked "what did we pay per unit for the ControlLogix processor
module from Omron Electronics Asia" as `u_ben`. Corpus-wide retrieval (no
PO filter — restricted_unfiltered by design, see the P1-08 gold set)
surfaced a table row `u_ben` could legitimately read. Binding passed: `[1]`
resolved to a real, permitted chunk. The chunk was `po_001.pdf` line 120 —
**Keyence Singapore Pte Ltd**, not Omron. The model's own sentence named
the wrong supplier for a real, correctly-cited number.

### Text matching fails specifically on table and footer chunks, not randomly

The first fix (`verify_attribution()`) checked whether a sentence's named
entities appeared as a *substring of the cited chunk's raw text*. Measured
against the same 11-question sample: **3 of 4 flagged answers were false
positives**, and all three shared one exact mechanism — `chunker.py` never
repeats a document's header fields (`supplier_name`, `po_number`) into its
table-row or footer chunks. `q_li_02`'s citation was a table row
(`| 420 | TRM-BLK-2P5 | ... | 2,179.45 |`) that correctly supported the
answer and, by construction, could never contain the phrase "PO
4591914574" no matter how correct the answer was. Text matching asks *does
this chunk mention Omron* — a question a correct table-chunk citation will
routinely fail. `Chunk.filters` (`po_number`, `supplier_id`, populated by
`filters_from_record()` at index time — see `index/chunker.py`) answers the
question that actually matters: *does this chunk belong to an Omron
order*, and it answers it correctly regardless of chunk shape, because
every chunk of a document carries the same filters dict whether it is the
header, a table row, or a footer.

**Re-checked all four cited chunks' `.filters` directly**: every one of
the three false positives carried the correct `po_number`/`supplier_id`
even though its text didn't. `u_ben`'s chunk carried `supplier_id: V103014`
(Keyence), not `V102337` (Omron) — filters catch the true positive too.
Switching the check from text to filters is not a broader net; it is a
narrower, more accurate one.

### The missing-metadata gap has to fail open, not closed

9 of 20 documents in this corpus have no `supplier_id` at all — genuinely
absent from the source PDF, not an extraction defect (`extract/masters.py`
already documents this: ID-only lookup was "never viable" for the same
reason). A chunk from one of those documents carries no `supplier_id`
filter key regardless of how correct the answer citing it is. Checked
directly against the full 56-question P1-08 gold set with verification
active: **3 sentences hit exactly this** — `q_cd_01` (`po_017.pdf`,
Omron), `q_re_11`/`q_re_12` (`po_014.pdf`, SKF Bearings Manufacturing).
Failing here would abstain a correct answer because a field was absent
from the PDF, which is a worse failure mode than the one this check exists
to catch — punishing the answer for a gap in the source document it had no
way to fill. `AttributionResult` separates `unverifiable` from
`mismatches`: a missing filter key on every cited chunk records as
unverifiable and does not block the answer; a filter key that is *present*
and *disagrees* is a real mismatch and does.

The raw text check is not deleted — it remains the fallback for an entity
kind with no filter-key mapping at all (`_filter_key_for()` returns
`(None, None)`; today neither `supplier` nor `po_number` ever does, so this
path is unexercised but present for a future entity kind that, per ADR
0007's "chunk-level filters are single-valued by design," structurally
can't have one — a table row's own `part_number`, for instance).

### Per-sentence granularity produced two more false positives — both were splitter bugs, not the granularity question

Measured on the full 56-question gold set with filters-based verification:
**2 of 35 answerable questions (5.7%) flagged `ATTRIBUTION_MISMATCH`.**
Both were investigated against ground truth before accepting the number,
not assumed:

- `q_cd_02` ("What did we spend with Keyence Singapore Pte Ltd in total,
  in SGD?") — the model retrieved 8 chunks, only 2 of which
  (`po_001.pdf`, `po_018.pdf`) are actually Keyence, and summed **all
  eight** order totals — `po_006.pdf` (Molex), `po_014.pdf` (SKF),
  `po_017.pdf` (Omron), `po_008.pdf` (Phoenix Contact) included —
  into a single confident "$5,489,309.94" attributed entirely to Keyence.
  A genuine hallucination, correctly caught.
- `q_cd_04` ("What did we spend with Kestrel Industrial AG in total, in
  USD?") — summed totals including `po_012.pdf`, whose `supplier_id` is
  `V100782` (**Kestrel Industrial *Pneumatics GmbH***, the corpus's
  other, deliberately confusable Kestrel entity — see ADR 0005/gen_gold.py's
  `q_am_*` ambiguous class), and `po_013.pdf` (Keyence Singapore, unrelated
  entirely). The model conflated two different Kestrel entities and pulled
  in a third supplier. Also a genuine hallucination, correctly caught.

**False positive rate on answerable questions: 0/35 (0%).** Both flagged
cases are real. This matters because the check does not itself weigh
sentence-vs-answer granularity in the abstract — it was verified against
what a live model actually produces on this corpus.

Two real bugs surfaced investigating this, both in `_split_sentences()`,
neither in the granularity design itself:

1. A buyer initial ("**A. Tan**", "J. Ruiz", "M. Weber" — this corpus's own
   `buyer_contact` values) reads as end-of-sentence to a naive
   `.`-followed-by-capital regex. Every `single_hop` question whose gold
   answer is a two-part name false-positived on this before the fix — a
   splitter bug, not evidence against per-sentence checking.
2. A numbered list item ("**1.** Order Total: ... [4]") reads the same
   way. `q_cd_04`'s intro sentence split before "1.", stranding the named
   supplier with no citation while the correctly-cited totals landed in
   later fragments — again a splitter bug, and it inflated the
   attribution-mismatch count with a spurious "cites no chunk at all"
   before being fixed.

Both fixed with one mechanism (`_LIST_MARKER_OR_INITIAL`, a placeholder
swap rather than accumulating lookbehind alternatives — Python's `re`
requires fixed-width lookbehind, and "how much whitespace precedes a list
marker" is not fixed width). Re-measuring after the fix did not change
which answers get flagged, only the accuracy of *why* — reinforcing that
false positives from sentence-tokenisation are a solvable engineering
problem, separate from the true positives the check exists to catch.

## Decision

Citation binding (`bind_citations()`) and attribution verification
(`verify_attribution()`) are two different checks, in that order, and a
system is not "grounded" until both pass:

1. **Binding**: does `[n]` resolve to a real, retrieved chunk.
2. **Attribution**: do the supplier names and PO numbers named in the
   sentence citing `[n]` match that chunk's `Chunk.filters` — the same
   metadata every chunk of a document carries regardless of chunk shape,
   not the raw text a table or footer chunk structurally never repeats.

Per-sentence, not per-answer: a chunk cited three sentences ago does not
license a claim in this one just because both appear somewhere in the same
draft. Known limitation, kept deliberately: a sentence that legitimately
continues a subject named by an earlier citation, without re-citing it
itself, still fails — there is nothing in *that* sentence's own markers to
check against (`test_verify_attribution_known_false_positive_on_an_uncited_later_reference`).
Not fixed here; not currently measured as a real cost on this corpus's
actual model outputs (0 occurrences across the 56-question run), so
loosening the granularity now would be solving a problem that has not
shown up, at the cost of the precision that just caught two real
hallucinations.

A missing filter key is `unverifiable`, treated as a pass, and recorded
separately from a present-but-disagreeing filter key, which is a `mismatch`
and aborts the whole answer. `AbstentionReason.ATTRIBUTION_MISMATCH` is
distinct from `UNGROUNDED_DRAFT` (binding failure) for the same reason
`INSUFFICIENT_CONTEXT` is distinct from both (ADR 0008's sibling decision)
— P1-09 needs to tell "the citation didn't even resolve" apart from "the
citation resolved to the wrong thing," and both apart from "the model
correctly declined."

**The general principle, stated plainly: a citation that resolves is not a
citation that supports.** Existence and support are different claims, and
a verifier that only checks the first will pass a confident, well-cited,
wrong answer every time the wrong chunk happens to be one the model was
allowed to read. In a supply-chain context that is the failure mode that
gets acted on.

## Consequences

- Every `AnswerOrchestrator` loads `extract.masters.supplier_ids_by_name()`
  once at construction — a new public function alongside the existing
  `supplier_names()`-shaped helpers in `extract/masters.py`, reusing that
  module's own `@lru_cache`'d file read rather than a second loader.
- `verify_attribution()`'s return type is `AttributionResult`
  (`mismatches`, `unverifiable`), not a bare list — the caller only acts on
  `mismatches`, but the split is what makes the unverifiable numbers in
  this ADR reportable at all.
- Cost: the 56-question full-gold-set run cost $0.0109 (64,536 prompt +
  2,060 completion tokens, gpt-4o-mini) for the ~45 questions not already
  cached from the P1-07 sample. Re-running verification alone after fixing
  the splitter bugs cost $0 — verification is pure post-hoc logic over an
  already-generated draft, never a second model call.

## Revisit if

- The per-sentence granularity's known false-positive shape (an uncited
  sentence continuing an earlier citation) is ever measured to occur on
  real model output — it has not, across 56 real questions on this corpus
  — before spending the complexity of a per-answer or hybrid design.
- Entity extraction is widened beyond supplier names and PO numbers to a
  field with no chunk-level filter (a table row's own `part_number` —
  ADR 0007) — that is what `_filter_key_for()`'s `(None, None)` / text
  fallback path exists for; verify it is actually exercised once it is,
  rather than trusting it unexercised indefinitely.
- `_split_sentences()`'s placeholder mechanism meets a third abbreviation
  shape this corpus doesn't produce yet (a currency-code-like token
  followed by a period, say) — extend `_LIST_MARKER_OR_INITIAL`, don't
  add a third accumulated lookbehind.

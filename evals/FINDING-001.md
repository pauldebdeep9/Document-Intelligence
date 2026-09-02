# Finding 001: po-002 payment_terms/currency contradiction — cross-document contamination

**Status: post-hoc exploratory investigation, not a pre-registered finding.** Nothing in
`evals/PREREGISTRATION.md` §3 predicted this question would be examined at this level of
detail; this document exists because `evals/BASELINE.md`'s two runs surfaced an apparent
contradiction worth explaining. Per §5's own separation rule, this is kept in its own document
rather than folded into `BASELINE.md`, which is not edited here. **n=2**: exactly two absent
questions (`po-002-q3`, `po-002-q4`) on exactly one document (po-002). Nothing below
generalizes beyond this one document pair without further data.

Investigation was entirely offline: `isc.pdf.extract_pdf_pages` run directly against
`data/gold/pdfs/po-002.pdf` (pure local PDF parsing, no network), plus the two already-saved
`RunRecord`s under `runs/` (`beda2db6-b925-4201-8c27-44d18c08e857` for k=3,
`d78f8dee-627f-4c0f-8cf0-6f27964dfc1d` for k=1). No API call was made or needed.

---

## What the model actually saw

**po-002's full extracted text** (one page, verbatim):

```
PURCHASE ORDER

PO Number: PO-1002
PO Date: 2026-02-10
Supplier: Meridian Fasteners
Ship-to Site: Dockside Warehouse 2

Line Items:
1. Part Number: HXN-0812
   Description: Hex nut, M8
   Quantity: 400
   Unit Price: 1.20
2. Part Number: WSR-0820
   Description: Flat washer, M8
   Quantity: 800
   Unit Price: 0.20

Total Amount: 640.00
```

Confirmed by direct substring check (case-insensitive) over this text: `"payment"`,
`"net 30"`, `"net30"`, `"usd"`, `"$"`, and `"currency"` are **all absent**. po-002 genuinely
never states payment terms or a currency anywhere in the document. The extraction path's null
verdict for both fields is correct.

**What was retrieved for po-002-q3 ("What are the payment terms on this purchase order?") and
po-002-q4 ("What is the currency for this purchase order?"), in both runs:**

| Run | Retrieved doc_ids | po-002's own chunk retrieved? |
|---|---|---|
| k=3 | `[po-001, po-010, po-004]` | **No — zero of three** |
| k=1 | `[po-001]` | **No — zero of one** |

In neither run, at neither `k`, was po-002's own (only) chunk retrieved for either of its own
absent questions. Every retrieved chunk, in both runs, came from a different document.

Those retrieved chunks are real and each genuinely states payment terms and currency:

- `po-001:page-001-chunk-001` — `"...Ship-to Site: Riverside Plant\nPayment Terms: Net
  30\nCurrency: USD\n..."`
- `po-010:page-001-chunk-001` (k=3 only) — `"...Ship-to Site: Hangar 4\nPayment Terms: Net
  30\nCurrency: USD\n..."`
- `po-004:page-001-chunk-001` (k=3 only) — `"...Ship-to Site: Bay 3\nPayment Terms: Net
  30\nCurrency: USD\n..."`

All three DEV documents that print payment terms happen to print the identical values (`Net
30` / `USD`) — corpus boilerplate, not a coincidence specific to which sibling got retrieved.

## Did the answer cite its sources, and were the citations supported?

`source_chunk_ids` in both runs, for both questions, exactly equals `retrieved_chunk_ids` —
every retrieved chunk was cited. Checking each cited chunk's own text directly (not the
question's gold document, the cited chunk's own content):

| Run | Question | Cited chunk(s) | Chunk text contains "Net 30" | Chunk text contains "USD"/"$" |
|---|---|---|---|---|
| k=3 | po-002-q3 | po-001, po-010, po-004 (all 3) | True, True, True | True, True, True |
| k=3 | po-002-q4 | po-001, po-010, po-004 (all 3) | True, True, True | True, True, True |
| k=1 | po-002-q3 | po-001 | True | True |
| k=1 | po-002-q4 | po-001 | True | True |

**Every citation is textually supported by the chunk it cites.** This rules out the specific
grounding-failure shape the task raised as a possibility — an answer asserting a value while
citing a chunk that doesn't actually contain it. That did not happen here. `answer_question`
did exactly what it is supposed to do with the context it was given: it answered using text
that is really there, and cited the chunk that really contains it. The chunk was simply not
from po-002.

## Which explanation the records support

**Cross-document contamination, not confabulation — determined directly from the data above,
not inferred.** The two candidate explanations made different, checkable predictions:

- Confabulation predicts: po-002's own chunk (or no chunk at all) was in context, and the
  model supplied "Net 30"/"USD" from parametric knowledge with nothing in the retrieved text
  to point to.
- Contamination predicts: a chunk from a document that genuinely states real payment
  terms/currency was retrieved and put in context, and the model's answer traces back to that
  chunk's real text.

The retrieved chunks and their content settle it: in both runs, po-002's own chunk was never
retrieved at all, and every chunk that was retrieved instead is a real chunk from a different
DEV document, genuinely containing the exact values the answer reported. The model's answer is
a faithful (and correctly cited) summary of the wrong document's content, not an invention.

## Implication

Corpus-wide pooled retrieval — the design this harness deliberately uses so the
`near_duplicate` class can test anything (`evals/runner.py`'s own docstring) — means an absent
question can be answered from a sibling document whenever that sibling shares enough
boilerplate structure with the query. Nothing currently in this harness detects that directly:
the retrieval metrics (`hit_at_k`/MRR) never run on `absent` questions at all (`build_report`
excludes them, by design, since they carry no anchors), and there is no metric anywhere that
checks an `absent` question's retrieved `doc_id`s against its own `doc_id`. This instance was
caught only as a side effect — the resulting free-text answer happened not to equal the exact
`INSUFFICIENCY_ANSWER` string, so `is_insufficiency_response` flagged it non-compliant. A
model that had instead wrapped the same contaminated content in exactly the required refusal
phrasing (or, differently, one that happened to answer using genuinely correct boilerplate by
chance) would have passed the absent-compliance check while the underlying retrieval failure —
zero relevant chunks retrieved for the document actually being asked about — went completely
unmeasured.

A plausible mechanism, visible directly in the data above and worth stating even though it
isn't proven by n=2: dense embedding similarity for a query like "What are the payment terms
on this purchase order?" is likely to favor chunks that *do* mention payment terms (po-001,
po-004, po-010 all do, verbatim) over po-002's own chunk, which never mentions payment terms
at all. For an absent-field question specifically, the correct document's chunk is
semantically *distant* from the query along the exact dimension the query asks about, while
every other document that happens to have that field is semantically close. This is a
structural property of how dense retrieval interacts with "the answer is that this field
doesn't exist here," not a one-off retrieval mistake.

## What would test this properly (proposed only — not implemented)

- Take the exact retrieved context from one of the contaminated calls above (e.g. k=1's single
  `po-001` chunk) and, offline, check whether `is_insufficiency_response` behavior is
  sensitive to *which* wrong-document chunk is substituted in — i.e., replay the same question
  against a hand-constructed context of (a) po-002's own chunk only, (b) a sibling chunk only,
  (c) both — to isolate whether the model's failure to refuse is caused by the presence of a
  plausible-looking wrong-document chunk specifically, independent of retrieval. This isolates
  the answer-path behavior from the retrieval-path failure, which this investigation could not
  do with the existing saved records alone since every recorded call already had contaminated
  context.
- Add a metric (not proposed as an implementation here, only as a gap) that checks, for
  `absent`-class questions specifically, whether any retrieved chunk's `doc_id` matches the
  question's own `doc_id` — today nothing computes this at all, for any question class, since
  `absent` questions are excluded from the retrieval section entirely.
- If this pattern were to be tested at scale, it would need more than n=2: every absent
  question on every document with at least one sibling that states the corresponding field for
  real, checked for the same zero-own-chunk-retrieved pattern, not just po-002.

## Determinism caveat

Extraction was identical across the two runs for po-002 (and for every other DEV document) —
both scored `payment_terms`/`currency` CORRECT (null/null), with `run_eval` re-executing
ingest independently per run rather than sharing it. This is **one paired observation on six
short, single-page-or-few-page documents**, not evidence that extraction is deterministic. It
should not be described as deterministic anywhere, including here: a different pair of runs,
a longer document, or a different field could disagree without contradicting anything
established so far.

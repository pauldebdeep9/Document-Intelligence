# Pre-registration: DEV baseline run

Written before any run, before any API call, before any result exists. This document is what
makes the eventual baseline number defensible rather than decorative — if it were written
after seeing results it would not be pre-registration, it would be a rationalization with a
timestamp on it. It is committed on its own, with nothing else in the diff, specifically so
its commit predates the run it governs.

Every claim below about what the code does was checked against `evals/metrics.py`,
`evals/runner.py`, `evals/report.py`, and `evals/gold/` on 2026-09-03, not written from memory
of having built them. One place where this document's draft would have described the code
differently than the code actually behaves is called out explicitly in §2.

---

## 1. Frozen configuration

| Field | Value | Status |
|---|---|---|
| `chat_model` | **`gpt-4.1-mini`** | proposed — **awaiting your confirmation** |
| `embedding_model` | **`text-embedding-3-small`** | proposed — **awaiting your confirmation** |
| `chunk_size` | `1200` | fixed — matches `goldset.chunking` (see below) |
| `overlap` | `200` | fixed — matches `goldset.chunking` |
| `k` | **`3`** | proposed — **awaiting your confirmation** |
| `goldset_version` | `1.0.0` | fixed — `data/gold/goldset.json`, confirmed on disk |
| `split` | `DEV` | fixed by this task — TEST and HELD are not touched |
| `git_commit_sha` | *the commit that adds this file* | see note below |

**Model names are proposals, not facts I looked up in this repo.** `.env.example` ships both
`OPENAI_CHAT_MODEL` and `OPENAI_EMBEDDING_MODEL` blank — nothing in this codebase names a
specific model anywhere. `gpt-4.1-mini` is proposed because `isc.llm` calls
`client.responses.parse(..., text_format=...)`, which needs a model supporting the Responses
API's structured-output mode; `text-embedding-3-small` is proposed as OpenAI's standard
small embedding model. I am not confident these are still the current recommended choices as
of your read of this document — confirm or replace both before the run.

**`k=3` is not free of a real caveat.** The DEV split's pooled candidate set has been measured
directly, not assumed: chunking all 6 DEV documents at `chunk_size=1200/overlap=200` produces
exactly **9 chunks total** (po-001: 1, po-002: 1, po-004: 1, po-005: 1, po-006: 3, po-010: 2).
`k=3` therefore retrieves **one third of the entire DEV candidate pool** for every question.
This is the existing default (`isc.retrieval.top_k_chunks(..., k=3)`, and `demo.py`'s
pipeline), so it is proposed for continuity rather than freshly invented — but at this pool
size it is a weak retrieval test almost by construction, and that weakens exactly the
near_duplicate slice this run cares about (§3, §6). Confirm `k=3` knowing this, or propose a
smaller value.

**`chunk_size`/`overlap` are not proposals — they are determined.** The goldset's anchors were
authored and validated against `chunk_size=1200, overlap=200` (`evals/gold/authoring.py`); that
pairing is recorded in `goldset.chunking` and is what `data/gold/goldset.json` actually
contains on disk (confirmed by reading it directly). A different chunking config is a
different, unvalidated set of anchors, not a variant of the same baseline.

**Frozen means frozen.** Changing `chat_model`, `embedding_model`, `chunk_size`, `overlap`, or
`k` after this document is committed produces a *different* run, not a comparable one. Any of
those five values changing invalidates this document as that run's pre-registration; a new one
must be written first.

**On the commit sha**: `run_eval` captures `git_commit_sha` automatically at execution time
(`evals/runner.py::_current_git_commit_sha`, a plain `git rev-parse HEAD`) — it is not
something this document needs to predict. This document is committed directly on top of
`2e7640e`. The run must execute at the commit that adds this file (this document's own
commit) with no intervening commits; if `RunRecord.git_commit_sha` from the actual run does
not equal that commit, the run happened against code this document did not review, and that
is itself a finding to report, not a detail to paper over.

---

## 2. What `evals/metrics.py` actually computes

**Retrieval, doc_id-aware.** An anchor (a verbatim span of gold page text, plus its page
number) is satisfied only if a retrieved chunk (a) is from the gold question's own `doc_id`
and (b) contains the anchor text as an exact substring of that chunk's own text.
Text matching in a chunk from a *different* document does not satisfy the anchor — it is
scored as a miss. This is not a minor detail: po-004 and po-005 have near-identical
line-item text differing only in the final digit of a part number (`4500123456` vs.
`4500123457`); without the doc_id check, a retriever that always returns po-005's chunk for a
po-004 question would score as correct on text-substring grounds alone. The doc_id check is
what makes the near_duplicate class capable of failing at all.

An anchor split across two retrieved chunks — present in neither one individually — is not
satisfied. Chunk texts are checked one at a time, never concatenated.

`hit_at_k` is *any* anchor satisfied (the standard IR hit@k definition), not *all* anchors
satisfied. This matters for `cross_page` questions specifically, whose gold has multiple
anchors (one per page the evidence spans) — `hit_at_k` here means "found at least one of the
required pieces," a weaker bar than "found every piece required to actually answer the
question." `first_hit_rank`/`reciprocal_rank` inherit the same "any" semantics — MRR reflects
how early the *first* relevant chunk was ranked, not whether all required evidence was
retrieved.

**Extraction, four-way per field, not binary.** `score_field` returns one of `CORRECT`,
`WRONG`, `MISSED` (gold expects a value, extraction returned null), `HALLUCINATED` (gold
expects null, extraction returned a value). Both-null is `CORRECT` — genuine agreement that a
field is absent, not a `MISSED` finding. Decimal fields compare by numeric value
(`Decimal("10.00") == Decimal("10")` is `True` natively); string fields compare exactly after
stripping only surrounding whitespace — no case-folding, no punctuation normalization. A
supplier name differing only in case, or differing by an ampersand vs. "and", scores `WRONG`,
full stop; the corpus was specifically built with an ampersand in a supplier name (po-009,
HELD — not touched by this run) to make that distinction real rather than theoretical.
`score_purchase_order` runs this over the seven header fields (everything on `PurchaseOrder`
except `line_items`).

**Line items exist as a metric but are not in the report.** `score_line_items` matches
expected line items against actual ones — by `part_number` when the expected item has one
(searched anywhere in the remaining unconsumed actual items, order-independent), by position
when it doesn't — and returns matched/missing/spurious counts. **This is the one place this
document's draft would have described the code differently than the code behaves**: I
expected `build_report` to surface a line-item slice given the metric exists and is tested,
but `evals/report.py` never calls `score_line_items` — it computes and renders only the
header-field verdicts. This run's report will therefore say nothing about line-item accuracy.
That is a real gap in `report.py`, not a misunderstanding on my part to paper over; it is out
of scope to fix in this document (this session writes no code), so §3 below pre-registers
nothing about line items — there is no report output to pre-register a question against.

**Insufficiency compliance is exact-string equality**, nothing else. `INSUFFICIENCY_ANSWER`
(`"I don't have enough information in the provided sources."`) is pinned directly in
`evals/metrics.py`, not imported from `isc.llm` (a prior session's deliberate decision, to
keep an `isc.llm` wording change from silently changing what this eval accepts). No
`.strip()`, no case-folding, no `startswith`. Trailing whitespace, a swapped final period, a
case difference — all fail. This is binary string compliance, not a judgment of whether the
answer was reasonable.

**Reporting has no aggregate and no percentage anywhere**, by construction — `format_kn`
renders every count as `"k/n"`, and a structural test
(`test_metrics_module_has_no_percentage_or_rate_formatter`) asserts no public callable in the
module has "percent", "pct", or "rate" in its name. MRR itself is a mean (of reciprocal
ranks, within one `question_class`), rendered as a plain float, not a count — it is not
`format_kn`'d and is not a percentage either.

**`build_report` excludes two categories from every scoring slice**, both confirmed by reading
`evals/report.py` directly:
- `absent` questions are excluded from the retrieval hit@k/MRR slices (they carry no anchors)
  and scored separately, by `is_insufficiency_response`, in their own report section. The
  excluded count is stated in the report header.
- Any item with a recorded `error` (a per-item runtime failure `run_eval` caught and
  continued past) is excluded from *every* slice — retrieval, extraction, compliance — and
  listed only under "Errors". An infrastructure failure is never counted as a retrieval miss
  or a non-compliant answer.

---

## 3. Pre-registered questions

Numbers below are the counts that exist *before* this run, not after. Each has a stated
adequate outcome and a stated problem outcome, decided now.

**Retrieval, hit@k by `question_class`** (17 non-absent DEV questions: `header_field` 5,
`line_item` 5, `cross_page` 3, `near_duplicate` 4; `absent`'s 7 are excluded per §2).
Reported as `format_kn` per class.
- **Adequate:** `header_field` ≥4/5, `line_item` ≥4/5, `cross_page` 3/3, `near_duplicate` 4/4.
  These documents are single-page, born-digital, with anchors that are near-verbatim spans —
  this is an easy retrieval task, and near-perfect performance is the reasonable bar, not a
  stretch goal.
- **Problem:** any class at or below half its total — `header_field` ≤2/5, `line_item` ≤2/5,
  `cross_page` ≤1/3, `near_duplicate` ≤2/4.

**MRR by `question_class`**, plain float per class (not `format_kn`'d — see §2).
- **Adequate:** ≥0.800 for every class (the right chunk is typically at rank 1 or close to it
  within `k=3`).
- **Problem:** <0.500 for any class (on average the right evidence isn't even in the top half
  of what was retrieved).

**Extraction FieldVerdict distribution by field**, `format_kn` over 6 DEV documents, for each
of the 7 header fields.
- **Adequate:** every field ≥5/6 `CORRECT`.
- **Problem:** any field ≤3/6 `CORRECT` (half or worse), **or** any `HALLUCINATED` verdict on
  `payment_terms` or `currency` for po-002 specifically — po-002 is the one DEV document with
  those two fields genuinely null in the source, so a hallucination there is the extraction
  prompt inventing a value that was never printed, the exact failure the prompt's "use null
  when a requested field is absent" instruction exists to prevent.

**The near_duplicate slice — does dense retrieval distinguish 4500123456 from 4500123457.**
4 questions, po-004 vs. po-005. Because the doc_id check makes a wrong-sibling match score as
a miss (§2), this is really asking: of the 4 questions, how many retrieve their *own*
document's chunk rather than the sibling's.
- **Adequate:** 4/4.
- **Problem:** ≤2/4 — half or worse of the near-duplicate pairs confused, which is the
  specific failure mode this class exists to catch, worth restating given the k=3-of-9-chunks
  caveat in §1: this slice is a *weak* test of discrimination by construction, so even a poor
  embedding model has a nontrivial chance of scoring adequately here by not having to work
  hard. A strong result on this slice should not be read as strong evidence either way, for
  the same reason it will still be reported plainly rather than hedged into unreadability — see
  §6.

**The absent slice — does the model return the exact insufficiency string.** 7 questions.
- **Adequate:** 6/7 or 7/7 compliant.
- **Problem:** ≤5/7 compliant (2 or more non-compliant) — this is pure instruction-following
  against an explicitly stated exact phrase, not a hard reasoning task, so more than one miss
  is a real signal, not noise.

---

## 4. Decision thresholds

Written now, in advance, as the numbers that would change what gets built next — not
descriptions of what "seems reasonable" after the fact.

**Investigate hybrid BM25 retrieval** if the near_duplicate slice scores ≤2/4 hit@k. Per your
explicit instruction: this threshold licenses **building BM25 behind the harness and
re-measuring** — nothing more. near_duplicate exists on exactly one document pair, entirely
within DEV (§6), so it cannot license a ship decision on its own regardless of how the number
lands; a BM25 build only becomes a ship decision if it also holds up on TEST, which this run
does not touch.

**Investigate `chunk_size`** if `cross_page` hit@k is ≤1/3, or if `header_field`/`line_item`
hit@k is weak (≤3/5 either) while `cross_page` and `near_duplicate` are both adequate — the
second pattern would suggest a chunk-boundary problem specific to how header fields get split
from their labels, rather than a general retrieval or discrimination problem.

**Revisit the extraction prompt** if any header field scores ≤3/6 `CORRECT`, if any
`HALLUCINATED` verdict appears on po-002's `payment_terms` or `currency`, or if the absent
slice scores ≤5/7 compliant — extraction correctness and insufficiency compliance both depend
on the model obeying an explicit "don't invent, don't fill gaps" instruction, so a failure
pattern spanning both is a prompt-discipline problem, not two unrelated problems.

---

## 5. Exploratory findings rule

Section 3 is the complete list of pre-registered questions. Anything this run's report
surfaces that is not one of those five bullets — a pattern in which specific documents fail,
a correlation between retrieval misses and extraction misses, an unexpected error, a
particular chunk that keeps getting retrieved wrongly, anything — is exploratory. Exploratory
findings get their own clearly separated section in the eventual results write-up, not a
parenthetical or a footnote inside the pre-registered results. The distinction is structural,
not a matter of hedging language: a reader must be able to tell, from section headers alone,
which numbers were predicted in advance and which were noticed afterward.

---

## 6. What this run cannot establish

Longer than is comfortable, on purpose.

- **DEV is 6 documents and 24 questions.** This cannot support a percentage and cannot support
  a significance claim — there is no statistical test in this codebase and none should be
  invented for n this small. A difference of two or three items between two slices, or between
  two runs, is noise, not a trend. Every count in this document and the eventual report is a
  raw `k/n`, never a rate, specifically because a rate at this n implies false precision.

- **`cross_page` and `near_duplicate` are DEV-only by construction, not by sampling
  accident.** `cross_page` exists on exactly one document (po-006, the only multi-page item
  table in the corpus); `near_duplicate` exists on exactly one document pair (po-004/po-005,
  the only near-identical part numbers in the corpus). Both classes are entirely absent from
  TEST and HELD. Any finding on either slice is measured on the same documents this system
  would be tuned against if changes are made in response to that finding — there is no
  held-out data for either class, anywhere in this corpus, at any split. An improvement on
  `cross_page` or `near_duplicate` after a change is evidence of fit to these two specific
  documents, not evidence of generalization. This is the direct reason §4's BM25 threshold is
  written as "investigate," never "ship."

- **`absent` is 7 of DEV's 24 questions — used to be reported as roughly 30%, but that
  phrasing is itself the kind of thing this document is arguing against, so: 7 of 24.** Those
  7 are scored by exact-string compliance, not retrieval quality. A model that is bad at
  retrieval but reliably says "I don't know" when told to can post a deceptively solid-looking
  DEV question count while the 17 retrieval questions underneath are weak — which is exactly
  why §2's report separates the two into different sections rather than blending them into one
  number, and why this run's headline is never going to be a single figure.

- **HELD is 2 documents and 5 questions.** This run does not touch it, but it is worth stating
  what HELD is *for* now, before it is ever used: it can catch a system that is grossly
  broken and nothing finer than that. It is structural provision for a larger corpus later,
  not evidence of anything today. Treating a future HELD result as confirmation of a DEV
  finding, at n=5, would be a mistake worth pre-empting in writing now.

- **Answer correctness is not measured, anywhere in this harness.** `evals/metrics.py` scores
  whether the right evidence was retrieved (`hit_at_k`/MRR) and whether structured extraction
  matched (`FieldVerdict`) and whether the model complied with the insufficiency string
  (`is_insufficiency_response`). Nothing scores whether a *free-text answer* is actually
  correct. A system that retrieves the right chunk and then answers the question wrong — states
  the wrong number while quoting the right source — scores exactly as clean on this harness as
  a system that gets it right. This is a real, structural blind spot in what "the baseline
  number" can claim, not an edge case.

- **The corpus is synthetic and entirely born-digital.** Twelve reportlab-rendered PDFs, no
  OCR, no scans, no faxes, no rotated pages, no handwriting, no multi-column layouts, no
  watermarks. Real ISC purchase orders are substantially scanned documents. Nothing in this
  run — not the retrieval numbers, not the extraction numbers, not the insufficiency
  compliance — speaks to how this pipeline behaves on that population at all. This run
  measures the pipeline on the easiest plausible input distribution it will ever see.

---

## 7. Cost and scope of the run

DEV only — 6 documents, 24 questions. TEST (4 documents, 11 questions) and HELD (2 documents,
5 questions) are not touched by this run or by anything this document authorizes.

Call counts, computed directly from `run_eval`'s structure (ingest once per document, then one
query-embed and one answer call per question), not estimated loosely:

| Call type | Per-unit | Count |
|---|---|---|
| Extraction (`extract_purchase_order`, one `responses.parse` call) | 1 × 6 documents | **6** |
| Chunk embedding (`embed_texts`, one call per document, all its chunks batched) | 1 × 6 documents | **6** |
| Query embedding (`embed_texts`, one call per question) | 1 × 24 questions | **24** |
| Answer (`answer_question`, one `responses.parse` call per question) | 1 × 24 questions | **24** |
| **Total** | | **60** |

Split by endpoint: 30 calls through the Responses API (6 extraction + 24 answer), 30 calls
through the embeddings endpoint (6 chunk-batch + 24 query). Every document in DEV is a single
short page or a handful of short pages (po-006 is the longest at 3 pages); every question is a
short natural-language sentence. Token volume per call is small in absolute terms regardless
of current per-token pricing, which this document does not attempt to quote — the count above
is exact, a dollar figure would not be.

---

*Nothing in this document authorizes running `run_eval`. The next session runs it, after you
have read this and confirmed the three flagged values in §1.*

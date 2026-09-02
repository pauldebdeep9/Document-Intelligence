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
| `k` | **`3` AND `1` — two runs** | proposed — **awaiting your confirmation** |
| `goldset_version` | `1.0.0` | fixed — `data/gold/goldset.json`, confirmed on disk |
| `split` | `DEV` | fixed by this task — TEST and HELD are not touched |
| `git_commit_sha` | *the commit that includes this document's final, amended text* | see note below |

**Model names are proposals, not facts I looked up in this repo.** `.env.example` ships both
`OPENAI_CHAT_MODEL` and `OPENAI_EMBEDDING_MODEL` blank — nothing in this codebase names a
specific model anywhere. `gpt-4.1-mini` is proposed because `isc.llm` calls
`client.responses.parse(..., text_format=...)`, which needs a model supporting the Responses
API's structured-output mode; `text-embedding-3-small` is proposed as OpenAI's standard
small embedding model. I am not confident these are still the current recommended choices as
of your read of this document — confirm or replace both before the run.

**`k` is not one proposed value — it is a design decision, and the decision is two runs.**
The DEV split's pooled candidate set has been measured directly, not assumed: chunking all 6
DEV documents at `chunk_size=1200/overlap=200` produces exactly **9 chunks total** (po-001: 1,
po-002: 1, po-004: 1, po-005: 1, po-006: 3, po-010: 2), and 4 of those 6 documents — including
both near-duplicate documents, po-004 and po-005 — produce exactly one chunk each. At `k=3`
against a 9-chunk pool, retrieval returns a third of every candidate that exists. Concretely,
that means po-004's one chunk and po-005's one chunk are both very likely to land in *any*
top-3, regardless of how good or bad the embedding is at telling `4500123456` from
`4500123457` apart — the near_duplicate slice cannot fail in the way it was built to detect,
because there usually isn't a third document's chunk around to bump either one out. Whatever
number that slice returns at `k=3` is close to uninformative.

So this run is pre-registered as **two runs**, same frozen config, differing only in `k`:
- **`k=3` — primary for `header_field` and `line_item`.** These slices ask "does retrieval
  work at all," and `k=3` (the existing default: `isc.retrieval.top_k_chunks(..., k=3)`,
  `demo.py`'s pipeline) is the reasonable bar for that question — proposed for continuity, not
  freshly invented.
- **`k=1` — primary for `near_duplicate` and `cross_page`.** These slices ask "does retrieval
  pick the single right thing when it matters which one," and only `k=1` forces that choice.
  At `k=1`, `near_duplicate` becomes a genuine forced choice between po-004's chunk and
  po-005's — the actual question that slice exists to answer.

Both runs report all five `question_class` slices (the harness doesn't run partial reports),
so every slice gets a number from both runs — but only one of the two is the *pre-registered,
decision-bearing* number for a given slice; §3 states which for each, and the other number is
reported as context, not evidence.

**`chunk_size`/`overlap` are not proposals — they are determined.** The goldset's anchors were
authored and validated against `chunk_size=1200, overlap=200` (`evals/gold/authoring.py`); that
pairing is recorded in `goldset.chunking` and is what `data/gold/goldset.json` actually
contains on disk (confirmed by reading it directly). A different chunking config is a
different, unvalidated set of anchors, not a variant of the same baseline.

**Frozen means frozen.** Changing `chat_model`, `embedding_model`, `chunk_size`, `overlap`, or
either `k` value after this document is committed produces a *different* run, not a comparable
one. Any of those changing invalidates this document as that run's pre-registration; a new one
must be written first.

**On the commit sha**: `run_eval` captures `git_commit_sha` automatically at execution time
(`evals/runner.py::_current_git_commit_sha`, a plain `git rev-parse HEAD`) — it is not
something this document needs to predict. This document was first committed as `c7933bf`, then
amended (wiring `score_line_items` into `evals/report.py` and adding the two-run `k` plan
above) by the commit that includes this revision — that amending commit is the frozen
reference point from here on, superseding `c7933bf`. Both runs (`k=3` and `k=1`) must execute
at that same commit, with no intervening commits between them; if the two runs'
`RunRecord.git_commit_sha` values differ from each other, or from that commit, the runs
happened against different code than this document reviewed, or than each other, and that is
itself a finding to report, not a detail to paper over.

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

**Line items.** `score_line_items` matches expected line items against actual ones — by
`part_number` when the expected item has one (searched anywhere in the remaining unconsumed
actual items, order-independent), by position when it doesn't — and returns matched/missing/
spurious counts. This was the one place an earlier draft of this document described the code
differently than the code behaved: `score_line_items` was implemented and tested but never
called by `build_report`, so line-item accuracy was silently unmeasured despite five DEV
questions being `line_item` class. That gap is now closed — `build_report` renders a `matched
X/E, missing X/E, spurious X/A` line **per document** (`E` = expected item count, `A` = actual
item count for that document), with no rollup across documents: a missing item on one document
and a spurious item on another must both stay visible as separate findings, not cancel out
into one aggregate. §3 now pre-registers a question against it.

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

Numbers below are the counts that exist *before* either run, not after. Each has a stated
adequate outcome and a stated problem outcome, decided now. Where a question is
retrieval-dependent, one `k` is designated **primary** — the decision-bearing number — per
§1's reasoning; the other `k`'s number for that slice is reported too but is context, not
evidence.

**Retrieval, hit@k by `question_class`** (17 non-absent DEV questions: `header_field` 5,
`line_item` 5, `cross_page` 3, `near_duplicate` 4; `absent`'s 7 are excluded per §2). Reported
as `format_kn` per class, from both runs.

*`header_field`/`line_item`, primary `k=3`:*
- **Adequate:** `header_field` ≥4/5, `line_item` ≥4/5. Single-page, born-digital documents
  with near-verbatim anchors — an easy retrieval task, near-perfect is the reasonable bar.
- **Problem:** either class ≤2/5.

*`cross_page`/`near_duplicate`, primary `k=1`:*
- **Adequate:** `cross_page` 3/3, `near_duplicate` 4/4.
- **Problem:** `cross_page` ≤1/3, `near_duplicate` ≤2/4.

*At `k=3`, `cross_page` and `near_duplicate` are still reported but not decision-bearing* —
see the near_duplicate discussion below for why a `k=3` number on these slices is expected to
look better than the underlying discrimination ability, not why it should be distrusted as
"wrong."

**MRR by `question_class`**, plain float per class (not `format_kn`'d — see §2), from both
runs, same primary-`k` assignment as hit@k above.
- **Adequate:** ≥0.800 for the primary-`k` run, on each class's primary `k`.
- **Problem:** <0.500 for the primary-`k` run, on any class's primary `k`.

**Extraction FieldVerdict distribution by field**, `format_kn` over 6 DEV documents, for each
of the 7 header fields. **`k`-independent**: `run_eval`'s ingest — page extraction, structured
extraction, chunking, chunk-embedding — never calls `top_k_chunks` or touches `k` at all, so
this slice's threshold applies identically to both runs' extraction output.
- **Adequate:** every field ≥5/6 `CORRECT`.
- **Problem:** any field ≤3/6 `CORRECT` (half or worse), **or** any `HALLUCINATED` verdict on
  `payment_terms` or `currency` for po-002 specifically — po-002 is the one DEV document with
  those two fields genuinely null in the source, so a hallucination there is the extraction
  prompt inventing a value that was never printed, the exact failure the prompt's "use null
  when a requested field is absent" instruction exists to prevent.

  Because ingest is re-executed independently in each of the two runs (the harness has no
  mechanism to share it — §7), the two runs' extraction numbers are expected to closely agree
  but are not guaranteed to be identical; a real divergence between them would reflect ordinary
  LLM output variance across repeated calls, not `k`, and would itself be exploratory (§5).

**Line items, matched/missing/spurious per document**, `format_kn`, newly wired into
`build_report` this session (§2). **`k`-independent**, same reasoning as extraction above —
line items come from the same ingest-time `PurchaseOrder` extraction.
- **Adequate:** every DEV document at `matched == expected_count, missing 0, spurious 0` —
  DEV's line items are 1–4 short, clearly delimited rows per document; exact reconciliation is
  the reasonable bar, not partial credit.
- **Problem:** any single document showing `missing ≥1` **or** `spurious ≥1` — per-document,
  not summed (§1 of this session's `report.py` change deliberately does not roll documents up),
  so one bad document does not need to move an aggregate to count as a problem.

**The near_duplicate slice — does dense retrieval distinguish 4500123456 from 4500123457.**
4 questions, po-004 vs. po-005. Because the doc_id check makes a wrong-sibling match score as
a miss (§2), this is really asking: of the 4 questions, how many retrieve their *own*
document's chunk rather than the sibling's. **Primary: `k=1`.**
- **Adequate (`k=1`):** 4/4.
- **Problem (`k=1`):** ≤2/4 — half or worse of the near-duplicate pairs confused, the specific
  failure mode this class exists to catch.
- **`k=3` is reported but not the decision number.** Per §1: with both po-004's and po-005's
  one chunk each very likely both inside any top-3 of a 9-chunk pool, this slice is
  structurally close to unable to fail at `k=3` regardless of embedding quality. A strong
  `k=3` result is expected by construction and is not evidence the model can discriminate the
  two part numbers; a *weak* `k=3` result, on the other hand, would be notable precisely
  because the test is so easy to pass — that asymmetry is itself worth flagging in the eventual
  report as exploratory (§5) if it occurs, even though `k=3` isn't primary here.

**The absent slice — does the model return the exact insufficiency string.** 7 questions.
**Not `k`-independent**, unlike extraction: every question — including `absent` ones — goes
through the same retrieve-then-answer path in `run_eval`, so the retrieved context size given
to the model differs between the two runs even though the correct answer ("insufficient")
shouldn't depend on it. Same thresholds apply to both runs, not because `k` provably doesn't
matter here but because it isn't expected to; a genuine divergence between the two runs on
this specific slice would be a real, notable exploratory finding (§5), since it would suggest
context volume affects this model's instruction-following.
- **Adequate:** 6/7 or 7/7 compliant.
- **Problem:** ≤5/7 compliant (2 or more non-compliant) — this is pure instruction-following
  against an explicitly stated exact phrase, not a hard reasoning task, so more than one miss
  is a real signal, not noise.

---

## 4. Decision thresholds

Written now, in advance, as the numbers that would change what gets built next — not
descriptions of what "seems reasonable" after the fact.

**Investigate hybrid BM25 retrieval** if the near_duplicate slice scores ≤2/4 hit@k **at its
primary `k=1`** — not at `k=3`, whose near_duplicate number is expected to look adequate
almost regardless of embedding quality (§1, §3). Per your explicit instruction: this threshold
licenses **building BM25 behind the harness and re-measuring** — nothing more. near_duplicate
exists on exactly one document pair, entirely within DEV (§6), so it cannot license a ship
decision on its own regardless of how the number lands; a BM25 build only becomes a ship
decision if it also holds up on TEST, which neither run touches.

**Investigate `chunk_size`** if `cross_page` hit@k is ≤1/3 **at its primary `k=1`**, or if —
within the `k=3` run specifically, the primary run for `header_field`/`line_item` — those two
classes' hit@k is weak (≤3/5 either) while that same `k=3` run's `cross_page`/`near_duplicate`
numbers look fine. The second pattern would suggest a chunk-boundary problem specific to how
header fields get split from their labels, rather than a general retrieval or discrimination
problem — comparing within one run keeps the comparison apples-to-apples rather than crossing
the two `k` values.

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

- **The DEV pooled candidate set is 9 chunks.** This is a retrieval problem small enough that
  the task is closer to *document selection* than *passage retrieval* — picking the right one
  of 6 (or, per question, effectively fewer) documents' single chunk, not ranking a genuine
  passage-level candidate pool. Nothing measured here predicts behavior on a corpus of
  thousands of chunks, where both the failure modes (near-duplicate passages *within* a
  document, not just between two documents; genuine ranking degradation; recall dropping
  before precision does) and the latency profile are different in kind, not just degree.

- **Most DEV documents produce exactly one chunk, so chunk-level retrieval quality is largely
  untested.** Of the 6 DEV documents, 4 (po-001, po-002, po-004, po-005) chunk to exactly one
  piece each — there is no *selection among a document's own chunks* happening for those at
  all, only selection among documents. `cross_page` (po-006, 3 chunks) and the
  long-description document (po-010, 2 chunks) are the only two places in DEV where multi-chunk
  selection within a single document happens at all — two documents, out of six, are carrying
  the entire weight of testing whether chunk-level ranking works, as distinct from
  document-level selection.

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

DEV only — 6 documents, 24 questions, **times two runs** (`k=3`, `k=1`). TEST (4 documents,
11 questions) and HELD (2 documents, 5 questions) are not touched by either run or by anything
this document authorizes.

Call counts, computed directly from `run_eval`'s structure (ingest once per document per
`run_eval` invocation, then one query-embed and one answer call per question), not estimated
loosely. `run_eval` has no mechanism to share ingest across two invocations, so running it
once for `k=3` and once for `k=1` repeats ingest in full — the extraction and chunk-embedding
calls are identical work done twice for a reason that has nothing to do with `k` (§3's
extraction/line-items sections are `k`-independent, precisely because ingest is):

| Call type | Per-unit | Per run | Both runs |
|---|---|---|---|
| Extraction (`extract_purchase_order`) | 1 × 6 documents | 6 | **12** |
| Chunk embedding (`embed_texts`, chunks batched per document) | 1 × 6 documents | 6 | **12** |
| Query embedding (`embed_texts`, one call per question) | 1 × 24 questions | 24 | **48** |
| Answer (`answer_question`) | 1 × 24 questions | 24 | **48** |
| **Total** | | **60** | **120** |

Of the 120, **24 (the extraction + chunk-embedding calls) are redundant** — genuinely
duplicated work, not two different measurements, a real limitation of `run_eval` as it exists
today, not something this session's `report.py` change fixes. The remaining 96 (query-embed +
answer, 48 per run) are the actual `k`-dependent work that has to happen twice, once per `k`,
since `top_k_chunks` and therefore `answer_question`'s input genuinely differ between the two
runs. Split by endpoint across both runs: 60 calls through the Responses API (12 extraction +
48 answer), 60 through the embeddings endpoint (12 chunk-batch + 48 query). Every document in
DEV is a single short page or a handful of short pages (po-006 is the longest at 3 pages);
every question is a short natural-language sentence. Token volume per call is small in
absolute terms regardless of current per-token pricing, which this document does not attempt
to quote — the counts above are exact, a dollar figure would not be.

---

*Nothing in this document authorizes running `run_eval`. The next session runs it — twice,
once per `k` — after you have read this and confirmed the flagged values in §1: the two model
names, and the two-run `k=3`/`k=1` plan itself.*

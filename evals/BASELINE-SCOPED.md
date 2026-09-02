# DEV baseline under scoped retrieval, vs. BASELINE.md

Executed against gold set version `1.1.0` (post near_duplicate q1 rewrite,
`GOLDSET-CHANGELOG.md`), DEV split only. `chunk_size=1200`, `overlap=200`,
`chat_model=gpt-5.6-luna`, `embedding_model=text-embedding-3-small`. All four runs executed at
git sha `ce07167e8d9abf838fb8bb08816f652326d6ae00` (`ce07167`), with
`expected_goldset_version="1.1.0"` passed explicitly on every call. TEST and HELD were not
touched.

| Run | scoping | k | `run_id` |
|---|---|---|---|
| 1 | scoped | 3 | `3354b7a6-1a5d-4671-9f36-a61f85332bb6` |
| 2 | scoped | 1 | `e3cb41df-6748-4bec-9f3a-35f887cd544a` |
| 3 | pooled (control) | 3 | `75c1a0c6-5768-48b3-94d2-aff559216fbe` |
| 4 | pooled (control) | 1 | `c846d1f5-aa71-471c-a979-ba20fb67a91b` |

**Harness-health check:** none of the four runs returned all-empty retrieval or all-errored
items; every run recorded 0 errored items. This is a quality result, not a harness fault.

---

## Full report text

### Run 1 — scoped, k=3 (`3354b7a6-1a5d-4671-9f36-a61f85332bb6`, sha `ce07167`)

```
Run 3354b7a6-1a5d-4671-9f36-a61f85332bb6 (dev)
7 absent questions excluded from retrieval slices (scored separately by string compliance, not retrieval).

=== Retrieval ===
header_field: hit@k 5/5, MRR 1.000
line_item: hit@k 5/5, MRR 1.000
cross_page: hit@k 3/3, MRR 1.000
near_duplicate: hit@k 4/4, MRR 1.000

=== Extraction ===
po_number: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
po_date: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
supplier_name: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
ship_to_site: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
payment_terms: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
currency: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
total_amount: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6

=== Line items ===
po-001: matched 3/3, missing 0/3, spurious 0/3
po-002: matched 2/2, missing 0/2, spurious 0/2
po-004: matched 1/1, missing 0/1, spurious 0/1
po-005: matched 1/1, missing 0/1, spurious 0/1
po-006: matched 4/4, missing 0/4, spurious 0/4
po-010: matched 1/1, missing 0/1, spurious 0/1

=== Insufficiency compliance (absent questions) ===
compliant: 7/7

=== Errors ===
0 item(s) errored.
```

### Run 2 — scoped, k=1 (`e3cb41df-6748-4bec-9f3a-35f887cd544a`, sha `ce07167`)

```
Run e3cb41df-6748-4bec-9f3a-35f887cd544a (dev)
7 absent questions excluded from retrieval slices (scored separately by string compliance, not retrieval).

=== Retrieval ===
header_field: hit@k 5/5, MRR 1.000
line_item: hit@k 5/5, MRR 1.000
cross_page: hit@k 3/3, MRR 1.000
near_duplicate: hit@k 4/4, MRR 1.000

=== Extraction ===
po_number: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
po_date: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
supplier_name: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
ship_to_site: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
payment_terms: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
currency: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
total_amount: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6

=== Line items ===
po-001: matched 3/3, missing 0/3, spurious 0/3
po-002: matched 2/2, missing 0/2, spurious 0/2
po-004: matched 1/1, missing 0/1, spurious 0/1
po-005: matched 1/1, missing 0/1, spurious 0/1
po-006: matched 4/4, missing 0/4, spurious 0/4
po-010: matched 1/1, missing 0/1, spurious 0/1

=== Insufficiency compliance (absent questions) ===
compliant: 7/7

=== Errors ===
0 item(s) errored.
```

### Run 3 — pooled control, k=3 (`75c1a0c6-5768-48b3-94d2-aff559216fbe`, sha `ce07167`)

```
Run 75c1a0c6-5768-48b3-94d2-aff559216fbe (dev)
7 absent questions excluded from retrieval slices (scored separately by string compliance, not retrieval).

=== Retrieval ===
header_field: hit@k 3/5, MRR 0.467
line_item: hit@k 5/5, MRR 0.767
cross_page: hit@k 3/3, MRR 0.778
near_duplicate: hit@k 4/4, MRR 0.708

=== Extraction ===
po_number: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
po_date: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
supplier_name: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
ship_to_site: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
payment_terms: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
currency: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
total_amount: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6

=== Line items ===
po-001: matched 3/3, missing 0/3, spurious 0/3
po-002: matched 2/2, missing 0/2, spurious 0/2
po-004: matched 1/1, missing 0/1, spurious 0/1
po-005: matched 1/1, missing 0/1, spurious 0/1
po-006: matched 4/4, missing 0/4, spurious 0/4
po-010: matched 1/1, missing 0/1, spurious 0/1

=== Insufficiency compliance (absent questions) ===
compliant: 5/7
Non-compliant items:
  po-002-q3 (doc po-002): answer = 'The payment terms are Net 30.'
  po-002-q4 (doc po-002): answer = 'The currency is USD.'

=== Errors ===
0 item(s) errored.
```

### Run 4 — pooled control, k=1 (`c846d1f5-aa71-471c-a979-ba20fb67a91b`, sha `ce07167`)

```
Run c846d1f5-aa71-471c-a979-ba20fb67a91b (dev)
7 absent questions excluded from retrieval slices (scored separately by string compliance, not retrieval).

=== Retrieval ===
header_field: hit@k 2/5, MRR 0.400
line_item: hit@k 3/5, MRR 0.600
cross_page: hit@k 2/3, MRR 0.667
near_duplicate: hit@k 2/4, MRR 0.500

=== Extraction ===
po_number: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
po_date: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
supplier_name: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
ship_to_site: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
payment_terms: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
currency: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6
total_amount: correct 6/6, wrong 0/6, missed 0/6, hallucinated 0/6

=== Line items ===
po-001: matched 3/3, missing 0/3, spurious 0/3
po-002: matched 2/2, missing 0/2, spurious 0/2
po-004: matched 1/1, missing 0/1, spurious 0/1
po-005: matched 1/1, missing 0/1, spurious 0/1
po-006: matched 4/4, missing 0/4, spurious 0/4
po-010: matched 1/1, missing 0/1, spurious 0/1

=== Insufficiency compliance (absent questions) ===
compliant: 5/7
Non-compliant items:
  po-002-q3 (doc po-002): answer = 'The payment terms are Net 30.'
  po-002-q4 (doc po-002): answer = 'The currency for this purchase order is USD.'

=== Errors ===
0 item(s) errored.
```

---

## Comparison: scoped vs. pooled, on gold set 1.1.0, slice by slice

**hit@k**

| Class | scoped k=3 | scoped k=1 | pooled k=3 | pooled k=1 |
|---|---|---|---|---|
| header_field | 5/5 | 5/5 | 3/5 | 2/5 |
| line_item | 5/5 | 5/5 | 5/5 | 3/5 |
| cross_page | 3/3 | 3/3 | 3/3 | 2/3 |
| near_duplicate | 4/4 | 4/4 | 4/4 | 2/4 |

**MRR**

| Class | scoped k=3 | scoped k=1 | pooled k=3 | pooled k=1 |
|---|---|---|---|---|
| header_field | 1.000 | 1.000 | 0.467 | 0.400 |
| line_item | 1.000 | 1.000 | 0.767 | 0.600 |
| cross_page | 1.000 | 1.000 | 0.778 | 0.667 |
| near_duplicate | 1.000 | 1.000 | 0.708 | 0.500 |

**Absent compliance**

| | scoped k=3 | scoped k=1 | pooled k=3 | pooled k=1 |
|---|---|---|---|---|
| compliant | 7/7 | 7/7 | 5/7 | 5/7 |

**Extraction and line items**: identical (adequate, no exceptions) across all four runs — see
the non-determinism note below.

Scoped retrieval is perfect (hit@k and MRR both maximal) on every slice, at both k values. On
this DEV split, restricting each question's candidate set to its own document eliminated every
retrieval miss this harness could detect.

## The 1.1.0 pooled runs vs. `BASELINE.md`'s 1.0.0 pooled runs — attributing the difference

`BASELINE.md`'s two runs are pooled retrieval against gold set 1.0.0. Comparing them directly
to this session's pooled runs (3 and 4 above), slice by slice:

| Class / metric | 1.0.0 pooled k=3 (`BASELINE.md`) | 1.1.0 pooled k=3 (this session) | 1.0.0 pooled k=1 | 1.1.0 pooled k=1 |
|---|---|---|---|---|
| header_field hit@k | 3/5 | 3/5 | 2/5 | 2/5 |
| header_field MRR | 0.467 | 0.467 | 0.400 | 0.400 |
| line_item hit@k | 5/5 | 5/5 | 3/5 | 3/5 |
| line_item MRR | 0.767 | 0.767 | 0.600 | 0.600 |
| cross_page hit@k | 3/3 | 3/3 | 2/3 | 2/3 |
| cross_page MRR | 0.778 | 0.778 | 0.667 | 0.667 |
| near_duplicate hit@k | 4/4 | 4/4 | **1/4** | **2/4** |
| near_duplicate MRR | 0.583 | **0.708** | **0.250** | **0.500** |
| absent compliant | 5/7 (same 2 items) | 5/7 (same 2 items) | 5/7 (same 2 items) | 5/7 (same 2 items) |

Every slice except `near_duplicate` is byte-identical between the 1.0.0 and 1.1.0 pooled runs,
at both k values — header_field, line_item, cross_page, and absent compliance did not move at
all. **Only `near_duplicate` differs**, and only in the metric the q1 rewrite was expected to
touch: hit@k moved from 1/4 to 2/4 at k=1 (unchanged, 4/4, at k=3 — consistent with §1/§3's own
prediction that k=3 against this pool is close to unable to fail regardless of question
wording), and MRR moved at both k values (0.583→0.708 at k=3, 0.250→0.500 at k=1). This is
attributed to the gold set change, not to scoping: runs 3 and 4 are pooled, the same retrieval
mode `BASELINE.md` used, so scoping plays no role in either side of this comparison. The only
thing that changed between the two gold set versions is `po-004-q1`/`po-005-q1`'s question
text (`GOLDSET-CHANGELOG.md`) — before the rewrite the two questions were byte-identical, so
their retrieval was identical by construction regardless of embedding quality; after the
rewrite each question's embedding differs (it now names its own `po_number`), so the two
questions can rank differently against the pool, which is what moved.

## po-002-q3 / po-002-q4 under scoping — the direct test of FINDING-001

**Both are compliant under scoping, at both k=3 and k=1.** `FINDING-001` established that
these two absent questions were answered "Net 30" / "USD" because pooled retrieval handed the
model a sibling document's chunk (po-001, po-004, or po-010) that genuinely states those
values, even though po-002's own chunk contains neither. Under scoping, each question's
candidate set is restricted to po-002's own single chunk — which contains no payment or
currency string — and the model correctly returned the exact insufficiency string for both
questions, at both k values (scoped absent compliance is 7/7, including these two). The pooled
control runs in this same session reproduce the original non-compliant answers exactly (same
two items, same failure), confirming the contamination mechanism was still present in the
control and specifically absent once scoping removed the cross-document candidates.

## What k no longer tests, under scoping

Under `chunk_size=1200/overlap=200`, DEV's per-document chunk counts are po-001: 1, po-002: 1,
po-004: 1, po-005: 1, po-006: 3, po-010: 2 (established in earlier sessions, unaffected by
anything since). Scoped to its own document, a question on any of the four 1-chunk documents
returns the identical single chunk regardless of k — there is nothing to rank. Only po-006
(cross_page) and po-010 (line_item) have more than one chunk in their own scoped candidate set,
so those are the only two documents where scoped k=3 vs. k=1 could structurally differ at all.

**In this run, they didn't.** Scoped k=3 and scoped k=1's reports are identical in every
metric, including on po-006 and po-010 — the top-ranked chunk within each of those documents'
own 2–3 chunks was already the correct one, so narrowing from k=3 to k=1 dropped nothing. This
is not evidence that k is meaningless under scoping in general; it means this run did not
happen to exercise a case where narrowing k within a multi-chunk document changed the outcome.
The pre-registration's original reasoning for treating k=3 vs. k=1 as separate, meaningful
measurements assumed a shared, pooled candidate pool where k determines how many *other
documents'* chunks compete for a slot — under scoping that competition doesn't exist, so a
scoped k=3-vs-k=1 comparison does not carry that meaning any more. Presenting scoped k=3 and
k=1 as two independent measurements, the way `PREREGISTRATION.md` did for pooled retrieval,
would overstate what this comparison can show.

## Errors

0 items errored across all four runs (24 questions × 4 = 96 item-runs, 0 errors, 96
successes).

## Extraction and line-item non-determinism

Every one of the 7 header fields scored 6/6 CORRECT, and every document's line-item
reconciliation showed `matched == expected, missing 0, spurious 0`, identically across all
four runs — expected, since ingest (extraction, chunking, chunk-embedding) does not depend on
`scoping` or `k` and ran independently four times. **No divergence was observed** across the
four independent ingest passes. This is one set of four paired observations on six short
documents, not evidence that extraction is deterministic, and is not described as
deterministic anywhere in this document.

---

## What this still does not establish

Restated from `PREREGISTRATION.md` §6, against these actual numbers:

- **DEV is 6 documents and 24 questions.** Every number above is a raw `k/n`. Scoped
  retrieval's 7/7, 5/5, 4/4 etc. are perfect scores on a 24-question split — they say scoping
  eliminated every miss this harness could detect on *these* 24 questions, not that scoping
  guarantees perfect retrieval in general.

- **The DEV pooled candidate set (for the pooled control) is still 9 chunks.** The pooled
  numbers above inherit every limitation `PREREGISTRATION.md` already stated about that pool
  size — this remains a document-selection-scale problem, not a passage-retrieval-scale one,
  for the pooled control specifically.

- **Under scoping, most DEV documents still produce exactly one chunk.** As stated above, only
  po-006 and po-010 test any within-document chunk selection at all under scoping; the other
  four documents' "retrieval" is really "return the only candidate that exists." Scoped
  retrieval's perfect score is real, but it was an easy test on 4 of 6 documents by
  construction, not evidence about scoped ranking quality under real multi-chunk competition.

- **cross_page and near_duplicate are still DEV-only by construction**, with no held-out
  equivalent anywhere in this corpus. The near_duplicate improvement documented above (both
  under scoping and from the goldset rewrite) is measured on the same one document pair this
  system would be tuned against if further changes were made — it is not evidence of
  generalization.

- **absent is still 7 of DEV's 24 questions.** Scoped compliance reached 7/7 here, but this
  remains a small slice measuring exact-string compliance, not retrieval quality broadly.

- **HELD (2 documents, 5 questions) was not touched by any of the four runs.** Nothing here is
  validated or contradicted by HELD.

- **Answer correctness is still not measured, anywhere in this harness.** Scoped retrieval
  fixed *contamination* (wrong-document evidence reaching the model) — it says nothing about
  whether a free-text answer built from the correct evidence is itself factually right.

- **The corpus is still synthetic and entirely born-digital.** Every result above, including
  scoped retrieval's perfect score, was produced on 12 reportlab-rendered PDFs with no OCR,
  scans, or layout noise. Nothing here speaks to that population.

---

*Config confirmed before any spend: git sha `ce07167` matched, `chat_model`/`embedding_model`
matched `.env`, goldset version `1.1.0` confirmed on every one of the four runs via
`expected_goldset_version="1.1.0"`. No tuning applied in response to results. No TEST/HELD
access. `evals/runner.py`, `evals/report.py`, and all frozen documents (`PREREGISTRATION.md`,
`BASELINE.md`, `FINDING-001.md`, `FINDING-002.md`, `GOLDSET-CHANGELOG.md`) are unedited.*

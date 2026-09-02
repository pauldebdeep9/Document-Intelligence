# DEV baseline: results against the pre-registered thresholds

Executed against `evals/PREREGISTRATION.md` (frozen, commit `32927a3`). Two independent
`run_eval` invocations, DEV split only, `chunk_size=1200`, `overlap=200`. TEST and HELD were
not touched.

**Config confirmation (before either run):**

| Field | `.env` value | `PREREGISTRATION.md` §1 proposal | Result |
|---|---|---|---|
| `chat_model` | `gpt-5.6-luna` | `gpt-4.1-mini` | **differed** — `.env`'s value confirmed by the user as the actual choice, overriding the doc's placeholder proposal |
| `embedding_model` | `text-embedding-3-small` | `text-embedding-3-small` | matched |

Git sha at both runs: `32927a35397ae086054f0fd3d5c3cef73ef6a971` (`32927a3`), confirmed as
the commit that last touched `evals/PREREGISTRATION.md` and matching between the two runs.

| Run | `run_id` | `k` | `git_commit_sha` |
|---|---|---|---|
| Primary for header_field/line_item | `beda2db6-b925-4201-8c27-44d18c08e857` | 3 | `32927a35397ae086054f0fd3d5c3cef73ef6a971` |
| Primary for cross_page/near_duplicate | `d78f8dee-627f-4c0f-8cf0-6f27964dfc1d` | 1 | `32927a35397ae086054f0fd3d5c3cef73ef6a971` |

**Harness-health check (§ task step 5):** not every anchor missed, not every extraction field
came back null, and 0 items errored in either run — this is a quality result, not a harness or
config fault. Proceeding to report it as such.

---

## Full report text

### k=3 run (`beda2db6-b925-4201-8c27-44d18c08e857`)

```
Run beda2db6-b925-4201-8c27-44d18c08e857 (dev)
7 absent questions excluded from retrieval slices (scored separately by string compliance, not retrieval).

=== Retrieval ===
header_field: hit@k 3/5, MRR 0.467
line_item: hit@k 5/5, MRR 0.767
cross_page: hit@k 3/3, MRR 0.778
near_duplicate: hit@k 4/4, MRR 0.583

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

### k=1 run (`d78f8dee-627f-4c0f-8cf0-6f27964dfc1d`)

```
Run d78f8dee-627f-4c0f-8cf0-6f27964dfc1d (dev)
7 absent questions excluded from retrieval slices (scored separately by string compliance, not retrieval).

=== Retrieval ===
header_field: hit@k 2/5, MRR 0.400
line_item: hit@k 3/5, MRR 0.600
cross_page: hit@k 2/3, MRR 0.667
near_duplicate: hit@k 1/4, MRR 0.250

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

**Extraction non-determinism (task reporting rule):** the k=3 run is authoritative for
extraction and line items. Comparing the two runs' independently-executed ingest: every one of
the 7 header fields scored identically (6/6 CORRECT) in both runs, and every document's
line-item reconciliation was identical in both runs (matched = expected, 0 missing, 0
spurious, everywhere). **No divergence was observed.** The only difference between the two
runs anywhere outside the retrieval-dependent slices is cosmetic: po-002-q4's non-compliant
answer text differs in wording between runs ("The currency is USD." vs. "The currency for this
purchase order is USD.") without changing its (non-)compliance verdict in either run — ordinary
free-text sampling variance, not a scored divergence.

---

## Pre-registered findings (§3 of PREREGISTRATION.md)

Structurally separate from Exploratory findings below, per §5.

### Retrieval — hit@k by question_class

| Class | Primary `k` | hit@k (primary) | Adequate | Problem | Verdict | hit@k (other `k`, context only) |
|---|---|---|---|---|---|---|
| header_field | 3 | 3/5 | ≥4/5 | ≤2/5 | **neither** (falls between the stated bars) | k=1: 2/5 |
| line_item | 3 | 5/5 | ≥4/5 | ≤2/5 | **adequate** | k=1: 3/5 |
| cross_page | 1 | 2/3 | 3/3 | ≤1/3 | **neither** (falls between the stated bars) | k=3: 3/3 |
| near_duplicate | 1 | 1/4 | 4/4 | ≤2/4 | **problem** | k=3: 4/4 |

The near_duplicate k=3 number (4/4) landing exactly at the "adequate" bar while the same
slice's k=1 number (1/4) lands at "problem" is exactly the inflation pattern §1/§3 predicted
before either run: at k=3 against a 9-chunk pool, both po-004's and po-005's one chunk each
are very likely both retrieved regardless of embedding quality. That prediction held.

### Retrieval — MRR by question_class (primary `k` only; not `format_kn`'d, per §2)

| Class | Primary `k` | MRR (primary) | Adequate | Problem | Verdict |
|---|---|---|---|---|---|
| header_field | 3 | 0.467 | ≥0.800 | <0.500 | **problem** |
| line_item | 3 | 0.767 | ≥0.800 | <0.500 | neither |
| cross_page | 1 | 0.667 | ≥0.800 | <0.500 | neither |
| near_duplicate | 1 | 0.250 | ≥0.800 | <0.500 | **problem** |

### Extraction — FieldVerdict distribution (k-independent)

Every one of the 7 header fields: **6/6 CORRECT, 0/6 WRONG, 0/6 MISSED, 0/6 HALLUCINATED**, in
both runs. Adequate bar (≥5/6 CORRECT per field): **met by every field.** Problem bar (≤3/6):
not met by any field. No HALLUCINATED verdict on po-002's `payment_terms` or `currency` (both
scored CORRECT — genuinely null in gold, genuinely null in extraction). **Verdict: adequate,
across the board.**

### Line items (k-independent)

Every DEV document: `matched == expected_count`, `missing 0`, `spurious 0`, in both runs.
**Verdict: adequate, across the board** — the clean-reconciliation bar was met on all 6
documents, with no per-document exception.

### near_duplicate slice — does dense retrieval distinguish 4500123456 from 4500123457

Primary `k=1`: **1/4** hit@k. Adequate is 4/4; problem is ≤2/4. **1/4 is a problem result** —
half or worse of the near-duplicate pairs confused, the specific failure mode this class
exists to catch. (k=3's 4/4 is reported per §3 but is explicitly not the decision-bearing
number for this slice.)

### absent slice — exact insufficiency-string compliance

**5/7** compliant in both runs, same two non-compliant question_ids in both
(`po-002-q3`, `po-002-q4`). Adequate is 6/7 or 7/7; problem is ≤5/7. **5/7 is a problem
result.** Both non-compliant items are on the same document (po-002) and the same two fields
(`payment_terms`, `currency`) — the two fields po-002 was specifically built to leave null.

---

## Decision thresholds (§4 of PREREGISTRATION.md) — evaluated, not acted on

Per the task's explicit reporting rule, none of the following were acted on this session — no
prompt change, no chunk_size change, no BM25 work. Only whether each pre-written trigger
condition fired is reported.

- **Investigate hybrid BM25 retrieval** — trigger is near_duplicate ≤2/4 hit@k at primary
  `k=1`. Actual: 1/4. **Triggered.** Per §4, this licenses building BM25 behind the harness
  and re-measuring, nothing more — near_duplicate is DEV-only by construction (§6), so this
  cannot license a ship decision regardless of how a re-measurement lands.

- **Investigate `chunk_size`** — trigger is either cross_page ≤1/3 at primary `k=1` (actual:
  2/3, not triggered by this path), or — within the k=3 run specifically — header_field/
  line_item hit@k ≤3/5 while that same run's cross_page/near_duplicate look fine. Actual, all
  within the k=3 run: header_field 3/5 (≤3/5), line_item 5/5 (not ≤3/5), cross_page 3/3,
  near_duplicate 4/4. header_field alone satisfies this pattern. **Triggered**, via
  header_field specifically. Per §4, this pattern would suggest a chunk-boundary problem
  specific to how header fields split from their labels, not a general retrieval or
  discrimination problem.

- **Revisit the extraction prompt** — trigger is any header field ≤3/6 CORRECT (not
  triggered — every field is 6/6), a HALLUCINATED verdict on po-002's `payment_terms`/
  `currency` (not triggered — both CORRECT), or absent ≤5/7 compliant (actual: 5/7).
  **Triggered**, via the absent-slice condition only.

All three pre-written "investigate"/"revisit" thresholds fired. None of the three extraction
header fields nor the line-items slice showed any weakness on their own — the extraction
prompt trigger fired solely through the absent-slice (insufficiency-compliance) path, not
through the structured-extraction path.

---

## Exploratory findings

Not pre-registered. Kept in this separate section per §5. Grounded in per-item detail pulled
directly from both saved `RunRecord`s (`runs/beda2db6-....json`, `runs/d78f8dee-....json`),
re-scored offline with the same `evals.metrics.hit_at_k`/`anchors_satisfied` functions
`build_report` uses — not a re-run, no new API calls.

**The header_field weakness at k=3 is two specific questions, not five weak-ish ones, and
both fail at k=1 too.** header_field's k=3 misses are exactly `po-006-q5` ("What are the
payment terms on this purchase order?", gold doc po-006) and `po-010-q2` ("What is the
ship-to site on this purchase order?", gold doc po-010). In neither case is the correct
document's own chunk present anywhere in the top-3 retrieved (po-006-q5 retrieves
`[po-001, po-010, po-004]`; po-010-q2 retrieves `[po-005, po-004, po-002]`) — this is not a
near-miss ranking problem, the right document's chunk isn't in the candidate window at all.
Both questions also miss at k=1 (top-1 is `po-001` and `po-005` respectively). Since these are
the same two questions failing regardless of `k`, this looks like a `k`-independent embedding
quality issue specific to these two questions, not a boundary effect of a small `k`. One
observation, not yet a pattern with n this small: both are asking about a field
(`payment_terms`, `ship_to_site`) using generic phrasing that doesn't reference the document's
own identifying details (PO number, supplier) — worth watching if it recurs on TEST, not
something this DEV-only run can generalize from.

**near_duplicate's three k=1 misses are not uniformly "confused with its sibling."** Of the
3 misses, only `po-004-q1` retrieves the actual sibling document (`po-005`) — the literal
confusion the class was built to detect. The other two (`po-004-q2`, `po-005-q2`) both
retrieve `po-001`, an unrelated document, not each other's sibling. This matters for the §4
BM25-investigate trigger: BM25's lexical matching would plausibly resolve the one genuine
sibling-confusion case (the differing final digit, `...456` vs. `...457`, is exactly what
token-level matching is good at), but the other two misses look like a broader embedding
weakness unrelated to near-duplicate text at all, and there's no particular reason to expect
BM25 to fix those.

**Two of cross_page's three questions succeed at k=1; the miss is the compound
first/last/total question.** `po-006-q1` and `po-006-q2` (each asking about one or two
specific part numbers) both hit at k=1. Only `po-006-q3` — "What are the part numbers of the
first and last line items ... and what is the total amount?", spanning page 1 and page 2 plus
the total — misses, retrieving `po-001` instead of any of po-006's own 3 chunks. This is
consistent with (not proof of) the idea that compound multi-fact queries embed less
predictably than single-fact ones, on a document that already has the most chunks (3) of any
DEV document.

**k=3 vs. k=1 inflation was not unique to near_duplicate.** §1/§3 predicted this effect
specifically for near_duplicate. The same direction of effect (more retrieval slots → higher
hit@k) also shows up on header_field (3/5 → 2/5) and line_item (5/5 → 3/5) between the two
runs — expected in general for hit@k as `k` shrinks, but worth naming explicitly since it's
visible directly in this data: the extra k=1 misses on those two classes
(`po-002-q1` for header_field; `po-001-q3`, `po-010-q1` for line_item) are questions that
succeeded at k=3, i.e. genuinely `k`-boundary-sensitive rather than structural failures like
the two header_field questions above.

**Extraction and line items showed zero divergence between the two independently-executed
ingest passes.** Already stated under "Extraction non-determinism" above; noted again here
because it's a positive confirmation of §2's k-independence claim for ingest, not a new
finding — worth recording as evidence the claim held, not just asserting it held.

---

## What this run does not establish (§6, restated against the actual numbers)

- **DEV is 6 documents and 24 questions — this document has reported every number as `k/n`,
  never a rate**, including the two problem-threshold slices above (near_duplicate 1/4, absent
  5/7). A difference of one or two items on either slice would move the raw count noticeably;
  neither number should be read as more precise than "n is small."

- **The DEV pooled candidate set is 9 chunks.** Both header_field misses above illustrate this
  directly: at k=3, the correct document's one chunk was entirely absent from a 3-item window
  drawn from a 9-chunk pool — a document-selection failure, not a passage-ranking one. Nothing
  here predicts behavior once the pool is in the thousands.

- **Most DEV documents produce exactly one chunk.** All 3 header_field/near_duplicate misses
  documented above involve single-chunk documents (po-006 is the exception, with 3 chunks, and
  its one cross_page miss is the compound query, not a chunk-selection failure within po-006).
  This run mostly tested document selection, as anticipated, not intra-document chunk ranking.

- **cross_page and near_duplicate are DEV-only by construction.** The near_duplicate problem
  result (1/4 at k=1) and the §4 BM25-investigate trigger it fires both apply to exactly one
  document pair, with no held-out equivalent anywhere in this corpus. This result cannot be
  checked against TEST or HELD data because no such data exists for this class.

- **absent is 7 of DEV's 24 questions.** The 5/7 problem result is concentrated on a single
  document (po-002) and its two genuinely-null fields — it does not by itself say anything
  about the other 17 retrieval questions' quality, which is exactly why this report kept the
  two in separate sections rather than blending them.

- **HELD (2 documents, 5 questions) was not touched by either run**, per the task's explicit
  instruction. Nothing in this document should be read as validated or contradicted by HELD.

- **Answer correctness was not measured.** The insufficiency-compliance failures on po-002 are
  scored as string non-compliance, not as "the model gave a wrong answer" — both non-compliant
  answers (`"Net 30"`, `"USD"`) happen to be plausible-sounding values for the field asked
  about, but whether they are factually wrong was never checked by anything in this harness,
  only whether the exact required refusal string was returned.

- **The corpus is synthetic and entirely born-digital.** Every result in this document —
  including the two problem-threshold findings — was produced on 12 reportlab-rendered PDFs
  with no OCR, scans, or layout noise. Nothing here speaks to that population.

---

*Config confirmed against `PREREGISTRATION.md` §1 (with the noted `chat_model` override),
both runs executed at `git_commit_sha 32927a3`, no tuning applied in response to results, no
TEST/HELD access.*

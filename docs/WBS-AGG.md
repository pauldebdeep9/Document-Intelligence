# AG Work Breakdown — Aggregate questions from extracted records

Working document for `isc-docint`, feature #1 of the post-P1 quality work:
answering total-spend and part-price questions from extracted `PurchaseOrder`
records instead of chunks. Design and rationale live in
`docs/adr/0011-aggregate-questions-from-records.md`. Drop this at
`docs/WBS-AGG.md`.

**Target:** 12 items, 3–7 prompts each (~57 prompts), ~1,450 LOC source +
~1,000 LOC tests. **Milestone:** AG-11 — the live eval that decides whether
`aggregate.enabled` flips to `true`. AG-12 is hardening for the Azure landing.

---

## Current state

| Layer | State |
|---|---|
| P1 pipeline (ingest → answer), eval harness, `outcomes.jsonl` (schema v1), `isc eval-diff` | complete |
| P1-09 reading on the target slice | total spend 0/4 (`q_cd_01`–`04`), ambiguous 0/4 (`q_am_01`–`04`), `cross_document` recall@8 0.875 |
| AG-01..AG-07 | reference implementation in `aggregate-records-path.patch` (one commit, 22 files) — sandbox-verified, not yet run on `Sai2608` |
| Live planner eval, route in eval outcomes, complete part-price gold, records carrying their own ACL | not started (AG-08..AG-12) |
| Dev/test env | `Sai2608` — the repo's own env; its editable install was repointed at this checkout during AG-00 (it pointed at the old `SaiBaba/BabaKoreDao` path) |
| Branch | `feat/ag-records-path` off `main`; merged after AG-11 |

**Baseline before AG-01:** AG-00 preflight green — `Sai2608` is Python 3.11 and
resolves `isc` to this checkout, the corpus is present and gold unchanged,
`make test` and `pytest -m acl` baselines recorded, the reference patch applies
cleanly to `main`, and this document committed as the branch's first commit.

---

## Two ways to run AG-01..AG-07

- **Build mode.** Run each item's prompts as written. Point Claude Code at the
  patch as a reference — "a reference implementation exists; read it, write
  only this item's scope". Use this to own every line and collect your own
  fail-first proof. Take only the ADR from the patch first, since every prompt
  frame reads it: `git apply --include='docs/adr/*' .ag-ref/aggregate-records-path.patch`.
- **Adopt mode.** `git am .ag-ref/aggregate-records-path.patch` once, then run only the
  prompts marked *adopt* in each item — verify, fail-first, review. The item's
  **Done when** list is the gate either way. First adopt step: `make test` with
  the real tokenizer (the sandbox ran an approximation because tiktoken's
  download host was blocked). Adopt mode cuts AG-01..AG-07 from ~31 prompts to
  ~10.

The patch lives at `.ag-ref/aggregate-records-path.patch`, excluded through
`.git/info/exclude` so it is never committed. It does not satisfy every
**Done when** below; gaps are marked **not in patch**.

---

## Decisions — settle before the item each one gates

Debdeep's call on each. The default is what the patch does.

| # | Decision | Default (patch) | Gates |
|---|---|---|---|
| D1 | Names: package `aggregate/`, `Answer.route` values `chunks`/`records`, config key `aggregate`, item prefix `AG` | as listed | AG-01 |
| D2 | Planner: model fills a typed plan behind a verbatim guard, vs a deterministic regex router | typed plan — generalises across phrasings; a wrong plan falls through instead of answering | AG-02 |
| D3 | Once a plan validates, the records path owns the answer (abstains if empty) vs falls back to chunks | owns it — falling back is how `q_cd_01`–`04` failed | AG-06 |
| D4 | Which extraction bands may contribute to a figure | `accept` + `review`, review values named in the answer — P1-03 measured 0 wrong of 93 review-band values | AG-04 |
| D5 | A value's route = weakest of the value and every field used to select it (supplier, currency, part) | yes | AG-04 |
| D6 | Currency: never summed across; other-currency orders counted, amounts not shown | yes | AG-04 |
| D7 | Wording: "visible to you", every exclusion named with a reason, one sentence per line | yes | AG-05 |
| D8 | Part-price gold = every priced line of that part the gold principal can read | yes (gold today lists a pair) | AG-10 |
| D9 | `aggregate.enabled` default | `false` until AG-11 passes | AG-11 |

---

## Conventions

Same as `WBS-P1.md`, tightened:

- Each item is **3–7 prompts**. The last prompt is always verify + review;
  items with a spare say so.
- **Done when** is checkable by running something. Every item ends with real
  `pytest` output pasted, `make test` and `pytest -m acl` green, and a line in
  the log.
- **Fail-first ×3 per item** — mutate or delete the check, paste the red run,
  restore, paste green. Each item names its three.
- Invariants asserted as identities (`included ∩ excluded == ∅`,
  `listed == oracle`) rather than specific numbers wherever possible.
- Results as **k/n with named questions** — never a percentage at this
  sample size.
- **Stop** marks a design decision: Claude Code stops and asks rather than
  choosing.
- Every command runs through `conda run -n Sai2608 …`.
- A prompt whose verify step is green ends with a commit `AG-0X.N: <summary>`.
  A red verify stops without committing, so every commit on the branch is a
  known-green point to reset to.

**Status:** ` ` not started · `~` in progress · `x` done · `-` dropped

### Prompt frame

Each prompt line below expands into one self-contained block:

```text
isc-docint · AG-0X prompt N/M — <title>
Read first: docs/WBS-AGG.md §AG-0X, docs/adr/0011.
Goal: <one sentence>.
In scope: <files>. Do not edit anything else.
Must hold: <invariants from this item's Watch and Done when>.
Do: <the prompt line, expanded>.
Verify: conda run -n Sai2608 pytest -q <paths> — paste the full output.
Stop after pasting. Do not start prompt N+1. If you hit a choice this
document or ADR 0011 does not settle, stop and ask.
```

---

## AG-01 — Query-side supplier resolution ☐

**Priority** critical · **~90 LOC** (+~70 test) · **Prompts** 3 · **Depends** none · **Stop** D1 before prompt 1

**Files** `src/isc/extract/masters.py` (additions only),
`src/isc/aggregate/__init__.py`, `src/isc/aggregate/resolve.py`,
`tests/unit/test_aggregate_resolve_and_plan.py`

**Prompts**
1. `masters.py`: public `normalise_supplier_name()` alias and
   `supplier_id_for_printed_name()` — the same exact-then-normalised,
   unique-only discipline as `resolve_supplier()`, which is not touched
2. `resolve.py`: `resolve_mention()` — a full master name (case- and
   whitespace-insensitive) → that entity only; otherwise whole-token prefix
   match on normalised names → every candidate; `supplier_names_by_id()`;
   parametrised tests including the printed-vs-asked contrast
3. *adopt* — fail-first ×3, review

**Done when**
- [ ] `resolve_mention("Kestrel Industrial") == ("V100781", "V100782")`;
      `"Kestrel Industrial AG"` → `("V100781",)`; `"Kes"`, `"Siemens"`, `""` → `()`
- [ ] Contrast test: the string `"Kestrel Industrial"` resolves to one id when
      printed and two ids when asked
- [ ] `git diff src/isc/extract/masters.py` shows additions only;
      `test_masters.py` unchanged and green
- [ ] Fail-first: prefix match → normalised equality (Kestrel case red);
      delete the exact-name branch (`…AG` case red); printed-name lookup
      returns the first of several matches (uniqueness test red)

**Watch** two policies, on purpose. A name printed on a document wants
exact-or-nothing; a name typed in a question wants every candidate. Reusing
the extraction resolver for a question is the `q_am_*` failure written as
code — "Kestrel Industrial" normalises to the same string as "Kestrel
Industrial AG".

**Watch** this is also the foundation for feature #2 (supplier filters in
`retrieve/`). Keep `resolve.py` free of aggregate-only imports so it can move
to a shared module without a rewrite.

**Patch** covers all of it — adopt mode runs prompt 3.

---

## AG-02 — Typed query plan and validation ☐

**Priority** critical · **~120 LOC** (+~70 test) · **Prompts** 3–4 · **Depends** AG-01 · **Stop** D2 before prompt 1

**Files** `src/isc/aggregate/plan.py`, `tests/unit/test_aggregate_resolve_and_plan.py`

**Prompts**
1. `QueryPlanRaw` (literal operation + three nullable strings copied from the
   question) and `QueryPlan` (resolved supplier ids) — the two-model pattern
   from `models/records/`; strict-schema test through `to_strict_schema()`
2. `validate_plan()` → `(plan, "")` or `(None, reason)`: supplier, part and
   currency must appear in the question verbatim; `PART_NUMBER` shape; part
   number matched as a whole token; supplier resolved through
   `resolve_mention()`; each operation's required parameter present
3. Tests for the four ways a planner changes the question — a completed
   supplier name, an inferred currency, an invented part number, a partial
   part token — plus lower-case part numbers accepted
4. *adopt* — fail-first ×3, review (spare if prompt 3 covered it)

**Done when**
- [ ] Strict schema: four required fields, `additionalProperties: false`,
      operation enum of exactly `total_spend` / `part_prices` / `none`
- [ ] Every rejection returns `None` with a reason — nothing raises, nothing abstains
- [ ] The four "changed the question" cases are each rejected by a named test
- [ ] Fail-first: drop the verbatim supplier check; drop the
      currency-in-question check; drop the whole-token boundary

**Watch** rejection means *not mine*, not *refuse*. A validator that abstains
turns a planner mistake into a user-visible refusal on a question the chunk
path would have answered.

**Watch** the verbatim rule is load-bearing. "Kestrel Industrial" silently
completed to "Kestrel Industrial AG" produces a precise, cited, wrong answer —
worse than the P1 failure it replaces.

**ADR** covered by 0011 (typed plan rather than generated SQL). Add an
addendum only if D2 goes the other way.

**Patch** covers all of it — adopt mode runs prompt 4.

---

## AG-03 — Permission-first record source ☐

**Priority** critical · **~90 LOC** (+~180 test incl. builder) · **Prompts** 4–5 · **Depends** none (parallel with AG-01/02)

**Files** `src/isc/aggregate/source.py`, `src/isc/storage/local_vector.py`
(`document_chunks`), `tests/aggregate_world.py`,
`tests/adversarial/acl/test_aggregate_acl.py`, `tests/unit/test_import_hygiene.py` (extend)

**Prompts**
1. `LocalVectorStore.document_chunks(document_id, principal)` — principal
   required, `may_read` per chunk, ordinal order; test that an unprivileged
   principal gets `[]`
2. `visible_records(principal, docs, store)` — `may_read(document.acl)` before
   `get_record()`; returns `RecordView(records, unextracted)`;
   `NotImplementedError` for anything but purchase orders
3. `tests/aggregate_world.py` — documents with ACLs, records with chosen
   per-field confidences, header/table/footer chunks shaped like the real
   chunker (the total printed only in the footer, `line_range` on tables)
4. Adversarial tests: a spy docstore proves a hidden record is never loaded;
   `unextracted` never counts a hidden document; the export-controlled
   jurisdiction case; AST hygiene — no module in `isc.aggregate` except
   `source.py` calls `.get_record(`
5. *adopt* — fail-first ×3, review

**Done when**
- [ ] Spy: `get_record()` is never called for a document the principal cannot read
- [ ] A hidden unextracted document does not appear in `unextracted`
- [ ] Hygiene test green: `source.py` is the only reader of records in
      `isc.aggregate` (**not in patch**)
- [ ] `pytest -q -m acl` green with the new tests included
- [ ] Fail-first: remove the `may_read` check in `source.py` (spy and leak
      tests red); drop the principal filter in `document_chunks` (chunk test
      red); call `get_record` from `execute.py` (hygiene test red)

**Watch** the `records` table has no ACL column — ACL lives on `Document`.
Without this item the aggregate path would be the first code path in the
repo that sees every document regardless of who asks. An aggregate leaks where
a chunk list does not: the sum discloses a hidden amount, the count discloses a
hidden order.

**Watch** P1 scale: it loads every `Document` payload per question. Fine at 20
documents; AG-12 changes the shape, not the contract.

**Patch** covers prompts 1–4 except the hygiene test — adopt mode runs the
hygiene part of prompt 4, then prompt 5.

---

## AG-04 — Executor: selection, confidence gate, Decimal ☐

**Priority** critical · **~280 LOC** (+~120 test) · **Prompts** 5–6 · **Depends** AG-01, AG-02, AG-03 · **Stop** D4–D6 before prompt 3

The densest item. Budget six prompts.

**Files** `src/isc/aggregate/execute.py`, `src/isc/common/config.py`
(`AggregateSettings.contributing_routes` + validator), `config/default.yaml`,
`tests/unit/test_aggregate_answer.py`

**Prompts**
1. Data model (`ValueRef`, `Group`, `AggregateResult`) and locators:
   `_record_supplier_id()` with the printed-name fallback, identity chunk,
   `_chunk_printing()` (prose chunks before tables), `_line_chunk()` via `line_range`
2. `_total_spend()` — select by resolved supplier id; currency filter with a
   count-only tally of other currencies; one group per (supplier, currency);
   Decimal sum
3. Confidence gate — route = weakest(value, supplier field, currency field);
   `contributing_routes` from config with a validator; every exclusion carries
   a reason (absent total, no currency, value not in indexed text, below band)
4. `_part_prices()` — every visible priced line of the part; line citation
   via `line_range`; unpriced lines excluded with a reason
5. Tests + fail-first ×3
6. *adopt* — review against D4–D6 (spare)

**Split seam if it overruns:** prompts 1–3 are total spend, 4–5 part prices.
Ship total spend green first.

**Done when**
- [ ] `0.10 + 0.20` renders exactly `0.30`
- [ ] One supplier in two currencies → two groups, and no cross-currency
      figure anywhere in the result
- [ ] A 0.99-confidence total under a 0.40-confidence currency field is excluded
- [ ] A record with no printed vendor code is selected through its printed
      name (the 9/20 case)
- [ ] `included ∩ excluded == ∅` and every excluded ref has a non-empty
      reason — asserted as identities
- [ ] Fail-first: remove the currency filter; remove the confidence gate;
      drop the printed-name fallback

**Watch** D4 is a process-owner decision, not an engineering default: which
review-band values a spend figure may include is exactly what finance will
argue with. Settle it before prompt 3; the config comment records why.

**Watch** "not located in the indexed text" is an exclusion, never a fallback
citation to the header. A number you cannot point at is a number you cannot cite.

**Patch** covers all of it — adopt mode runs prompts 5–6.

---

## AG-05 — Deterministic renderer through the same grounding checks ☐

**Priority** critical · **~170 LOC** (+~90 test) · **Prompts** 4 · **Depends** AG-04 · **Stop** D7 before prompt 1

**Files** `src/isc/aggregate/render.py`, `tests/unit/test_aggregate_answer.py`

**Prompts**
1. Evidence registry (first-seen numbering builds `supporting`, so `[n]`
   resolves to `supporting[n-1]`), `money()`, total-spend lines: a single
   supplier; an ambiguous mention → one line per supplier, never summed; one
   line per currency when the question names none
2. Part-price lines; review-pending, excluded, other-currency and not-checked lines
3. Grounding test parametrised over the single-supplier, ambiguous-supplier
   and part-price shapes: every marker resolves and none is unused,
   `bind_citations()` count == `len(supporting)`, `verify_attribution()`
   finds 0 mismatches, sentences == lines, each PO line cites only its own PO's chunks
4. *adopt* — fail-first ×3, wording review against D7

**Done when**
- [ ] Grounding test green on all three shapes
- [ ] The ambiguous case asserts the combined sum is absent from the text
- [ ] No sentence names a supplier without citing a chunk of that supplier
- [ ] Fail-first: merge PO lines into one sentence (grounding red); strip the
      markers from the headline (attribution red); add the two suppliers
      together (ambiguity test red)

**Watch** `verify_attribution()` splits only on `[.!?]` followed by a capital
or `[`. A `- ` bullet list merges every line into one sentence and lets one
PO's citation vouch for another PO's number — hence one sentence per line,
each starting with a capital. The patch's own mutation check found the
single-supplier path uncovered by the first version of this test; that is why
the test is parametrised.

**Watch** never name a supplier in an uncited sentence ("…Pneumatics GmbH has
no orders visible to you"). It fails attribution, and it tells a user
something they may not be entitled to know.

**Patch** covers all of it — adopt mode runs prompt 4.

---

## AG-06 — Planner prompt, answerer, orchestrator and CLI wiring ☐

**Priority** critical · **~200 LOC** (+~70 test) · **Prompts** 5 · **Depends** AG-02, AG-05 · **Stop** D3 before prompt 2

**Files** `config/prompts/aggregate/query_plan.v1.md`,
`src/isc/aggregate/answerer.py`, `src/isc/answer/orchestrator.py`,
`src/isc/models/answer.py` (`route`), `src/isc/common/config.py` +
`config/default.yaml` (`aggregate.enabled`), `src/isc/cli.py`, tests

**Prompts**
1. Planner prompt v1 — examples use invented suppliers and part numbers only;
   "when in doubt, `none`"; a test that no master supplier name or corpus part
   number appears anywhere in the prompt file
2. `RecordAnswerer.try_answer()` — plan via `parse_structured()`; a planner
   failure returns `None`; the `aggregate.plan` span carries `route` and
   `reason` on every question; an empty result gives `NO_RESULTS` (text
   identical to the chunk path's); nothing summable gives
   `INSUFFICIENT_CONTEXT`; bind + verify; `route="records"`
3. Orchestrator step 0 (optional aggregator; `None` is byte-for-byte the P1
   path), `Answer.route`, CLI `_aggregator()` behind `aggregate.enabled: false`
4. Tests: a records answer skips retrieval; `none` falls through;
   unparseable planner output falls through; route recorded
5. Live smoke, cents: `ISC_AGGREGATE__ENABLED=true isc ask` as `u_alice` for
   `q_cd_01` and one `single_hop` question, as `u_chen` for `q_am_04` — paste
   the answers, routes and the run's cost

**Done when**
- [ ] Prompt-leakage test green (**not in patch**)
- [ ] With the flag off, every pre-existing test passes unmodified
- [ ] Smoke: `q_cd_01` → `route=records`, `2,972,338.10 SGD`; `single_hop` →
      `route=chunks`; `q_am_04` → both Kestrel entities, not summed
- [ ] The `runs/<id>/` trace shows `aggregate.plan` with route and reason per question
- [ ] `test_import_hygiene.py` green — no `openai` outside `isc.llm`
- [ ] Fail-first: fall back to the chunk path on an empty result (NO_RESULTS
      test red); drop the verify step (attribution test red); add a real
      master supplier name to a prompt example (leakage test red)

**Watch** D3: once a plan validates, the records path owns the answer.
Falling back on an empty result reproduces `q_cd_01`–`04` exactly.

**Watch** the planner call runs on every question while the flag is on,
including questions that fall through. Record its per-question cost from the
smoke run; AG-11 needs it.

**Patch** covers prompts 2–4 — adopt mode runs prompt 1 (leakage test only)
and prompt 5.

---

## AG-07 — Offline gold harness ☐

**Priority** high · **~170 LOC** test · **Prompts** 3–4 · **Depends** AG-06

**Files** `tests/integration/test_aggregate_gold.py`

**Prompts**
1. Module-scoped fixture: a slice from `data/synthetic` into a temp dir — real
   ingest, parse and extract-wrap (the model scripted to return gold raw output
   with a confident logprob), chunking and indexing with a hash embedder;
   skips without `make corpus`
2. The nine gold aggregate questions with scripted plans, each asked as its
   gold principal: `answer_contains_gold()`, leak check, citations present,
   `route == "records"`
3. Part-price completeness — listed lines equal an oracle recomputed from the
   gold extraction files and ACL sidecars, independently of the code under test
4. *adopt* — run with no API key set (and assert it), fail-first ×3, review

**Done when**
- [ ] 9/9 named — `q_cd_01`–`08`, `q_am_04` — offline, 0 leaks
- [ ] Runs with `OPENAI_API_KEY` unset (**not in patch** as an explicit assertion)
- [ ] Oracle equality for `q_cd_05`–`08`
- [ ] Fail-first: skip the ACL check in `source.py`; drop the printed-name
      fallback; resolve mentions with the extraction policy — each turns
      named gold tests red

**Watch** this measures execute + render + ACL + citations. It does **not**
measure the planner — say so in the docstring and the log, and never quote
9/9 as the feature's accuracy. That number belongs to AG-11.

**Watch** extraction here is gold by construction, so extraction errors are out
of scope. AG-11 is where real extraction meets the records path.

**Patch** covers prompts 1–3 — adopt mode runs prompt 4.

---

## AG-08 — Planner routing eval, live ☐

**Priority** critical · **~150 LOC** (+~60 test) · **Prompts** 4–5 · **Depends** AG-06

**Files** `scripts/gen_planner_gold.py`, `data/gold/planner/cases.json`,
`scripts/eval_planner.py` (or `isc eval --harness planner`),
`tests/unit/test_eval_planner.py`

**Prompts**
1. Expected plan for all 56 retrieval-gold questions, generated from gold —
   9 aggregate with (operation, supplier, part, currency), 47 expecting
   `none`. Deterministic, never model-generated (the P1-08 rule)
2. Hand-written paraphrase and near-miss set (~20): aggregate phrasings the
   gold does not use ("How much have we paid Omron overall, in SGD?") and
   look-alikes that must route `none` ("What did we order from Kestrel
   Industrial?", "What is the order total on PO 4522345741?")
3. Runner, planner only, no retrieval: validated plan vs expected; outcomes
   are exact, wrong plan, misroute-in (non-aggregate → records), misroute-out
   (aggregate → chunks), and guard rejection by reason; k/n report to
   `runs/<id>/eval/planner.md`
4. Live run twice; paste both reports and the cost
5. **Stop** on any misroute-in — one prompt change at a time into
   `query_plan.v2.md` with a sibling `.NOTES.md`, re-run, log (spare)

**Done when**
- [ ] k/n per class: aggregate exact (target 9/9), non-aggregate `none`
      (target 47/47), paraphrases, near-misses
- [ ] Misroute-in reported as its own headline number, not folded into accuracy
- [ ] Two runs reported side by side — temperature 0 is not deterministic
- [ ] Cost per planner call recorded
- [ ] Fail-first: flip one expected `none` to `total_spend` (runner flags a
      misroute-in); corrupt one expected supplier (wrong plan); feed a
      guard-rejected plan (counted as misroute-out with its reason)

**Watch** misroute-in is the expensive direction. Misroute-out just reproduces
P1 behaviour; misroute-in replaces a working single-PO answer with a records
answer to a different question.

**Watch** don't tune the prompt against the gold questions' exact wording.
The paraphrase set is the check that you haven't.

**Patch** none — new work.

---

## AG-09 — Route in eval outcomes and the report ☐

**Priority** high · **~120 LOC** (+~80 test) · **Prompts** 4–5 · **Depends** AG-06; EV-01/EV-02 in place · **Stop** before prompt 2

**Files** `src/isc/eval/retrieval.py` (`QuestionOutcome.route`),
`src/isc/eval/outcomes.py`, `src/isc/eval/report.py`, `src/isc/eval/diff.py`,
tests + fixtures

**Prompts**
1. `QuestionOutcome.route`, populated from `answer.route` in `run()`
2. `OUTCOMES_SCHEMA_VERSION` 1 → 2. **Stop:** the loader rejects any other
   version by design. Decide: read v1 with `route="chunks"` (true of every
   pre-AG run), or stay strict — strict means historical v1 runs can no
   longer be diffed against new ones
3. Report: answer accuracy by route × subtype as k/n cells, plus a named list
   of non-aggregate questions that were answered from records
4. `eval-diff`: route flips classified and shown per question
5. *spare* — fail-first ×3, review

**Done when**
- [ ] v2 files round-trip; the chosen v1 policy is tested against
      `tests/fixtures/ev02_synthetic_outcomes/`
- [ ] Rescoring the v1 fixture reproduces the previous report, plus the new section only
- [ ] Every new cell carries its denominator (EV-02)
- [ ] Fail-first: drop `route` from the writer (round-trip red); read v1
      without the default (fixture red); hide route flips in the diff (diff test red)

**Watch** `--rescore-from` reuses the persisted `answer_correct`; it does not
re-check against current gold. Anything that changes gold (AG-10) needs a
fresh run, not a rescore.

**Patch** none — new work.

---

## AG-10 — Complete the part-price gold ☐

**Priority** high · **~60 LOC** (+~30 test) · **Prompts** 3–4 · **Depends** none — land before AG-11 · **Stop** D8 before prompt 2

**Files** `scripts/gen_gold.py` (`_cross_doc_part`),
`data/gold/retrieval/questions.json`, `tests/integration/test_gold_fidelity.py`

**Prompts**
1. Record the gap against current gold: for each of `q_cd_05`–`08`, oracle
   line count vs gold line count (known: 12 vs 2, 20 vs 2, 3 vs 2, 11 vs 2)
2. `_cross_doc_part()` — every priced line of the part in documents the gold
   principal can read (gold extraction ∩ ACL sidecars); `gold_chunk_ids` =
   every covering line chunk, deduplicated
3. Regenerate, then check fidelity: provenance fingerprints unchanged (the
   corpus is unchanged), every new gold chunk id resolves against a fresh
   `make slice` index, and gold lines == the AG-07 oracle
4. *spare* — before/after note in the log

**Done when**
- [ ] For `q_cd_05`–`08`, gold's listed lines == the oracle, asserted as an identity
- [ ] `test_gold_fidelity.py` green against a fresh `make slice`
- [ ] The log records this as a measurement change, with the chunk path's k/n
      on these four questions before and after
- [ ] Fail-first: drop the ACL intersection (oracle mismatch red); keep only
      the first two lines (identity red); skip the unpriced-line filter (mismatch red)

**Watch** this is a measurement change, not a system change. On the chunk path,
answer accuracy on these four questions should *drop*, because 8 chunks cannot
list 12–20 lines. That is the gold becoming honest; don't compare across it
silently. `isc eval-diff` will classify these four as `gold_change`.

**Watch** new `gold_chunk_ids` move `cross_document` recall@8 as well.

**Patch** none — found while building the patch, not fixed in it.

---

## AG-11 — Live eval and the go/no-go on the flag ☐ ← **MILESTONE**

**Priority** critical · **~20 LOC** + docs · **Prompts** 4–5 · **Depends** AG-06, AG-08, AG-09 (AG-10 first, recommended)

**Files** `config/default.yaml`, `README.md`, `docs/LIMITATIONS.md`,
`docs/adr/0011-aggregate-questions-from-records.md`, this document's log

**Prompts**
1. Fresh `make slice`; retrieval eval with the flag **off** → run A. This is
   the same-index baseline: diff against A, not the historical P1-09 run, so
   the flag is the only thing that differs
2. The same eval with `ISC_AGGREGATE__ENABLED=true`, twice → runs B1, B2; paste both reports
3. `isc eval-diff A B1` and `A B2`; classify every changed outcome as fixed,
   regressed or measurement change
4. **Go/no-go** against the criteria below — go: flip the default, ADR 0011 →
   accepted, README numbers updated; no-go: log what failed and leave the flag off
5. *spare* — LIMITATIONS update either way

**Done when**
- [ ] Aggregate questions as named k/n — target 5/5 total spend (`q_cd_01`–`04`,
      `q_am_04`) and 4/4 part prices against the completed gold, on both B runs
- [ ] 0 misroute-in across the other 47 questions (AG-09's table)
- [ ] 0 ACL leaks — the hard gate, unchanged
- [ ] No regression on any other subtype per `eval-diff`
- [ ] Per-question cost delta from the planner call recorded
- [ ] README states measured numbers; ADR 0011 status updated

**Watch** two B runs, both reported, before flipping. One passing run is a
single reading, not a result.

**Feature done here.** AG-12 is hardening for the Azure landing.

---

## AG-12 — Records carry their own ACL; record-source port ☐

**Priority** medium · **~150 LOC** (+~100 test) · **Prompts** 6–7 · **Depends** AG-11 · **Stop** before prompt 2 · **ADR** yes (0012)

**Files** `src/isc/storage/ports.py` (`RecordSource`),
`src/isc/storage/sqlite_docstore.py`, `src/isc/extract/pipeline.py`,
`src/isc/aggregate/source.py`, `tests/adversarial/acl/test_aggregate_acl.py`,
`docs/adr/0012-…`

**Prompts**
1. ADR 0012 draft — project the document's `AclSet` onto its record at
   extract time, the way chunks carry theirs from index time (ADR 0004: a
   filter that needs a lookup gets skipped), vs keeping the join at query time
2. **Stop** on the draft; then the schema: `records.acl` required —
   `upsert_record()` has no way to write a record without permissions,
   mirroring the `Chunk` invariant; backfill for existing docstores
3. `RecordSource` protocol with one method, `permitted_records(principal,
   doc_type)`; the SQLite implementation filters on the stored ACL before
   deserialising any payload and never loads `Document`
4. Consistency: record ACL == document ACL for every corpus document
   (identity); re-ingesting with a changed ACL updates the record
5. Point `aggregate/source.py` and the adversarial tests at the port; fail-first ×3
6. Azure landing, design only: an Azure SQL table plus a principal-terms
   child table, with a row-level security predicate over the caller's
   expanded terms as the second enforcement point
7. *spare* — review

**Done when**
- [ ] A record cannot be written without an `AclSet` (validator test)
- [ ] Spy: `permitted_records()` never deserialises a hidden record
- [ ] Record ACL == document ACL for all 20 documents
- [ ] `isc.aggregate` no longer imports `SqliteDocStore` or `Document`
- [ ] Fail-first: allow a record write without an ACL; filter after
      deserialising; let the record ACL drift from the document's

**Watch** staleness. A SharePoint permission change must re-project onto
records exactly as it re-indexes chunks, or the records path serves
yesterday's permissions. If this item doesn't solve it, it goes in LIMITATIONS.

**Patch** none — new work.

---

## Order

```
AG-01 ─► AG-02 ─┐
AG-03 ──────────┴─► AG-04 ─► AG-05 ─► AG-06 ─┬─► AG-07
                                             ├─► AG-08 ─┐
                                             └─► AG-09 ─┼─► AG-11 ─► AG-12
AG-10  (independent — land before AG-11) ───────────────┘
```

The critical path runs through AG-04. AG-03 and AG-10 can run alongside it.

---

## Verification ladder

Run at every item, cheapest first:

```bash
conda run -n Sai2608 make test                                         # unit + adversarial; stays green
conda run -n Sai2608 pytest -q -m acl                                  # permission invariants; never skip
conda run -n Sai2608 pytest -q tests/integration/test_aggregate_gold.py  # from AG-07
conda run -n Sai2608 python scripts/eval_planner.py                    # from AG-08 — live, cents
conda run -n Sai2608 make slice                                        # AG-10, AG-11
```

---

## Definition of done

1. AG-11's criteria met on both B runs, and `aggregate.enabled` flipped on that evidence
2. ADR 0011 accepted; ADR 0012 written if AG-12 ran
3. README and LIMITATIONS state measured numbers, including the part-price gold change
4. The aggregate tests are part of the adversarial suite, unskipped and passing
5. Zero ACL leaks on both answer paths

---

## Log

| Date | Item | Prompts | Note |
|---|---|---|---|
| 2026-09-28 | AG-01..07 reference patch | — | Built in a sandbox with an approximate tokenizer (tiktoken's download host blocked); 71 new tests; 9/9 offline with scripted plans; the mutation check found the single-supplier sentence layout uncovered — test parametrised; found the part-price gold incomplete (→ AG-10). Not yet run on `Sai2608`. |
| 2026-09-28 | AG-00 preflight | 2 | Sai2608 editable install repointed from the removed BabaKoreDao path; baselines: make test 561 passed, acl 16, ruff 55, mypy 18 errors in 8 files; patch --check clean |
| 2026-09-28 | AG adopt — patch applied | 1 | make test 632 passed, acl 22 passed, aggregate gold 13/13 offline; ruff 55, mypy 18 |
| | | | |

# P1 Work Breakdown — ISC Document Intelligence & Knowledge Platform

Working document for `isc-docint` only. Cross-project sequencing lives in
`docs/WBS.md`. Drop this at `docs/WBS-P1.md`.

**Target:** ~1,670 LOC across 12 items. **Milestone:** P1-10, after which P1 is
demonstrable end to end.

---

## Current state

| Layer | State |
|---|---|
| `common/` confidence, tracing, ids, config | complete |
| `llm/` ports, registry, OpenAI client, structured output | complete |
| `models/` acl, document, chunk, answer, records | complete |
| `storage/` blob, sqlite docstore, vector store | complete |
| `retrieve/fusion.py` RRF | complete |
| `eval/` metrics, report scaffolding | complete; comparison functions stubbed |
| `scripts/` masters, corpus, acl graph | complete, deterministic, fidelity-tested |
| ingest, parse, extract, index, retrieve, answer | typed stubs |
| Corpus | 20 POs, gold in raw + normalised form, ACL sidecars |

**Baseline before P1-01:** `make corpus && make test` green.

---

## Conventions

Each item is **3–4 prompts**: build → verify → review, with one spare. Definitions
of Done are written to be checkable by running something, not by reading code.

**Every item ends with:** tests added, `make test` green, and a line in the log.

**ADR trigger:** write one when the item makes a structural choice a reviewer would
question. Marked per item below.

**Status:** ` ` not started · `~` in progress · `x` done · `-` dropped

---

## P1-01 — Native text parser ☒

**Priority** critical · **~150 LOC** · **Depends** none

**Files** `src/isc/parse/chain.py`, `src/isc/parse/pipeline.py` (new),
`src/isc/cli.py`, `tests/unit/test_parse.py`

**Prompts**
1. Implement `NativeTextParser.parse()` — pypdf text layer → `Page` / `Block`,
   heuristic `BlockType` assignment, `reading_order` sequencing
2. Stage runner + `isc parse` verb: blob → fallback chain → `Document` JSON in
   `runs/<run_id>/parse/`
3. Tests + run over the full corpus, fix what breaks

**Done when**
- [ ] 20/20 corpus PDFs parse without falling through the chain
- [ ] The 4 multi-page documents produce 2 `Page` objects each
- [ ] `Document.text()` contains every gold value (reuse the fidelity check)
- [ ] `parser="native"`, `parser_degraded=False`, `LAYOUT=1.0` factor returned
- [ ] `runs/<run_id>/parse/*.json` round-trips through `Document.model_validate_json`

**Watch** pypdf gives no bounding boxes. Leave `bbox=None` — P1-02 locates spans by
string search. A fabricated bbox is worse than a missing one, because review UI will
point a human at the wrong place on the page.

**Watch** the reprinted table header on page 2 will appear twice in `text()`. Do not
dedupe it here; that is table reconstruction's job, and silently dropping repeated
lines will eat legitimate duplicate rows.

---

## P1-02 — Extraction wrap + validators ☒

**Priority** critical · **~250 LOC** · **Depends** P1-01

The densest item in the project. Budget four prompts and expect to use them.

**Files** `src/isc/extract/extractor.py`, `src/isc/extract/validators.py`,
`src/isc/extract/spans.py` (new), `tests/unit/test_extract.py`

**Prompts**
1. `spans.py` — locate a raw string value in the parsed blocks, return `Span`.
   Normalised matching (whitespace, thousands separators) with a miss returning
   `None` rather than a guess
2. `_wrap()` field mapping — `PurchaseOrderRaw` → `PurchaseOrder`, every field an
   `ExtractedField` with value + span, no confidence yet
3. Confidence assembly — fold in the six signals, wire the HITL enqueue
4. Master-data loading, line-item handling, run against the real corpus

**Split seam if it overruns:** prompts 1–2 are "values in the right shape", 3–4 are
"values with defensible confidence". Ship the first half working before starting the
second.

**Done when**
- [ ] Every `ExtractedField` has a value (or explicit `absent_reason`), a
      `Confidence` with ≥2 factors, and a `Span` where the value was found
- [ ] Six signals wired: `MODEL` (logprob), `SCHEMA` (repair discount),
      `LEXICAL` (pattern), `MASTER_DATA` (supplier/part lookup),
      `AGREEMENT` (line sum vs header total), `LAYOUT` (from parse)
- [ ] Ambiguous dates carry the reduced `LEXICAL` factor from `parse_iso_date`
- [ ] Documents with unprinted extended prices do **not** raise a false
      `AGREEMENT` conflict
- [ ] Unmastered parts produce a `MASTER_DATA` miss without a wrong-value flag
- [ ] Fields below threshold appear in `review_queue` with the weakest factor named
- [ ] Second run is free (cache hit); `runs/<id>/summary.json` shows the cost of the first

**Watch** the model must never set its own confidence. If a prompt revision tempts
you to ask for a certainty score, that is the wrong fix.

**Watch** `total_amount` corroboration: the corpus prints the true total even when
individual extended prices are absent. Reconciliation must sum `quantity × unit_price`
in that case, not the printed extendeds.

**ADR** yes — span location by string search, and what happens on a miss.

---

## P1-03 — Extraction eval harness ☐

**Priority** critical · **~150 LOC** · **Depends** P1-02

**Files** `src/isc/eval/extraction.py`, `src/isc/eval/normalise.py` (new),
`src/isc/cli.py`, `tests/unit/test_eval_extraction.py`

**Prompts**
1. `normalise.py` + `compare()` — type-aware comparison, four outcome classes
2. `isc eval --harness extraction`, report writing
3. Run over the corpus, read the calibration curve, tune or explain

**Done when**
- [ ] `correct` / `wrong` / `missed` / `correct_absent` scored separately
- [ ] Dates compare ISO-to-ISO, decimals to 2dp, strings casefold + strip
- [ ] Per-field P/R/F1 table; no single aggregate headline number
- [ ] `calibration_bins` produces a curve with ≥4 populated bins
- [ ] `auto_accept_error_rate(0.90)` reported in the Markdown report
- [ ] Report at `runs/<id>/eval/report.{json,md}`

**Watch** this item tells you whether the confidence design actually works. If the
0.9 bin is right 60% of the time, the combinators are wrong — fix them here rather
than proceeding and inheriting a meaningless threshold.

**Watch** scoring `correct_absent` in the same bucket as `correct` makes a model that
returns null for everything look good on sparse documents. Keep them separate in the
report, not just in the code.

**ADR** only if you change the combinators as a result.

---

## P1-04 — Layout-aware chunker ☐

**Priority** critical · **~150 LOC** · **Depends** P1-01 (parallel with 02/03)

**Files** `src/isc/index/chunker.py`, `tests/unit/test_chunker.py`

**Prompts**
1. Windowing with `target_tokens` / `overlap_tokens`, heading breadcrumbs,
   ACL inheritance
2. Table handling — emit whole under `max_table_tokens`, prepend header row
3. Tests including determinism and the table-split guard

**Done when**
- [ ] Chunks respect the configured token budget; `section_path` populated
- [ ] Every chunk carries the document's `AclSet` — no post-attachment step exists
- [ ] No chunk splits a table row (assert directly in a test)
- [ ] Same document + settings → identical `chunk_id`s across runs
- [ ] `filters` populated with `po_number` and `supplier_id` where extraction has them

**Watch** a half table retrieves confidently and answers wrongly. This is the one
correctness rule in the item.

**Watch** chunk ids feed P1-08's gold. Changing chunking settings after the gold set
exists silently invalidates every recall number.

---

## P1-05 — Index pipeline wiring ☐

**Priority** critical · **~80 LOC** · **Depends** P1-04

**Files** `src/isc/index/pipeline.py`, `src/isc/cli.py`

**Prompts** — small enough for two: build + verify.

**Done when**
- [ ] `isc index` populates the store for all 20 documents
- [ ] `store.count()` non-zero; `store.save()` / `load()` round-trips
- [ ] Embeddings cache — second run makes no API calls
- [ ] The index-time `AclViolation` guard is exercised by a test

---

## P1-06 — Retriever completion ☐

**Priority** critical · **~120 LOC** · **Depends** P1-05

**Files** `src/isc/retrieve/retriever.py`, `src/isc/retrieve/filters.py` (new),
`tests/unit/test_retriever.py`

**Prompts**
1. `rewrite()` passthrough; `infer_filters()` deterministic regex over PO / part /
   supplier patterns
2. Wire hybrid dense + BM25 through RRF, filters applied pre-search
3. Tests including a restricted principal getting an empty result

**Done when**
- [ ] Hybrid retrieval fuses both paths via RRF
- [ ] `infer_filters("PO 4500123")` → `{"po_number": "4500123"}`, deterministic
- [ ] A filter that matches nothing returns empty rather than falling back to unfiltered
- [ ] A restricted principal's results are empty pre-ranking, verified by the ACL suite

**Watch** regex before model. A hallucinated filter silently empties the result set,
which presents to the user as a relevance problem and is very hard to trace.

**Watch** keep the original question as query one, always. Rewrites lose the exact
part numbers BM25 depends on.

---

## P1-07 — Answer orchestrator ☐

**Priority** critical · **~150 LOC** · **Depends** P1-06

**Files** `src/isc/answer/orchestrator.py`, `src/isc/answer/citations.py` (new),
`tests/unit/test_answer.py`

**Prompts**
1. `_generate()` — numbered context blocks, grounded prompt
2. `_bind_citations()` — parse `[n]` markers, resolve to chunk ids, drop unresolvable
3. Abstention paths + tests

**Done when**
- [ ] Every retained claim maps to a retrieved chunk id
- [ ] `INSUFFICIENT_CONTEXT` from the model → abstention, never passed through as text
- [ ] Zero valid citations → `UNGROUNDED_DRAFT` abstention
- [ ] `NO_PERMITTED_RESULTS` and `NO_RESULTS` are byte-identical to the caller
- [ ] `isc ask "..." --as u_alice` returns a cited answer against the real index

**Watch** the citation binding step is the one people skip. An answer that cannot be
traced to a source is indistinguishable from a fabrication, and in a supply-chain
context it will be acted on.

**ADR** yes — abstention policy and why binding failure abstains rather than degrades.

---

## P1-08 — Retrieval gold set ☐

**Priority** critical · **~180 LOC** · **Depends** P1-05

**Files** `scripts/gen_gold.py`, `tests/integration/test_gold_fidelity.py`

**Prompts**
1. Question generation from extraction gold — answerable (single and multi-hop)
2. Unanswerable and restricted classes, principal assignment
3. Resolve gold chunk ids against the built index; fidelity tests

**Done when**
- [ ] 40+ questions: ~70% answerable, ~15% unanswerable, ~15% restricted
- [ ] Every question carries the principal it must be asked as
- [ ] Gold chunk ids resolve against the current index (verified, not assumed)
- [ ] Restricted questions verified answerable for A and invisible to B
- [ ] Deterministic under `--seed`, same convention as the corpus
- [ ] Unanswerable questions are *plausible* — they should look answerable

**Watch** generate questions from the gold records, never from the model. A
model-generated question set encodes the model's own reading of the corpus, which is
circular when you then use it to evaluate that model.

**Watch** if a restricted question happens to be answerable from a document the
restricted user *can* see, it silently tests nothing. Verify both directions.

---

## P1-09 — Retrieval eval runner ☐

**Priority** critical · **~120 LOC** · **Depends** P1-07, P1-08

**Files** `src/isc/eval/retrieval.py`, `src/isc/eval/report.py`, `src/isc/cli.py`

**Done when**
- [ ] recall@5, recall@8, MRR, nDCG@8 on answerable questions
- [ ] Abstention precision and recall on the unanswerable slice
- [ ] Each question run **as its gold principal**, not as a superuser
- [ ] One ACL leak → `passed() == False` and a non-zero exit code
- [ ] Both harnesses render into one report

**Watch** running the eval as an unrestricted principal would make every number look
better and test nothing about permissions. The principal comes from the gold.

---

## P1-10 — CLI end-to-end ☐ ← **MILESTONE**

**Priority** critical · **~100 LOC** · **Depends** P1-09

**Files** `src/isc/cli.py`, `Makefile`, `README.md`, `docs/architecture.md`

**Done when**
- [ ] `make slice` runs corpus → ingest → parse → extract → index → eval from clean
- [ ] A fresh clone reproduces it from the README alone, no undocumented steps
- [ ] `runs/<id>/` contains trace, per-stage artifacts, and both eval reports
- [ ] `summary.json` reports total tokens and cost for the full slice
- [ ] README states the actual measured numbers, not aspirations

**P1 is demonstrable here.** Items 11 and 12 are enrichment. If time runs short,
this is the honest stopping line.

---

## P1-11 — Second doc type (Invoice) ☐

**Priority** high · **~120 LOC** · **Depends** P1-10

**Files** `src/isc/models/records/invoice.py`, `config/prompts/extract/invoice.v1.md`,
`scripts/gen_corpus.py`, `data/gold/`

**Done when**
- [ ] Invoices generate with gold, ACL sidecars, and the same determinism guarantees
- [ ] Extraction runs through the identical code path — no `if doc_type ==` branching
- [ ] Eval reports both types separately
- [ ] Some invoices reference real PO numbers from the corpus (P2 needs this link)

**Watch** the value here is proving the registry dispatch generalises. If adding the
second type requires touching `extractor.py`, the abstraction was wrong and that is
worth fixing now rather than at type three.

---

## P1-12 — HITL review + calibration report ☐

**Priority** high · **~100 LOC** · **Depends** P1-03

**Files** `src/isc/cli.py`, `src/isc/eval/report.py`, `src/isc/storage/sqlite_docstore.py`

**Done when**
- [ ] `isc review` lists the queue weakest-first with the failing signal named
- [ ] Resolution writes back with `Signal.HUMAN` and the reviewer id
- [ ] A resolved field is excluded from the next review pass
- [ ] Calibration curve rendered in the Markdown report

**Watch** the reviewer needs to know *why* a field was flagged, not just that it was.
Showing the weakest factor is the whole point of decomposed confidence.

---

## Verification ladder

Run at every item, cheapest first:

```bash
make test                 # unit + adversarial; must stay green
make corpus-verify        # gold still matches the rendered PDFs
pytest -q -m acl          # permission invariants; never skip
make slice                # after P1-10 exists
```

---

## Definition of P1 done

1. `make slice` runs clean from a fresh clone
2. Extraction and retrieval reports both produced, with real numbers in the README
3. Zero ACL leaks; adversarial suite unskipped and passing
4. Four ADRs present and current
5. `docs/LIMITATIONS.md` written — see below

---

## Known limitations to record as you go

Start `docs/LIMITATIONS.md` at P1-01 and append. It is the clearest evidence of
critique depth, and it pre-empts questions you would otherwise answer live.

Seeded from what is already true:

- No OCR path; scanned documents fall through the chain
- No bounding boxes from the native parser; spans are located by string search
- No table reconstruction; tables are emitted as text
- Two document types, not the fifteen a production system would need
- Synthetic corpus — no real ERP export formats, no genuinely adversarial layouts
- Local vector store is brute force; not the ANN behaviour of AI Search
- Header-based identity in the API; Entra token validation is a placeholder
- Group expansion is static, not refreshed from Graph
- Calibration measured on 20 documents — indicative, not tight

---

## Log

| Date | Item | Prompts | Note |
|---|---|---|---|
| 2026-08-09 | P1-00 scaffold + corpus | ~8 | Predicted page counts instead of measuring; sampled where stratification was needed |
| 2026-08-09 | LLM layer fixes (.env credential propagation, strict schema, cache collision) | 3 | Lint debt, left alone: `UP035` (Sequence import from `typing` not `collections.abc`) in `ports.py`/`openai_client.py`, pre-existing, not introduced here |
| 2026-08-09 | P1-01 (retroactive) | — | Status checkbox was never flipped when the item finished; corrected alongside P1-02 |
| 2026-08-09 | P1-02 extraction wrap + validators | ~14 | Content-anchored row scoping tried first, measured at 18% resolution, rejected for positional + alignment self-check (90%); found and fixed a real max_tokens truncation bug (2048 too small above ~36 line items) that silently failed 4/20 documents until finish_reason was checked; found and fixed a missing %d.%m.%Y date format that scored every DE-site date as unparseable; found a genuine model extraction error (wrong unit_price on po_010) via three independent signals agreeing |
| | | | |

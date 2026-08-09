# ISC Document Intelligence & Knowledge Platform

Combined document extraction and ACL-aware knowledge assistant over a synthetic
industrial supply-chain corpus. This repo is the P1 scaffold: the seams are built
and tested, the pipeline stages are typed stubs.

## Quick start

```bash
conda env create -f environment.yml
conda activate Sai2608
cp .env.example .env          # add OPENAI_API_KEY
make corpus                   # generates masters, ACL graph, 20 synthetic POs + extraction gold
make test                     # 200 tests, no network required
```

The env name is case-sensitive — `conda activate sai2608` will fail.

## What is already real

| Component | State |
|---|---|
| `common/confidence.py` | Complete — factors, four combinators, routing thresholds |
| `common/tracing.py` | Complete — nested spans, JSONL traces, cost rollup |
| `common/ids.py`, `config.py` | Complete — deterministic ids, layered config |
| `llm/ports.py` + registry | Complete — provider Protocols, import hygiene enforced |
| `llm/openai_client.py` | Complete — retries, logprobs, cache, cost accounting |
| `llm/structured.py` | Complete — constrained JSON with bounded repair loop |
| `models/acl.py` | Complete — expansion, deny precedence, sensitivity, export control |
| `models/records/base.py` | Complete — `ExtractedField[T]`, rollup, registry |
| `storage/local_vector.py` | Complete — dense + BM25, ACL pre-filtering |
| `eval/` metrics + reports | Complete — the comparison functions are stubs |
| `retrieve/fusion.py` | Complete — RRF |
| `tests/adversarial/acl/` | Complete — 15 hard invariants |
| `scripts/gen_masters.py`, `gen_acl_graph.py`, `gen_corpus.py` | Complete — masters, ACL graph, 20-PO synthetic corpus with extraction gold |
| ingest / parse / extract / index / retrieve / answer | Typed stubs |

## First slice — build this before widening anything

One document type, end to end, ~1,200 LOC. It exercises every seam; everything
after it is additive rather than architectural.

1. `scripts/gen_corpus.py` — 20 POs. **Done.** Generates the gold record first,
   renders the document from it. Annotating generated documents afterwards
   reintroduces the labelling error you were avoiding. Enforced by
   `tests/integration/test_corpus_fidelity.py`; output is deterministic under
   `--seed` (default 2608).
2. `parse/chain.py::NativeTextParser` — pypdf pages → `Block(PARAGRAPH)`.
3. `extract/extractor.py::_wrap` — map 8 `PurchaseOrderRaw` fields, locate spans
   by string search, apply `validators.py`. This is the most important function
   in the repo.
4. `index/chunker.py` — fixed-size windows with heading breadcrumbs.
5. `retrieve/retriever.py::rewrite` — return `[question]`. Nothing clever yet.
6. `answer/orchestrator.py::_generate` / `_bind_citations`.
7. `eval/extraction.py::compare` and the retrieval runner.

Then: `make slice`.

## Widening order

1. Second doc type (`Invoice` — schema already written) to prove the registry
   dispatch generalises
2. Table reconstruction + `LayoutParser`, because line items are where extraction
   actually gets hard
3. Hybrid retrieval and metadata filter inference
4. OCR path and the confidence penalties it triggers
5. Graph connector (replaces `LocalDirectorySource`, not the interface)
6. Remaining doc types

## Invariants

These are not preferences. They are enforced by tests and validators.

- **No module outside `isc.llm` imports `openai`.** Enforced by AST inspection in
  `tests/unit/test_import_hygiene.py`.
- **A chunk cannot exist without ACL terms.** Enforced at three layers: `AclSet`
  validator, `Chunk` model validator, and an index-time guard.
- **Filtering is pre-ranking, on both the dense and lexical paths.**
- **One ACL leak fails the eval run** regardless of every other metric.
  `tests/adversarial/acl/` may never be skipped or weakened to land a feature.
- **The model never reports its own confidence.**
- **Prompts live on disk, versioned in the filename.** Never inlined.

## Layout

```
config/prompts/     versioned prompts, loaded by path
src/isc/common/     confidence, tracing, config, ids  ← becomes isc-core
src/isc/llm/        provider port + OpenAI/Azure impls ← becomes isc-core
src/isc/models/     domain models, storage-agnostic    ← becomes isc-core
src/isc/storage/    ports + local implementations
src/isc/<stage>/    ingest parse extract index retrieve answer eval
docs/adr/           why, not what
tests/adversarial/  the ACL suite
```

See `docs/architecture.md` for the stage pipeline and `docs/adr/` for the four
structural decisions.

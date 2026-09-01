# Document Intelligence PoC

A minimal Purchase Order document-intelligence proof of concept that:

- extracts structured Purchase Order fields from native-text PDFs;
- chunks PDF text page by page;
- embeds the user question and document chunks;
- ranks chunks with cosine similarity;
- answers questions using only retrieved evidence; and
- returns source chunk IDs and application-owned evidence.

This is a focused demonstration, not a production-ready document platform.

## End-to-end flow

```text
PDF -> PDFPage[] -> PurchaseOrder
PDFPage[] -> Chunk[]
question + chunks -> embeddings -> cosine retrieval -> SourceEvidence[]
question + SourceEvidence[] -> GroundedAnswer
all stages -> PipelineResult
```

## Core models

- `PDFPage` holds native text from one PDF page, numbered from 1.
- `LineItem` holds an optional part number, description, quantity, and unit price.
- `PurchaseOrder` holds the extracted header fields and line items.
- `Chunk` is a deterministic, page-local text window.
- `SourceEvidence` combines a retrieved chunk with its cosine similarity score.
- `GroundedAnswer` contains an answer and its supporting source chunk IDs.
- `PipelineResult` contains the extracted Purchase Order, answer, and resolved evidence.

Quantity and monetary values use `Decimal` internally.

## Technical design

### PDF extraction

`pypdf` extracts native text with one-based page numbering. Blank pages are preserved
in the page list. Encrypted, malformed, and all-empty documents are rejected. There is
no OCR fallback.

### Structured extraction

The OpenAI Responses API parses structured output directly into `PurchaseOrder`.
Prompts require explicit source support: missing values remain null, and the model is
told not to calculate absent values, infer facts, or use outside knowledge.

### Chunking

Each nonblank page is split independently into overlapping character windows. The
defaults are 1,200 characters with 200 characters of overlap. Chunk IDs are
deterministic and retain the one-based source page number.

### Embeddings and retrieval

The question and all chunks are sent together in one embeddings request. Retrieval
uses pure-Python cosine similarity and stable top-k ranking; no vector database is
used. Similarity is a ranking signal, not factual confidence.

### Grounded answering

Only the retrieved `SourceEvidence` objects are supplied to the answering model. The
model returns source chunk IDs, and the application validates and resolves those IDs
back to application-owned, PDF-derived evidence text. If the sources are insufficient,
the required answer is:

> I don't have enough information in the provided sources.

### Pipeline behavior

`process_document()` retrieves the top three chunks. Pipeline errors propagate to the
caller. There is no automatic retry or repair framework and no persistence layer.

## Setup

Create the declared Conda environment and activate it:

```bash
make install
conda activate Sai2608
```

Copy `.env.example` to `.env` and configure exactly these variables:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`

Do not commit secrets.

## Run the demo

```bash
python demo.py
```

The demo prompts for a PDF path and a question, then displays the extracted Purchase
Order, grounded answer, supporting sources, and cosine retrieval similarity. The
similarity value ranks retrieved text; it is not a factual-confidence score.

For non-interactive runs, pass both as CLI flags instead of using the prompts:

```bash
python demo.py --pdf path/to/purchase_order.pdf --question "What are the payment terms?"
```

`--pdf` and `--question` are independent: omit either one and the demo falls back to
its interactive prompt for that value.

## Test and lint

From the activated environment:

```bash
make test
make lint
```

The current offline suite contains 181 tests. `make lint` runs Ruff and strict MyPy.

## Project tree

```text
.env.example
.gitignore
.vscode/
  settings.json
Makefile
README.md
demo.py
environment.yml
pyproject.toml

src/isc/
  __init__.py
  chunking.py
  llm.py
  models.py
  pdf.py
  pipeline.py
  retrieval.py

tests/
  test_chunking.py
  test_demo.py
  test_llm.py
  test_models.py
  test_pdf.py
  test_pipeline.py
  test_retrieval.py
```

## Limitations

- Native-text PDFs only; there is no OCR.
- Purchase Orders are the only supported document schema.
- Structured extraction, answering, and embeddings use OpenAI services.
- Retrieval is in memory only; there is no persistence or vector database.
- There is no ACL or security model.
- There is no API service or CLI framework beyond `demo.py`.
- There is no hybrid BM25/RRF retrieval.
- There is no automatic retry or repair framework.
- Retrieval similarity is not factual confidence.
- Grounded answers are structurally validated against supplied source IDs, but their
  semantic citation support is not independently verified.
- The PoC is not production hardened.

## Validation status

The PoC has been validated with 181 offline tests, Ruff, strict MyPy, and one
controlled live end-to-end run using a synthetic Purchase Order. This validation is
evidence for the scoped PoC behavior, not production certification.

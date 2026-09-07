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

## Local FastAPI Deployment

From the repository root, activate the existing environment:

```bash
conda activate Sai2608
```

This local deployment assumes `fastapi`, `uvicorn`, and `python-multipart` are already
installed in `Sai2608`. Set nonblank values in the repository-root `.env` for:

- `OPENAI_API_KEY`: your OpenAI API key.
- `OPENAI_CHAT_MODEL`: the chat model used by the existing pipeline.
- `OPENAI_EMBEDDING_MODEL`: the embedding model used by the existing pipeline.

Start the server in that terminal:

```bash
python -m uvicorn isc.api:app --app-dir src --host 127.0.0.1 --port 8000
```

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in a browser with JavaScript enabled.
2. Select one unencrypted PDF containing selectable text, enter a question, and click **Ask**.
3. Wait while **Processing...** appears. The page displays the answer and supplied source
   evidence, or a concise error message. The Ask button becomes available again when complete.

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) returns
`{"status": "ok"}`. It checks application liveness only; it does not call OpenAI or validate
model configuration. Stop the server with **Ctrl+C** in its terminal.

`GET /` serves plain HTML, CSS, and JavaScript. The browser posts multipart fields `file`
and `question` to `/ask`. FastAPI saves the PDF temporarily and calls the existing
`process_document()` pipeline, then returns its `PipelineResult` as JSON. The browser shows
the answer and existing evidence; structured Purchase Order fields remain available in the
JSON response. Temporary files are removed on completion or failure, with no saved history.

Submitting a valid document uses real OpenAI services and requires network access and valid
credentials. Automated API tests mock those calls. API failures use JSON `detail` messages:
400 for invalid input/unreadable PDFs, 422 for recognized no-text failures, 502 for OpenAI
failures, 504 for OpenAI timeouts, and 500 for unexpected failures. Development tracebacks
remain in the server terminal. This server is intended only for local PoC use.

## Test and lint

From the activated environment:

```bash
python -m pytest tests/test_api.py -q
make test
make lint
```

The current offline suite contains 382 tests, including 34 API/UI tests. The focused tests
cover `/`, `/health`, and mocked `/ask` success/error paths without real OpenAI calls.
`make lint` runs Ruff and strict MyPy.

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
  api.py
  chunking.py
  index.html
  llm.py
  models.py
  pdf.py
  pipeline.py
  retrieval.py

tests/
  test_api.py
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
- The FastAPI/browser interface is for local use, with no application-level upload-size
  limits, authentication, or production hardening.
- There is no hybrid BM25/RRF retrieval.
- There is no automatic retry or repair framework.
- Retrieval similarity is not factual confidence.
- Grounded answers are structurally validated against supplied source IDs, but their
  semantic citation support is not independently verified.
- The PoC is not production hardened.

## Validation status

The local deployment checkpoint has been validated with 382 offline tests, Ruff, strict
MyPy, and the exact Uvicorn command above with live HTTP checks of `/` and `/health`.
`/ask` is validated using mocks; no real OpenAI calls are made during these automated
checks. This does not verify live OpenAI credentials, model availability, or answer quality.

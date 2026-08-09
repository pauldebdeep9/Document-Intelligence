# Architecture

## Stage pipeline

```
ingest → parse → extract → index → retrieve → answer
                    ↓                  ↑
                  eval ────────────────┘
```

Stages are file-to-file. Each CLI verb reads artifacts from `runs/<run_id>/<stage>/`
and writes the next. This is slower than an in-memory pipeline and deliberately so:
every stage is independently re-runnable and inspectable, a failure is attributable
to one stage rather than to "the pipeline", and the eval harness gets cheap
re-execution without re-running upstream work.

## The four seams

| Seam | Module | Why it exists |
|---|---|---|
| Provider | `llm/ports.py` | OpenAI now, Azure OpenAI later, as a config change |
| Confidence | `common/confidence.py` | Routing, HITL and abstention all need decomposed signals |
| Permission | `models/acl.py` | Enforced at construction and pre-ranking, on both retrieval paths |
| Storage | `storage/ports.py` | Local now; Blob / Azure SQL / AI Search later |

## Data flow of a single field

```
PDF bytes
  → parse/     Block(text, bbox, ocr_confidence, layout_confidence)
  → extract/   PurchaseOrderRaw.po_number  (LLM, constrained JSON)
               + Signal.MODEL       from logprob
               + Signal.SCHEMA      discounted per repair round
               + Signal.OCR/LAYOUT  from the source block
               + Signal.LEXICAL     PO number pattern check
               + Signal.MASTER_DATA supplier master lookup
  → ExtractedField[str](value, Confidence, Span)
  → Thresholds.route() → accept | review | low_confidence | reject
  → review_queue row with the weakest factor attached
```

## Two-model extraction pattern

`<Type>Raw` is what the LLM is asked for: flat, plain types, nullable. Simple
schemas raise strict-mode compliance and cut repair rounds.

`<Type>` is what the pipeline stores: the same fields wrapped in `ExtractedField`,
with confidence computed by us. The model never sets its own confidence — a
self-reported certainty is not a signal worth auditing.

## Where P3 and P2 attach

`common/`, `llm/` and `models/` are the substrate that becomes the shared
`isc-core` package. P3 (shared GenAI platform) takes the tracing, cost accounting,
eval harness and guardrails. P2 (agentic automation) takes the extraction records —
three-way match needs `PurchaseOrder` and `Invoice` to share supplier, currency
and total, which is why `Invoice` was the second doc type rather than a
harder one.

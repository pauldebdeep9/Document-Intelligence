"""Aggregate questions answered from extracted records, not from chunks.

"What did we spend with Omron in total, in SGD?" asks for set selection and
arithmetic over several documents. The chunk path hands a model eight chunks
and asks it to do both; P1-09 measured 0/4 on exactly these questions, each
failure with the right documents already retrieved (docs/LIMITATIONS.md).
extract/ already produces a typed, confidence-scored PurchaseOrder per
document, so this path selects and sums those records in code and cites the
chunks each value was read from. A model is used only to classify the
question into a typed plan; it never sees a number and never writes one.

Order of operations (answerer.py):
  1. plan      -- model fills QueryPlanRaw; code validates it (plan.py) and
                  resolves the supplier mention against the master
                  (resolve.py). Anything that does not validate falls
                  through to the chunk path unchanged.
  2. source    -- the principal's permitted records, filtered BEFORE any
                  selection or arithmetic (source.py). There is no accessor
                  that returns an unfiltered record.
  3. execute   -- select, gate on confidence, sum with Decimal (execute.py).
  4. render    -- deterministic text with [n] markers over the evidence
                  chunks, then the same bind_citations()/verify_attribution()
                  every generated draft goes through (render.py).
"""

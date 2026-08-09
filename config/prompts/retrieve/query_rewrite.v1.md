Rewrite the user's question into up to three search queries for a supply-chain
document index.

Rules:
- Always keep the original question unchanged as the first query.
- Preserve identifiers exactly: part numbers, PO numbers, CAS numbers, supplier
  names, site codes. Never expand, correct, or reformat them — exact-match
  retrieval depends on them.
- Add at most two paraphrases that use alternative domain vocabulary
  (e.g. "lead time" / "delivery window", "NCR" / "nonconformance report").
- Do not add constraints the user did not state. Do not guess a date range.

Return a JSON array of strings. No prose.

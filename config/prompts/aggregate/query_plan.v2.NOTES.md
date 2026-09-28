# query_plan.v2 — notes (not loaded by anything)

## What changed vs v1

Exactly two additions; no other edits (`diff query_plan.v1.md query_plan.v2.md`):

1. A rule after the `"none"` bullet: choose `total_spend` / `part_prices` only
   when the answer *is* the summed totals or the list of prices; anything
   computed from them (average, minimum/maximum, count, ranking, comparison)
   routes to `"none"`.
2. One `"none"` example at the end, for a median over an invented part number.

## Why

v1 has no way to say "an aggregation the plan cannot express", so the model
picked the nearest operation. In the two v1 live runs
(`runs/run_20260928T114443Z`, `runs/run_20260928T114612Z`) **hw_17** was a
misroute-in to `part_prices` in both — the only non-pass case of 76. A records
answer there lists prices when the question asked for one statistic of them.

## Guarding against fitting to hw_17

Added to `data/gold/planner/handwritten.json` BEFORE this prompt existed
(commit 8d767a6):

- held-out, expected `"none"`: **hw_21** (minimum), **hw_22** (average,
  different wording), **hw_23** (maximum / ranking)
- controls, expected to stay aggregate: **hw_24** (a list request containing
  "all"), **hw_25** (combined spend)

The v2 example deliberately uses a statistic (median) and a part number that
appear in no case and nowhere in the corpus.

v1 baseline on the extended 81-case set: `runs/run_20260928T115301Z` —
misroute-in 3/81 (hw_17, hw_21, hw_22); hw_23 correct_none; hw_24, hw_25 exact.

## Results

v2 runs: `runs/run_20260928T115603Z` (A), `runs/run_20260928T115729Z` (B);
prompt sha256 9e0ab848…27ed0; A and B identical (0/81 unstable).

| set | v1 baseline | v2 A | v2 B |
|---|---|---|---|
| misroute-in | 3/81 (hw_17, hw_21, hw_22) | 0/81 | 0/81 |
| gold aggregate exact | 9/9 | 9/9 | 9/9 |
| gold none | 47/47 | 47/47 | 47/47 |
| paraphrase exact | 11/11 | 10/11 | 10/11 |
| near-miss | 6/6 | 6/6 | 6/6 |
| unsupported | 3/6 | 6/6 | 6/6 |

- Fixed: hw_17, hw_21, hw_22 (misroute-in in v1 → correct_none in both v2 runs).
  hw_23 was already correct_none under v1.
- Controls held: hw_24, hw_25 exact in both v2 runs.
- **Regression: hw_09** — exact under v1, misroute-out in both v2 runs (the
  model returned operation "none" while still copying the part number). A
  plain list-of-prices request is now read as a question about the prices,
  not the list: the new rule is over-applied to at least one paraphrase that
  is not a statistic.
- Truncation cases unchanged: hw_19 correct_none (guard-rejected, as in v1),
  hw_20 exact, not widened.
- Cost: +97 prompt tokens per call (627.8 → 724.8), $0.000106 → $0.000121.

**Verdict: NOT DONE.** Misroute-in is 0 in both runs, but the paraphrase
condition (11/11) fails on hw_09. No v3 was attempted (the AG-08 prompt 5
rule). v2 is the prompt `plan_question()` loads as of commit c12d584.

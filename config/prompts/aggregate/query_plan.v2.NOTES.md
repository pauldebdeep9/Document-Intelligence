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

(filled in after the two v2 live runs)

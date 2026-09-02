# Gold set changelog

## 1.1.0 — 2026-09-03

**What changed.** `po-004-q1` and `po-005-q1`'s question text, in `evals/gold/authoring.py`.
Both previously read the byte-identical `"What is the exact part number for the laser-cut
steel bracket on this purchase order?"`. They now read:

- `po-004-q1`: `"What is the exact part number for the laser-cut steel bracket on purchase
  order PO-1004?"`
- `po-005-q1`: `"What is the exact part number for the laser-cut steel bracket on purchase
  order PO-1005?"`

Anchors for both are unchanged (`"1. Part Number: 4500123456"` / `"1. Part Number:
4500123457"` respectively) — re-verified against real `extract_pdf_pages` output on both PDFs
as part of this change; they were already correct, only the question text was ambiguous.
`po-004-q2`/`po-005-q2` are untouched — they already name their distinguishing part numbers
and were not ambiguous.

**Why.** `evals/FINDING-002.md` established that `po-004-q1` and `po-005-q1` shared
byte-identical question text: nothing in the query distinguished which of the two documents
was meant, and at k=3 the two candidate chunks' scores differed by only 0.0019 — effectively a
coin flip. At least one of the two questions was unanswerable by construction: no ranker can
be right on both, so a hit on either was luck, not discrimination, and this was on a collision
course with the retrieval-scoping work planned for the next session — scoping `po-004-q1` to
`po-004` would have turned it into a guaranteed hit for a reason that has nothing to do with
retrieval quality, making the slice read 4/4 while measuring nothing.

**Which slice is affected.** `near_duplicate` only, and only half of it: 2 of its 4 questions
(`po-004-q1`, `po-005-q1`). `po-004-q2`/`po-005-q2` are unchanged. No other slice
(`header_field`, `line_item`, `cross_page`, `absent`) is touched. `goldset.version` bumped
`1.0.0` → `1.1.0`; `data/gold/goldset.json` regenerated via `python -m evals.gold.authoring`.

**Comparability.** `near_duplicate` numbers from any run against goldset_version `1.1.0` or
later are **not directly comparable** to `evals/BASELINE.md`'s `near_duplicate` numbers — the
question set itself changed, not the pipeline or the corpus PDFs. `evals/BASELINE.md`,
`evals/FINDING-001.md`, and `evals/FINDING-002.md` are unedited and remain accurate
descriptions of the runs they describe; they are not retroactively invalidated, only no longer
a like-for-like baseline for `near_duplicate` going forward.

The two saved run records this repository has produced so far — `beda2db6-b925-4201-8c27-
44d18c08e857` (k=3) and `d78f8dee-627f-4c0f-8cf0-6f27964dfc1d` (k=1) — both carry
`config.goldset_version: "1.0.0"` and therefore **predate this change**. Any future run's
`RunRecord.config.goldset_version` field is how to tell, mechanically, whether a given run's
`near_duplicate` numbers are comparable to those two or not: `"1.0.0"` means before this
change, `"1.1.0"` (or later) means after.

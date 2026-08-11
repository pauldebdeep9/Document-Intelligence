# ADR 0008: A rank-fused score cannot gate support, and similarity score doesn't separate on this corpus

**Status:** accepted · **Date:** 2026-08-11

## Context

`answer/orchestrator.py`'s `ask()` abstains `LOW_SUPPORT` when
`hits[0].score < settings.retrieval.min_support_score` (`0.35` at P1-06/07
scaffolding time, never exercised against a live score until P1-07's sample
run). `hits[0].score` is `retrieve()`'s output after `retrieve/fusion.py`'s
`rrf()` — a Reciprocal Rank Fusion score, not a similarity.

**RRF is rank-bounded by construction, not similarity-bounded.** For one
query producing two result lists (dense + lexical — `rewrite()`'s first
slice returns just `[question]`) and `rrf_k=60`, a chunk ranked #1 in *both*
lists scores `1/(60+1) + 1/(60+1) ≈ 0.0328`. That is the ceiling. It does
not fall as relevance falls — a chunk ranked #1 by both retrievers scores
the same `0.0328` whether the underlying match is exact or barely related.
Measured directly against all 56 P1-08 gold questions: top-hit RRF score
range is `0.0311`–`0.0328` for **both** answerable and unanswerable
questions — a band five times narrower than the `0.35` threshold, entirely
below it, and telling the two classes apart. No threshold on this field
expresses "weak support" at any value; `0.35` was never reachable, so
`LOW_SUPPORT` fired unconditionally on every query regardless of retrieval
quality (found running P1-07's first live sample: `dense_score=0.58` on the
top hit — a good match — scored `0.0328` and abstained anyway).

**Rescaling the threshold onto `dense_score` doesn't work either — checked,
not assumed.** Retrieved `retrieve()` for all 43 answerable/unanswerable
gold questions (their gold principal, no chat calls) and compared top-hit
`dense_score`:

| class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| answerable | 35 | 0.364 | 0.464 | 0.533 | 0.572 | 0.640 |
| unanswerable | 8 | 0.312 | 0.421 | 0.473 | 0.536 | 0.539 |

The distributions overlap substantially: **7 of 8 unanswerable questions
score above the lowest answerable score**, and the answerable 25th
percentile (`0.464`) sits *below* the unanswerable median (`0.473`). Any
cutoff in the crossover band (`~0.45`–`0.54`) that catches the low
unanswerable questions (`q_ua_08` `0.312`, `q_ua_07` `0.387`, `q_ua_05`
`0.421`) also abstains **9 genuinely answerable questions** — and they are
not scattered: **7 of the 9 are `line_item` questions** (`0.364`–`0.464`).
Table-row text scores systematically lower on dense similarity against a
natural-language query than header prose does, so a `dense_score` gate
would not just be imprecise — it would selectively fail one whole
answerable subtype while still letting `q_ua_01`/`02`/`03`/`06`
(`0.497`–`0.539`) through as "sufficient support." A threshold that is both
wrong in aggregate and systematically biased against one question type is
worse than no threshold.

## Decision

`retrieval.min_support_score = 0.0` (`config/default.yaml` and
`RetrievalSettings`'s own default) — the gate is disabled, not removed.
`ask()`'s `if hits[0].score < min_support_score` stays in the code exactly
as written; a non-negative score can never be below `0.0`, so the branch is
inert rather than deleted. Abstention on genuinely unanswerable questions
currently comes from `AbstentionReason.INSUFFICIENT_CONTEXT` — the model
reading the actual context and saying so — which the P1-07 sample run
already showed working (`q_ua_01`, `q_ua_04` both abstained correctly on
it; no score gate involved).

The code path is not dead: it becomes live again once a reranker sits
between fusion and generation. A reranker's score is built to express
relevance on a fixed, interpretable scale (unlike RRF's rank-only
arithmetic), and unlike raw `dense_score`, it is not a component of an
already-fused ranking decision — gating on it is not the same
category error `LOW_SUPPORT` on RRF was. That reranker is out of scope
here.

## Consequences

- `LOW_SUPPORT` will not fire in the current pipeline. Every abstention on
  an unanswerable question now depends on `_generate()` correctly producing
  `INSUFFICIENT_CONTEXT` — a single point of failure this ADR does not
  independently re-verify, only observes working on the P1-07 sample.
- `min_support_score`'s Python-level default and its `config/default.yaml`
  entry were both changed, not just the yaml, so `Settings()` constructed
  directly (as several unit tests do) reflects the same decision rather
  than silently disagreeing with the shipped config.
- No change to `retrieve/fusion.py`'s `rrf()` or to `hits[0].score`'s
  meaning elsewhere (RRF fusion is still the right choice for *ranking* —
  see `fusion.py`'s own docstring; this ADR is only about using its output
  as a support gate, a different question from whether it orders results
  well).

## Revisit if

- A reranker (P1-09+) is added between fusion and generation — re-derive
  `min_support_score` against its score, don't reuse `0.35` or any number
  from this measurement; neither was calibrated for that scale either.
- `rewrite()` stops returning a single query (query expansion ships) —
  RRF's ceiling rises with more lists (`n/(rrf_k+1)`), which changes the
  bound this ADR measured but does not, by itself, restore separation
  between answerable and unanswerable — re-measure before trusting it.
- `INSUFFICIENT_CONTEXT`'s reliability is ever measured directly (false
  negative rate: unanswerable questions where the model answers anyway
  instead of declining) and found weak enough that a second signal is
  worth the complexity this ADR just removed.

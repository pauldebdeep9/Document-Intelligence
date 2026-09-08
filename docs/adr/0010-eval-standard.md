# ADR 0010: The evaluation harness must be able to demonstrate its own results

**Status:** accepted · **Date:** 2026-09-08

## Context

The retrieval eval harness (`eval/retrieval.py`, `eval/report.py`) computed
correct numbers from P1-09 onward. It could not, until this item, show its
own work: no raw per-question record survived the process that produced it,
so nothing prevented a corrected metric definition from costing a live
re-run to re-score — only luck did, once (below); eight retrieval rates,
and a ninth once the README's headline `auto_accept_error_rate` joins them,
were emitted with no denominator anywhere in the file; the repo's single
hardest invariant ("one
leak fails the run regardless of every other metric") had never been tested
as a claim about the *run*, only as a claim about ACL primitives; and the
`--rescore-from` path built specifically to avoid a live re-run did not
persist the record a second rescore, or any comparison tool, would need.

The motivating incident for persistence: `abstention_precision`'s original
definition (P1-09) scored every legitimate denial — a `restricted` question's
non-gold-principal side, every `no_reader` principal — as "imprecise"
abstention, because it was scoped to `question_class == "unanswerable"`
alone. On the full 56-question run this put 22 of 29 abstained outcomes in
the wrong bucket and reported 0.241 for a run with 0 ACL leaks and a clean
restricted pass. The fix (`_expected_to_abstain()`, `retrieval.py:471-498`)
was verified cheaply that time — 65885bc's own commit message: "recomputed
on the same outcomes under the corrected definition," giving 0.862 with no
fresh model call — but only because the fix landed in the same working
session, while the run's outcomes were still available in memory to
recompute against. Nothing about that was guaranteed: `outcomes.jsonl` did
not exist yet, so a definition bug caught even one day later, after the
process that produced those outcomes had exited, would have had no record
to recompute against and would have cost a full live re-run of all 56
questions just to get a comparable number. EV-01 removes the dependency on
session luck: the raw record now survives the process, not only the session
it ran in.

### Provenance caveats

Three things this ADR's own evidence depends on, stated plainly rather than
left implicit:

1. `tests/fixtures/ev02_synthetic_old_format_combined_report.json` is
   fa5f2b0's `report.py`, actually executed, against the report classes
   (`ExtractionReport`/`RetrievalReport`) as they stood *after* 3a410f8. That
   combination — old renderer, new classes — never existed as a real state of
   this repo; it was constructed to demonstrate the 9th bare rate
   (`extraction.auto_accept_error_rate`) in one place. It carries 9 genuinely
   bare rates and does its job. It is not a historical artifact and must not
   be cited as one.
2. `eval/diff.py`'s composite mutation test
   (`test_composite_leak_and_verdict_and_retrieval_set_keeps_all_three_diffs`,
   `tests/unit/test_eval_diff.py`) proved the primary-class assignment by
   attacking *ordering* — reversing `_FIELD_PRIORITY` and watching the wrong
   class win. It did not independently prove *retention* — that a correctly
   computed primary tag still keeps every other field's diff underneath it.
   The assertion covers retention (`set(by_field) == {...}` checks all three
   survive); no mutation was constructed to attack retention specifically
   (e.g. one that drops `field_diffs` down to just the primary field). The
   gap is in mutation coverage, not in the assertion.
3. An earlier pass at this work asserted 148 outcomes for the full gold set,
   taken from a `grep`-style count of `answer.ask` spans in
   `runs/p1_09_full56/trace.jsonl`. `trace.jsonl` has exactly 148 of five span
   types — 2×74, not a fresh number. `Run` seeds its span counter and cost
   totals from whatever is already on disk for a run_id rather than zero
   (`tracing.py:43-56`) and writes `trace.jsonl` append-only (`:89`,
   `.open("a")`), specifically so a later stage sharing a run_id does not
   collide span ids with an earlier one — the same mechanism means two
   *evaluation* invocations sharing a run_id append into the same trace
   instead of colliding or erroring. `report.json` is plain `write_text()`,
   last-write-wins. Both are consistent with `p1_09_full56` having been
   invoked twice; neither is a log proving it was. The number itself (74, not
   148) was only caught because it was re-derived from `_principals_for()`
   (`retrieval.py:434-444`) against `data/gold/retrieval/questions.json`
   directly, rather than re-quoted from a trace grep — the transferable
   lesson is not the specific history of this one run, it is that a count
   taken from a trace survived several turns before a derivation from the
   code that actually determines it contradicted it.

## Decision

1. **`outcomes.jsonl` is the unit of record.** Every `QuestionOutcome`
   `RetrievalReport.outcomes` holds is persisted one-per-line, in run order
   (`eval/outcomes.py`'s `write()`), alongside `failed.jsonl` for questions
   that raised rather than scored. `report.json`/`report.md` are aggregates
   computed *from* that record, not the record itself — the aggregate is
   allowed to be wrong and re-derivable; the record is not allowed to be
   gone. Both artifacts carry their own schema version —
   `OUTCOMES_SCHEMA_VERSION` (`outcomes.py:58`, currently 1) and
   `REPORT_SCHEMA_VERSION` (`report.py:41`, currently 3) — each with its own
   mismatch error (`OutcomesSchemaMismatch`, `ReportSchemaMismatch`) that
   refuses to silently misread a shape it wasn't built for. This is what
   makes a corrected metric definition (the `abstention_precision` incident
   above) a `--rescore-from` invocation against already-captured data instead
   of a second live run against the model.

2. **No rate is emitted without its own denominator, enforced by a checker,
   not by convention.** `tests/unit/test_eval_report_rates.py`'s
   `find_bare_rates()` (`:115-122`, walking via `_walk()` at `:80-113`) scans
   every path in `report.json` for a float in `[0, 1]` whose immediate parent
   object has no sibling `"n"`, and fails the build if it finds one. Before
   this existed, 8 retrieval rates were bare (`recall@5`, `recall@8`, `mrr`,
   `ndcg@8`, `abstention_precision`, `abstention_recall`, and both
   `restricted.*.primary_recall@8`) — proven against a real, captured
   pre-change `report.json`, not a synthetic example
   (`test_old_pre_ev02_retrieval_report_has_exactly_8_bare_rates`). With
   extraction present, a 9th joins: `extraction.auto_accept_error_rate` — the
   README's headline number. 54 remain deliberately excluded and open (EV-06
   below), not fixed by this checker's presence.

3. **One ACL leak fails the run, regardless of every other metric — declared,
   not implied.** `GatePolicy` (`retrieval.py:335-347`: `name`, `passed`,
   `reason`) and `evaluate_acl_gate()` (`:350-375`) extract what
   `RetrievalReport.passed()` already decided (`acl_leaks == 0`) into a named
   verdict with a reason a reader can act on without cross-referencing
   `report.md`'s prose. `evaluate_acl_gate()` reads nothing off the report
   except `leaks()`/`leaks_by_subtype()` — the independence from every other
   metric is structural (there is no code path by which a metric value could
   reach the verdict), not a promise kept by convention. Proven by mutation
   in both directions, which is the actual content of "regardless of every
   other metric" and was not covered by any of the 534 tests that existed
   before this item: floor-metric outcomes (every retrieval metric at 0.0,
   zero leaks) still pass
   (`test_floor_metrics_zero_leaks_gate_passes`); perfect-metric outcomes
   (every retrieval metric at 1.0, one leak) still fail
   (`test_perfect_metrics_one_leak_gate_fails`) — both in
   `tests/unit/test_acl_gate_policy.py`.

4. **Small-n reporting: `k/n`, never a bare percentage.** The answerable set
   this corpus's gold measures against is 35 questions; per-subtype it goes
   as small as 2 (`underspecified`). At this scale a single flipped outcome
   moves a subtype's rate by tens of percentage points — a bare `0.500` reads
   identically whether it is 1/2 or 50/100. Every rate in `report.json` is
   its own `{"n": <denominator>, ...: <rate>}` object (rule 2); every rate
   reported in prose (this document, the review section, `report.md`) is
   stated as `k/n` first, the ratio second.

5. **Determinism: cache-replay end to end, not cold-cache variance.**
   `isc eval-diff ev01_gate_source ev04_gate_live_check` — two independent
   *live* invocations of the retrieval harness against the same 56-question
   gold set and the same index — returned zero differences across all 74
   outcomes. Both runs' `trace.jsonl` show zero `llm.complete` and zero
   `llm.embed` spans: every retrieval and generation call in
   `ev04_gate_live_check` was served from `llm/cache.py`'s `ResponseCache`
   rather than executed — the run exercised no inference at all, at either
   stage. What this demonstrates is cache-replay determinism end to end:
   given a fully warm cache, re-running the same questions reproduces the
   prior run's outcomes byte for byte, including free-text `answer_text`. It
   does **not** bound cold-cache variance, and it does not explain the
   0.833-vs-0.830 `mrr` spread reported elsewhere in this work's history,
   which came from a cold, clean clone — a configuration this comparison did
   not exercise. Cold-cache determinism is unmeasured (Open items).

   This result also exposes an ambiguity rather than resolving one:
   `complete()` and `embed()` (`llm/openai_client.py:63-67`, `:121-132`)
   both check the cache *before* entering their `span(...)` block and return
   early on a hit, so a cache hit produces no span of either kind. A trace
   showing zero `llm.complete`/`llm.embed` spans is consistent with "this run
   made no calls of this kind" and with "every call of this kind hit cache" —
   `trace.jsonl` alone cannot distinguish them. Every "zero live calls"
   statement in this work's own history (including this ADR's determinism
   claim above) is therefore a claim about span *absence*, not a claim about
   what was actually executed. Registered as an open item, not resolved
   here.

6. **What the byte-identical gate proves, and what it does not.** Rule 5's
   zero-difference result is evidence that `eval/diff.py`'s classification
   and `report.py`'s emission are a deterministic function of the persisted
   `QuestionOutcome` records — same input, same output, down to field order
   and free text. It is not evidence that the *numbers are right*: `mrr =
   0.833` reproducing exactly does not mean 0.833 is the correct value for
   this system's retrieval quality, only that computing it twice from the
   same evidence gives the same answer. Format determinism and metric
   correctness are different claims; this item established the first and
   says nothing new about the second.

7. **`outcomes.jsonl` is a cross-principal superset — evaluator-only, never
   shared.** One `outcomes.jsonl` holds every principal's retrieved chunk
   ids, citations, and answer text for a gold question, including outcomes
   that exist *precisely because* a principal must be denied — a
   `restricted` question's `principal_b` side, every `no_reader` principal
   checked. No single principal in the identity graph could legitimately
   read this file's own contents: reading it would itself be the disclosure
   the ACL model exists to prevent (`outcomes.py:36-43`,
   `EVALUATOR-ONLY ARTIFACT -- DO NOT SHARE`). `runs/` is gitignored, so this
   is not a live leak today, but it is not a redaction boundary either, and
   none is added by this item. This is the most generalisable observation in
   this work: any ACL-aware evaluation harness that persists per-principal
   retrieval data for scoring has this property by construction, and most
   will not notice it, because the file looks like an ordinary eval artifact
   until you ask who, specifically, is allowed to open it.

8. **The committed synthetic fixture exists because the real one cannot be
   committed.** `tests/fixtures/ev02_synthetic_outcomes/outcomes.jsonl` (5
   hand-built outcomes, including a deliberate leak) is committed so tests
   have a permanent, clone-safe input that does not depend on a live run or
   a gitignored artifact. The real `outcomes.jsonl` (74 outcomes, real
   principal ids, real retrieved chunk ids) cannot be committed for the same
   reason rule 7 states: it is a cross-principal superset, evaluator-only by
   its own construction, and committing it would be committing a document no
   single principal is entitled to read.

## Consequences

- `report.json` is larger and more heavily nested than its pre-EV-02 shape —
  every rate now carries its denominator as a sibling object rather than a
  bare float. This is a deliberate readability-over-compactness trade;
  `report.md`'s prose renders the same numbers more compactly for humans.
- A reader of `report.json` can now check `schema_version` before trusting
  its shape; `REPORT_SCHEMA_VERSION` has already moved once since it was
  introduced (2 → 3, for `acl_gate`) within the same week of work, which is
  a real cost future changes should weigh against how much shape churn is
  actually necessary.
- `--rescore-from` remains cheaper to run than a live re-run (no model,
  index, or docstore access), and is deliberately terminal: it never
  produces a run directory `eval-diff` can compare against anything — see
  the rescore-gap entry under Open items for the reasoning and what EV-07
  still owes it.
- The three provenance caveats above apply to any future citation of this
  item's own evidence — in particular, the synthetic old-format fixture must
  never be presented as something that once ran in production.

## Revisit if

- A cold-cache run is ever performed and its outcomes are diffed against a
  cached run — that is the measurement rule 5 explicitly does not make, and
  it either confirms or narrows what the 0.833-vs-0.830 spread can be
  attributed to.
- A second gate is ever added (a cost ceiling, a latency floor). One
  sentence, not built for here: it becomes a second `evaluate_x_gate()`
  returning its own `GatePolicy`, aggregated at the call site — not a change
  to this gate's shape.
- EV-07 (Open items) lands and a recorded source turns out not to be enough
  — a case shows up where knowing the source run_id isn't sufficient to
  answer the question someone actually had. Not anticipated; not a reason to
  build more than EV-07 scopes today.

## Open items

Unresolved by this item, recorded rather than fixed, per this item's own
constraint against fixing what it finds.

- **EV-06: extraction's remaining bare rates.** `extraction.by_field`'s PRF
  shape (`precision`/`recall`/`f1`) is deliberately excluded from
  `find_bare_rates()`'s checker (`test_eval_report_rates.py:85-100`) — not
  because it isn't a rate, but because fixing it means restructuring
  `metrics.py`'s shared `PRF` dataclass (`support` exists but isn't
  recognised as `n` by the checker's convention; `precision`/`f1`'s true
  denominators aren't present under any name), not a `report.py`
  presentation change. Freshly computed against the real 20-document corpus
  this session (not estimated): **18 fields × 3 rates = 54 bare rates**,
  currently present and excluded on purpose. This supersedes the "19-55"
  range in df755b9's commit message — that range was a rough estimate before
  the checker existed to count precisely; 54 is a direct count against
  `ExtractionReport.by_field()`'s real output, not a re-estimate.
- **The rescore gap is resolved: a rescore is terminal.**
  `eval_outcomes.write()` is called only in `cli.py`'s live-run branch
  (`:293-295`); `--rescore-from` (`:265-269`) does not call it. That is now
  the decision, not an accident awaiting one: a rescored run does not emit
  `outcomes.jsonl`. Its `report.json` instead records the source run it was
  scored from. On disk today, before this decision had a name: 6 run
  directories are exactly the old, accidental shape (`ev01_gate_rescore`,
  `ev02_gate_rebaseline_a`, `ev02_gate_rebaseline_b`, `ev02_gate_rescore_a`,
  `ev02_gate_rescore_b`, `ev03_gate_policy_check`) — a `report.json`/
  `report.md` with no `outcomes.jsonl` and no recorded source either, since
  the field EV-07 (below) adds does not exist yet.

  The reasoning: a rescore generates no new raw retrieval evidence — its
  `QuestionOutcome`s are, by construction, identical to its source's,
  re-aggregated through (possibly corrected) scoring logic. Emitting a copy
  of that evidence under the rescore's own run_id would create a second copy
  of a file that already exists, with no mechanism keeping the two
  consistent — if the source's `outcomes.jsonl` were ever regenerated, the
  copy would go stale silently. Exactly one `outcomes.jsonl` exists per set
  of observations, and every report scored from it, live or rescored, points
  back at that one file rather than at a copy of it.

  The cost, stated rather than glossed: `eval-diff` compares live runs only.
  A rescored run is invisible to it, permanently and by design — there will
  never be an `outcomes.jsonl` in its directory to diff. This is acceptable
  because two rescores of one source have identical observations by
  construction, so that diff would always report zero differences — it would
  test nothing `isc eval-diff <source> <source>` (EV-04's own identity case)
  doesn't already cover. The comparison someone would actually want in that
  situation — two scorings of the same outcomes under different metric
  definitions — is a `report.json` comparison, not an outcome comparison,
  and `git diff` (or any text diff) handles it adequately now that the
  emitted format is a deterministic function of the persisted outcomes
  (rule 6).

  **Filed as EV-07, three or four prompts, not urgent:**
  - `report.json` records the source as a `run_id`, not an absolute path (an
    absolute path breaks on a clone) and not anything that assumes the
    `runs/` layout directly — every other run-addressing convention in this
    codebase already resolves through `s.paths.runs / run_id`.
  - A reader encountering a recorded source that is absent fails loudly,
    rather than treating a missing source as "this must be an unrescored
    run" — those are different claims and must not be conflated by omission.
  - A test asserts the rescore path writes no `outcomes.jsonl`. Today that
    behaviour is an accident of where `eval_outcomes.write()` happens to sit
    — inside the live branch's `else`, not because anyone decided the
    rescore branch should skip it. Now that it is a decision, it needs an
    assertion, or someone will "fix" the asymmetry in six months believing
    they are closing a gap.
- **`GatePolicy` is unversioned.** `report.json` is at schema 3,
  `outcomes.jsonl` at 1, both with a mismatch error a reader can catch. The
  policy's own identity is a bare `name: str` field (`retrieval.py:335-347`).
  If `acl_leak_gate`'s rule ever changes (a second condition added, the
  reason wording changed materially), two `report.json`s from different code
  versions carry the identical name `"acl_leak_gate"` with different
  semantics, and nothing detects the mismatch the way `REPORT_SCHEMA_VERSION`
  detects a shape mismatch.
- **`acl_gate.reason` embeds an outcome count.** The pass-branch reason
  string (`retrieval.py:371-375`) interpolates `len(report.outcomes)` —
  `"no chunks leaked... (74 outcomes checked)"`. Two runs of different size
  (a gold set that grew, a `no_reader` question added) produce a textual
  diff in this field even when the decision content (`passed: true`, no
  leaks either time) is identical.
- **`citations_valid` is inert, yet persisted, diffed, and named as if it
  were a safety property.** `run()` never assigns it (`retrieval.py:638-655`
  — the `QuestionOutcome(...)` construction omits it, leaving the dataclass
  default `True`), but it round-trips through `outcomes.jsonl`
  (`outcomes.py`), and `eval/diff.py` raises `CitationsValidMismatch` naming
  the record if two files ever disagree on it. The name reads like it
  guards something; today it guards nothing that can vary between two real
  runs.
- **Tracing/cache ambiguity.** Per rule 5's own note: `complete()` and
  `embed()` both skip their `span(...)` on a cache hit
  (`llm/openai_client.py:63-67`, `:121-132`). No trace in this repo can be
  read as evidence of what was actually executed versus what was served from
  cache — the absence of a span type means "cache hit," not "uninstrumented"
  and not "no call was needed." This has consequences past cost accounting:
  any claim in this repo's history of the shape "zero live calls" (including
  this ADR's own determinism claim) is a claim about span absence, and a
  reviewer is right to ask, the first time they see it, how it is
  distinguished from "no calls were needed at all."
- **`summary.json`'s cost totals are unreliable, not merely absent.**
  `totals` reads `{}` in every `summary.json` examined this session
  (`ev01_gate_source`, `ev04_gate_live_check`, `p1_09_full56`,
  `run_20260811T081030Z`, `p1_04_verify_padding`) — including
  `p1_09_full56`, which has 3 real `llm.complete` spans in its own
  `trace.jsonl`. `add_total()` (`tracing.py:92-95`) is the only thing that
  populates `totals`, and it demonstrably fails to reach `summary.json` even
  on a run that made real, billed calls. Cost is not merely unrecorded on
  the runs this item touched; the recording mechanism itself is broken on at
  least one run where it should have fired. Not root-caused here.
- **Cold-cache determinism is unmeasured.** No run in this item's evidence
  disabled the cache. What rule 5 demonstrates is cache-replay determinism;
  whether two independent cold runs against the same gold set and index
  reproduce each other — which is the comparison that would actually bound
  the 0.833-vs-0.830 spread — has not been performed. A real, separately
  costed experiment, not attempted here.

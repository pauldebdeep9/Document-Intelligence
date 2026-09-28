# ADR 0011: Aggregate questions are answered from extracted records, not chunks

**Status:** proposed (off by default -- `aggregate.enabled: false` until the live eval below) · **Date:** 2026-09-28

## Context

P1-09 measured 0/4 on total-spend questions (`q_cd_01`..`04`) and 0/4 on the
ambiguous-entity questions, one of which (`q_am_04`) is also a total. Every
total-spend failure had the right documents in the retrieved set
(`cross_document` recall@8 0.875) and still produced a wrong figure, each in a
different way: a partial sum, two over-inclusive sums across up to six
suppliers, and a fabricated total that ADR 0009's attribution check could not
catch because the sentence named no supplier. The model was being asked to do
set selection and arithmetic over eight chunks.

Meanwhile `extract/` already produces a typed, confidence-scored
`PurchaseOrder` for every document -- supplier, currency, total and line
items, each with a decomposed `Confidence` and a `Span` -- stored in the
`records` table and never read at question time.

## Decision

A second answer path, `src/isc/aggregate/`, runs before retrieval in
`AnswerOrchestrator.ask()`:

1. **Plan.** A model fills a small typed `QueryPlanRaw` (`total_spend` |
   `part_prices` | `none`, plus supplier / part number / currency copied from
   the question). Not generated SQL: nothing to audit, nothing the ACL
   filter can be written around, and a plan is a value that can be logged
   and unit-tested. `validate_plan()` rejects any plan whose identifiers are
   not in the question verbatim (a planner completing "Kestrel Industrial" to
   "Kestrel Industrial AG", inventing a part number, or inferring SGD from a
   Singapore supplier), and resolves the supplier mention against the
   master. It also rejects a supplier mention the planner truncated (the
   next word in the question would extend it toward a longer master name),
   and accepts a currency only as a capitalised code in the question --
   found in the AG-02 review. A question asking for a statistic (average,
   minimum, count, ranking …, by a whole-word list) never gets an aggregate
   plan — the plan has no way to express one; AG-08 measured the planner
   choosing the nearest operation instead. Any rejection returns `None` and
   the chunk path runs unchanged.
2. **Query-side supplier resolution is a different policy from
   extraction-side.** A printed name wants exact-or-nothing; a typed mention
   wants every candidate. The extraction resolver maps "Kestrel Industrial"
   to AG alone (both normalise to "kestrel industrial");
   `aggregate/resolve.py` returns both, and each is reported separately,
   never summed together.
3. **Permissions before anything else.** `records` has no ACL column; ACL
   lives on `Document`. `visible_records(principal, ...)` checks
   `may_read(document.acl)` before a record is loaded -- there is no accessor
   that returns an unfiltered record -- and takes citation chunks from
   `LocalVectorStore.document_chunks(document_id, principal)`, which applies
   the same per-chunk check as search. An aggregate over hidden records
   would leak amounts through the sum and existence through the count, so
   the answer only ever says "visible to you", and a principal who can read
   none of the matching orders gets the same `NO_RESULTS` text as a supplier
   with no orders at all.
4. **Compute in code.** Selection by resolved `supplier_id` (with the
   printed-name fallback for the 9/20 documents with no vendor code), one
   group per (supplier, currency), `Decimal` arithmetic, no currency
   conversion.
5. **Confidence gate, not a new threshold.** A value's route is the weakest
   of the value and every field used to select it (supplier, currency, part
   number). `aggregate.contributing_routes` (default `accept`, `review`) says
   which existing extraction bands may contribute; review-band values are
   named as pending review, and anything excluded -- low confidence, no total
   printed, value not locatable in the indexed text -- is named with its
   reason, never silently dropped.
6. **Deterministic rendering, same grounding checks.** No model writes the
   answer. The text carries `[n]` markers over `supporting` (identity chunk +
   the chunk that prints the value) and goes through the same
   `bind_citations()` and `verify_attribution()` as a generated draft; one
   sentence per line so each PO's number is only vouched for by its own
   chunks. `Answer.route` records which path answered. Headlines state how
   many matching orders or lines the figure covers; exclusions and
   review-pending values are placed under the supplier they affect; review
   status is stated in plain language — the D7 review.

## Measured (offline, not a substitute for the live eval)

`tests/integration/test_aggregate_gold.py` builds a slice from the real
corpus with the extraction model scripted to return gold raw output (so
confidence, spans and routing are the real extractor's) and the planner
scripted to return each question's intended plan. As each question's gold
principal: **9/9** of `q_cd_01`..`08` and `q_am_04` pass
`answer_contains_gold()`, 0 supporting chunks fail `may_read()`, every answer
binds and verifies. This measures execute + render + ACL + citations. It does
**not** measure the planner.

Mutation check on the new tests: skipping the source's ACL check (12
failures), using the extraction resolver for mentions (9), dropping the
printed-name fallback (4), removing the confidence gate (2), dropping the
currency filter (1), dropping the verbatim-supplier guard (1), merging PO
lines into one sentence (1). An adversarial test mutates the answerer to
load records as a fully-cleared principal and asserts the leak gate fails.

## Found while building it

**The part-price gold is incomplete.** `q_cd_05`..`08` each list one pair of
lines, but the corpus has more: `PLC-1756-L83` is on 12 priced lines across
6 POs `u_alice` can read (gold lists 2); `TRM-BLK-2P5` on 20 lines / 5 POs
for alice and 11 lines / 4 POs for `u_ewan` (gold lists 2 each);
`ENC-INC-1024` on 3 lines for `u_chen` (gold lists 2). Recomputed from the
gold extraction files and ACL sidecars, independently of this code.
`answer_contains_gold()` only checks the listed pair is present, so it cannot
tell a complete answer from a two-line one -- the P1-09 score on these four
questions could not have detected an incomplete answer. Regenerate these
gold entries as "every visible priced line of this part".

## Consequences

- Statistic questions: a whole-word guard in validate_plan() backs the planner;
  phrasings outside the list are a measured known gap (docs/LIMITATIONS.md), bounded
  by the renderer showing a list, never a computed statistic.
- One extra small model call per question when enabled (the planner), on
  every question, including the ones that fall through.
- Misrouting is the new failure mode: a `single_hop` question answered from
  records, or an aggregate one falling through. The planner prompt uses
  invented supplier names and part numbers only, so the gold cannot leak
  into it. `aggregate.plan` spans record `route` and the rejection reason for
  every question; `QuestionOutcome` does not carry `route` yet (it would need
  an outcomes schema version bump -- EV follow-up).
- P1 scale: `visible_records()` walks every document per question. The
  Azure landing keeps the contract, not the loop: records in Azure SQL, the
  permitted set by a join on the principal's expanded ACL terms (optionally
  row-level security as a second enforcement point). AI Search facets count
  but cannot sum, so this does not belong in the index.

## Revisit if

- The live eval with `ISC_AGGREGATE__ENABLED=true` shows any misroute on the
  non-aggregate subtypes, or fewer than 9/9 on the aggregate ones -- fix the
  planner prompt against named failures, one change at a time (see
  `grounded_answer.v2.NOTES.md` for why).
- A second record type (Invoice) needs aggregating -- `source.py` raises
  `NotImplementedError` for anything but purchase orders by design.
- `contributing_routes` should drop `review`: only on a measured review-band
  error rate, not by default.

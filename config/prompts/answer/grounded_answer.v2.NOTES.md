Not loaded by anything -- load_prompt() sends prompt files to the model
verbatim, so this note lives beside grounded_answer.v2.md rather than inside
it. `isc.answer.orchestrator.AnswerOrchestrator._generate()` loads v1.

## Result: rejected. v1 remains live.

v2 added two rules on top of v1: (3) ask for clarification when the question
can't resolve to one document/part/order, (5) name and cite every distinct
entity a query matches, rather than picking one.

Measured against the same 11-question sample run for both:

- **q_am_01 (ambiguous, "What did we order from Kestrel Industrial?")** --
  regressed. v1 answered confidently from one of the two Kestrel entities
  (po_007), missing the other. v2 abstained entirely (`INSUFFICIENT_CONTEXT`)
  even though retrieval still returned both entities tied at rank 1,
  unchanged from v1. Rule 3 appears to have absorbed rule 5's case here --
  the model treated "which Kestrel did you mean" as needing clarification
  rather than as multiple entities to name.
- **q_ua_07 (underspecified, "What was the unit price?")** -- unchanged. No
  abstention, no clarification request under either version. v1 hedged
  across 3 conflicting prices from different POs; v2 hedged across 2. Rule 3
  did not fire on the exact question it was written for.

Two rules added together against one observation each, then tested together
-- can't tell from this whether rule 3 and rule 5 are individually sound and
only interact badly, or whether either one is wrong on its own. That
decomposition is P1-09's systematic comparison, not another single-prompt
guess from here.

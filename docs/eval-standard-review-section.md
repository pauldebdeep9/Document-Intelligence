# Making an evaluation harness prove its own results

This system's retrieval eval harness (`isc eval --harness retrieval`) scores
56 gold questions — asked as their own gold principal, never as an
unrestricted superuser — against a 20-document corpus, and gates the run on
one hard invariant: a single ACL leak fails it, regardless of every other
metric. The harness computed correct numbers from the start. Auditing it did
not find a wrong number. It found that the harness could not demonstrate, to
anyone other than the person who just ran it, that its numbers were what it
claimed, that its central invariant was actually enforced, or that its own
comparison tooling could be used twice in a row.

## What this review found

**Per-outcome data was never persisted, so a corrected metric definition's
safety depended on session luck, not on the harness.**
`abstention_precision`'s original definition scored every legitimate
denial — a restricted question's denied principal, every `no_reader`
principal checked — as imprecise abstention, because it was scoped to
unanswerable-class questions alone. On the full 56-question gold set this put
22 of 29 abstained outcomes in the wrong bucket and reported 0.241 for a run
with 0 ACL leaks and a clean restricted pass. That specific fix was verified
cheaply: recomputed against the same run's outcomes, still in memory in the
same working session, no fresh model call — 0.862, the number moved because
the definition was wrong, not because behavior had changed. But no raw
per-question record was written anywhere, so nothing forced that luck: a
definition bug caught the next day, in a different session, after that
process had exited, would have had nothing left to recompute against and
would have cost a full live re-run of all 56 questions just to get a
comparable number.

**Eight retrieval rates, and the extraction rate this project's own README
leads with, were emitted with no denominator anywhere in the file.**
`recall@5`, `recall@8`, `mrr`, `ndcg@8`, `abstention_precision`,
`abstention_recall`, and both `restricted.*.primary_recall@8` fields sat in
`report.json` as bare floats — no `n`, no count, nothing establishing whether
a given rate was 1/2 or 50/100 in a gold set where the smallest reported
subtype has exactly 2 questions in it. A ninth — `auto_accept_error_rate`,
the extraction accuracy figure this project's own README leads with — joins
the count the moment an extraction subtree is present. Proven against a real,
captured `report.json` from an actual 74-outcome run, not a synthetic
example: 8 bare rates, named individually, confirmed by a checker that walks
the emitted JSON and fails on any float in `[0, 1]` with no sibling
denominator. 54 more — every `precision`/`recall`/`f1` triple in the
extraction harness's per-field breakdown, 18 fields × 3 — are known,
currently bare, and deliberately deferred, not fixed, because fixing them
means restructuring a metrics dataclass shared across the codebase, not
relabeling a report.

**The system's central claim — one leak fails the run regardless of every
other metric — had zero test coverage.** Not weak coverage: zero, across
503 tests at the point this audit began and still zero across 534 by the
time the gate was made an explicit, named policy rather than an implicit
boolean. `tests/adversarial/acl/test_permission_boundaries.py` exists from
the repo's first commit specifically to guarantee ACL correctness never
regresses, and it does exactly that — at the primitive level: fail-closed
construction, group expansion, deny-beats-allow precedence, sensitivity
labels, retrieval-time filtering on both the dense and lexical paths. It
never once constructs a report and asks whether the *run-level gate*
actually enforces the claim printed next to it in every eval summary. The
claim that "regardless of every other metric" is not a tunable threshold but
a structural guarantee is exactly the part nothing tested: a run with every
retrieval metric at its floor and zero leaks was never asserted to pass; a
run with every metric at its ceiling and one leak was never asserted to
fail. Both are now declared as one named policy — an identifier, a verdict,
and a reason a reader can act on without parsing prose — and both are proven
by mutation: invert either check and a specific, named test goes red.

**The path built specifically to avoid a second live re-run does not persist
what its own comparison tool needs, so the chain it exists to enable
terminates after one hop.** `--rescore-from` re-scores a prior run's raw
outcomes against corrected logic without touching the model, the index, or
the corpus — the fix for the exact problem the first finding describes. But
the code path that writes it only runs on a live invocation; a rescored
run's directory gets an aggregate report and nothing else. The result: a
rescore can be produced cheaply, but a *second* rescore — or any outcome-level
comparison against it — cannot, because there is no raw record left in its
own directory to compare against. Six run directories produced by this
project's own gate-verification work are exactly this: an aggregate report,
and no way to ask what specifically it was computed from.

## The argument

None of these four findings is a defect in a metric. Every rate this harness
ever reported was arithmetically correct; the ACL gate's pass/fail logic was
correct before it had a name or a test; `--rescore-from` correctly re-scores
what it's given. All four are gaps in what the harness could show *about
itself* — whether a rate's denominator was visible, whether its hardest
invariant was demonstrated rather than assumed, whether a corrected
definition could be checked against evidence instead of a second live run,
whether its own comparison tooling worked more than once.

A harness that computes correct numbers and cannot demonstrate that it does
is not an evaluation standard. It is a number generator that happens, this
time, to be right. The standard this work established — persist the raw
record before aggregating it, never emit a rate without its own denominator,
declare and mutation-test the one invariant that overrides every other
metric, and build comparison tooling against the raw record rather than the
aggregate — is the difference between a harness whose output can be
independently checked and one whose output has to be taken on trust because
nothing underneath it survives the run that produced it.

# Finding 002: near_duplicate, header_field, and line_item misses classified by mechanism

**Status: post-hoc exploratory analysis, not a pre-registered finding.** Nothing in
`evals/PREREGISTRATION.md` §3 pre-registered this classification; this document exists to
answer a question `FINDING-001` opened but didn't settle — whether the near_duplicate misses
share `FINDING-001`'s "topic match" mechanism or are genuine sibling confusion. Per §5, kept
separate from `BASELINE.md` and `FINDING-001.md`, neither of which is edited here.
**near_duplicate is 4 questions.** header_field is 5, line_item is 5. All on a 24-question DEV
split. Nothing below generalizes past this corpus.

Entirely offline: both saved `RunRecord`s (`runs/beda2db6-....json` for k=3,
`runs/d78f8dee-....json` for k=1), the DEV goldset via `evals.gold.split.load_gold`, and the
real `evals.metrics.hit_at_k` (not a reimplementation) — no API call.

---

## Step 1 & 2: near_duplicate — every question, both runs, classified

Sibling map: po-004 ↔ po-005.

### k=3 (run `beda2db6-b925-4201-8c27-44d18c08e857`)

| question_id | gold_doc | question | retrieved (rank order, doc/score) | gold retrieved | sibling retrieved | hit_at_k | classification |
|---|---|---|---|---|---|---|---|
| po-004-q1 | po-004 | "What is the exact part number for the laser-cut steel bracket on this purchase order?" | po-005 (0.6478), po-004 (0.6459), po-010 (0.5775) | True | True | **True** | HIT (sibling ranked above gold, but both fit) |
| po-004-q2 | po-004 | "What quantity was ordered for part number 4500123456?" | po-001 (0.5264), po-005 (0.5198), po-004 (0.5195) | True | True | **True** | HIT |
| po-005-q1 | po-005 | "What is the exact part number for the laser-cut steel bracket on this purchase order?" | po-005 (0.6478), po-004 (0.6459), po-010 (0.5775) | True | True | **True** | HIT |
| po-005-q2 | po-005 | "What quantity was ordered for part number 4500123457?" | po-001 (0.5249), po-005 (0.5203), po-004 (0.5202) | True | True | **True** | HIT |

**0/4 missed.** All four are hits — but note po-004-q1 and po-005-q1's sibling is ranked
*above* gold in both cases; only k=3's extra slots make this a hit at all (see "what document
scoping would and would not fix" below).

### k=1 (run `d78f8dee-627f-4c0f-8cf0-6f27964dfc1d`)

| question_id | gold_doc | question | retrieved (rank order, doc/score) | gold retrieved | sibling retrieved | hit_at_k | classification |
|---|---|---|---|---|---|---|---|
| po-004-q1 | po-004 | "What is the exact part number for the laser-cut steel bracket on this purchase order?" | po-005 (0.6478) | False | True | **False** | **SIBLING DISPLACEMENT** |
| po-004-q2 | po-004 | "What quantity was ordered for part number 4500123456?" | po-001 (0.5264) | False | False | **False** | **TOPIC MATCH** |
| po-005-q1 | po-005 | "What is the exact part number for the laser-cut steel bracket on this purchase order?" | po-005 (0.6478) | True | False | **True** | HIT |
| po-005-q2 | po-005 | "What quantity was ordered for part number 4500123457?" | po-001 (0.5249) | False | False | **False** | **TOPIC MATCH** |

**3/4 missed.**

## Step 2: classification counts, near_duplicate

| Category | k=3 | k=1 |
|---|---|---|
| SIBLING CONFUSION (gold+sibling both retrieved, sibling ranked higher, still a miss) | 0/0 | **0/3** |
| SIBLING DISPLACEMENT (sibling retrieved, gold not) | 0/0 | **1/3** |
| TOPIC MATCH (neither gold nor sibling retrieved) | 0/0 | **2/3** |
| OTHER | 0/0 | 0/3 |

k=3 has zero misses to classify. Counts stated as `k/n` over misses; not rounded into one
blended claim, per the task.

**A structural note the counts alone don't show:** SIBLING CONFUSION, as strictly defined
(both gold and sibling retrieved, sibling ranked above gold, and it's still a miss), cannot be
observed at either tested `k` in this data. At k=3, whenever both siblings are retrieved
together, gold's own chunk is *also* in the window — so it's a hit by definition, regardless
of which one ranks first (see po-004-q1/po-005-q1 above, where the sibling outranks gold in
both directions and it's still a hit). At k=1, only one document can occupy the single slot,
so "both retrieved" is structurally impossible. The classification scheme's SIBLING CONFUSION
bucket is empty here not because sibling competition didn't happen, but because neither tested
`k` value can produce the specific pattern that bucket describes.

**po-004-q1's SIBLING DISPLACEMENT is a near-tie, not a clean loss.** The k=3 data for the
identical query shows po-005 at 0.6478 vs. po-004 at 0.6459 — a gap of 0.0019, against a
third-place candidate (po-010) trailing both by roughly 0.07. When k drops to 1, this
near-tie resolves in the sibling's favor and gold is displaced entirely. This is the closest
this dataset comes to demonstrating genuine sibling confusion, even though it lands in the
DISPLACEMENT bucket rather than the CONFUSION bucket by the letter of the definition — the
DISPLACEMENT/CONFUSION distinction collapses at k=1 for exactly the reason above, so treating
this case as "just displacement, unrelated to sibling confusion" would understate what the
scores show.

**po-004-q2 and po-005-q2 are unambiguous topic match.** Both siblings score *lower* than the
unrelated po-001 (0.5264/0.5249 for po-001 vs. 0.5195–0.5203 for the siblings, visible in the
k=3 row for the same queries) — po-001 isn't a near-miss third place, it's the actual winner
over both siblings combined. The near-duplicate-specific mechanism plays no role in these two
misses; an unrelated document beat both candidates that were supposed to be competing.

**A fact visible directly in the question text, not inferred:** po-004-q1 and po-005-q1 use
**byte-identical** question text ("What is the exact part number for the laser-cut steel
bracket on this purchase order?") — nothing in the query itself distinguishes which sibling is
meant; this pair is undecidable by text alone regardless of embedding quality. po-004-q2 and
po-005-q2 differ only in the literal part-number digit (`4500123456` vs. `4500123457`) — this
pair *does* carry a distinguishing token. But neither q2 case reached the sibling-discrimination
test at all at k=1: both lost to po-001 first. The question this class exists to answer — can
retrieval tell `...456` from `...457` apart — was never actually decided by either run; it was
preempted by an unrelated-document topic-match loss before the sibling comparison mattered.

---

## Step 3: header_field and line_item misses, same treatment

No sibling concept applies to these two classes (no near-duplicate pairing in either), so only
two categories apply: TOPIC MATCH (gold doc_id entirely absent from the retrieved window) or
something else, described.

### header_field

| question_id | gold_doc | k=3 retrieved doc_ids | k=3 hit | k=3 classification | k=1 retrieved doc_ids | k=1 hit | k=1 classification |
|---|---|---|---|---|---|---|---|
| po-001-q1 | po-001 | [po-001, po-004, po-010] | True | HIT | [po-001] | True | HIT |
| po-001-q2 | po-001 | [po-001, po-010, po-004] | True | HIT | [po-001] | True | HIT |
| po-002-q1 | po-002 | [po-005, po-004, po-002] | True | HIT | [po-005] | False | **TOPIC MATCH** |
| po-006-q5 | po-006 | [po-001, po-010, po-004] | False | **TOPIC MATCH** | [po-001] | False | **TOPIC MATCH** |
| po-010-q2 | po-010 | [po-005, po-004, po-002] | False | **TOPIC MATCH** | [po-005] | False | **TOPIC MATCH** |

Counts: k=3 **2/2 misses are TOPIC MATCH**. k=1 **3/3 misses are TOPIC MATCH**. Zero misses in
either run fall outside TOPIC MATCH.

### line_item

| question_id | gold_doc | k=3 retrieved doc_ids | k=3 hit | k=3 classification | k=1 retrieved doc_ids | k=1 hit | k=1 classification |
|---|---|---|---|---|---|---|---|
| po-001-q3 | po-001 | [po-010, po-006, po-001] | True | HIT | [po-010] | False | **TOPIC MATCH** |
| po-001-q4 | po-001 | [po-001, po-002, po-010] | True | HIT | [po-001] | True | HIT |
| po-002-q2 | po-002 | [po-002, po-001, po-010] | True | HIT | [po-002] | True | HIT |
| po-002-q5 | po-002 | [po-002, po-010, po-006] | True | HIT | [po-002] | True | HIT |
| po-010-q1 | po-010 | [po-006, po-010, po-010] | True | HIT | [po-006] | False | **TOPIC MATCH** |

Counts: k=3 **0/0 misses** (no misses to classify). k=1 **2/2 misses are TOPIC MATCH**. Zero
misses in either run fall outside TOPIC MATCH.

---

## What the records support, per slice

| Slice | k=3 miss mechanism | k=1 miss mechanism |
|---|---|---|
| absent (`FINDING-001`) | n/a (not retrieval-scored at k=3 vs k=1 differently — both runs 5/7 compliant, same contamination pattern) | TOPIC MATCH, 2/2 |
| header_field | TOPIC MATCH, 2/2 | TOPIC MATCH, 3/3 |
| line_item | no misses | TOPIC MATCH, 2/2 |
| near_duplicate | no misses | TOPIC MATCH 2/3, SIBLING DISPLACEMENT (near-tie) 1/3 |

**One mechanism — an unrelated document's chunk outscoring the gold document's own chunk —
accounts for every miss across absent, header_field, and line_item, and for 2 of
near_duplicate's 3 misses.** This is the same mechanism `FINDING-001` identified on the absent
slice, now confirmed to recur across three more slices, not something specific to absent
questions or to po-002.

**near_duplicate is the one slice with a genuinely distinct component.** 1 of its 3 misses
(po-004-q1) is not explained by an unrelated document winning — it's explained by the gold
document's own chunk narrowly losing to its near-duplicate sibling specifically, a mechanism
that does not appear anywhere else in this data.

## What document scoping would and would not fix

Document scoping — constraining the retrieval candidate pool to (or routing first to) the
question's own target document before ranking — presupposes the target document is knowable
independent of a flat, corpus-wide embedding comparison.

**It would fix every TOPIC MATCH miss recorded above** (absent's 2, header_field's 2–3,
line_item's 0–2, and near_duplicate's po-004-q2/po-005-q2): in each case, an unrelated
document — one that is not a near-duplicate of the gold document — won only because it was in
the same undifferentiated candidate pool. Remove documents outside the correct one (or the
correct small set) from contention, and none of these specific competitors would have had a
chance to win regardless of how close their embedding score was.

**It would not fix po-004-q1 on its own.** po-004 and po-005 are near-duplicates by
construction — the entire premise of this question class is that the two are difficult to
tell apart. Any document-scoping mechanism still has to decide, from the query, which of the
two is meant; if that decision is made using the same embedding that already produced a
0.0019 gap between them, the gap doesn't go away just because the candidate pool is narrower —
scoping only helps once it has correctly excluded the unrelated documents, and po-004 vs.
po-005 is precisely the comparison it cannot resolve by construction. Also worth restating
from above: po-004-q1's question text is byte-identical to po-005-q1's, so no signal in the
query itself picks one document over the other — this is not solvable by any mechanism that
operates on the query text alone, scoping included. Resolving it would need a signal scoping
doesn't provide — something with lexical/exact-match discriminating power on the differing
token (`4500123456` vs. `4500123457`), evaluated after candidates are already narrowed to just
the near-duplicate pair, not a broader corpus-wide ranking fix. Naming the category of fix,
not proposing an implementation: this is the kind of gap a lexical/exact-match signal (e.g.
BM25-style token matching) addresses and pure semantic embedding does not — consistent with
the "investigate BM25" trigger `BASELINE.md` §4 already fired on this same slice, for a
related but not identical reason (that trigger fired on the raw 1/4 hit@k number; this finding
adds that at least one of the underlying misses is specifically a near-tie between the two
candidates that matter, not noise).

## Determinism caveat

Both runs' retrieval scores for the *same* query, where directly comparable (e.g. po-004-q1's
and po-005-q1's retrieved candidates and scores at k=3), are identical between the two saved
records — expected, since retrieval scores derive from embeddings computed once at ingest, not
from the model call being classified. This is not a new determinism claim about extraction or
free-text answers; `FINDING-001`'s caveat about extraction being "one paired observation, not
evidence of determinism" still stands as written there and is not restated as settled here.

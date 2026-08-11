# Known limitations

Defects found during eval that are out of scope for the milestone they were
found in, logged here instead of fixed inline so they are not silently lost
or rediscovered from scratch later.

## P1 wrap-up (X-03): what this system does and does not do

Written at P1-10 close, from the P1-09 full retrieval run
(`runs/p1_09_full56/eval/report.md`, 56 gold questions, live index, no
tuning applied afterward) and the P1-03 extraction run. Numbers, not
adjectives — see those reports for the full breakdown.

### What it does

Ingest → parse → extract → chunk/index → retrieve → answer, for one
document type end to end: purchase orders. `Invoice` is a registered
schema (`models/records/invoice.py`) with no prompt and no gold set —
schema-complete, not operationally usable. Of `DocType`'s 8 defined
values, 2 have a record class; 1 has a corpus.

Extraction on that one type is solid: the P1-03/P1-04 measured error
surface on the 20-document corpus is close to empty (docs/adr/0007) — one
unrelated single-digit slip, caught cleanly at low confidence. That
number describes a corpus this system's own generator produced to be
readable; see the sample-size and corpus caveats below before extending
the claim anywhere else.

Retrieval + answering is real (dense + BM25 fused with RRF, citation
binding, attribution verification against `Chunk.filters`, per-principal
ACL enforcement pre-search) and was measured for the first time this
milestone, live, end to end, as each question's own gold principal.

### What it does not do

- **No OCR.** `parse/` is native-text only (`pypdf`'s text layer); a
  scanned or image-only PDF is out of scope entirely, not degraded.
- **No reranker.** `LOW_SUPPORT` (the intended gate between fusion and
  generation) is disabled — measured, not guessed: RRF's fused score is
  rank-bounded (~0.033 ceiling under current settings) regardless of
  relevance, and raw `dense_score` doesn't separate answerable from
  unanswerable questions on this corpus either (answerable p25 0.464
  sits *below* unanswerable median 0.473 — see ADR 0008). The code path
  is live again the day a reranker produces a score built to express
  relevance; nothing here should be read as "abstention doesn't need a
  support gate."
- **No query rewriting.** `Retriever.rewrite()` returns the question
  unchanged — every gold question was phrased plainly enough for
  dense+lexical search directly, so this was deferred, not found
  unnecessary in general.
- **Synthetic corpus, 20 documents, one supplier-confusable pair, one
  export-control case.** Generated, not collected — real SharePoint/ERP
  documents are messier in ways a generator tuned against its own known
  answers cannot produce by construction (inconsistent formatting,
  genuinely ambiguous OCR, real near-duplicate suppliers rather than one
  deliberately planted pair).

### What the P1-09 numbers do and do not support

**Sample size, stated before any figure below is read on its own:** 35
answerable questions over 20 documents. At this scale one flipped outcome
moves a per-subtype recall or accuracy figure by roughly 10–25 percentage
points. These numbers say *where* the system is weak; they are not a
tight estimate of *how* weak. Unlike the extraction corpus (measured to
near-zero error), retrieval has not been measured before this run — this
is a first reading, not a converged benchmark.

- **Cross-document aggregation is unreliable: 4 of 4 total-spend
  questions wrong**, each a distinct failure mode, all with the right
  documents in the retrieved set: `q_cd_01` summed only 1 of 2 documents'
  totals (partial sum); `q_cd_02`/`q_cd_04` summed totals from up to 6
  unrelated suppliers into one figure (over-inclusive sum); `q_cd_03`
  stated a "conflicting total" that traces to an unrelated document and
  does not appear in gold at all (fabricated figure). The system
  retrieves the right documents and reasons over them badly — recall@8
  for the `cross_document` subtype (n=8: 4 total-spend + 4 same-part
  price comparisons) is 0.875, well above its accuracy: 0/4 correct on
  the total-spend questions specifically.
- **Ambiguous entities: 4 of 4 wrong.** Retrieval surfaces both
  confusable Kestrel entities (tied for rank 1 in every case checked);
  the answer names one and omits the other, or (one case) abstains
  outright rather than surfacing either. `grounded_answer.v2.md` added an
  explicit rule for this and regressed the case it targeted (partial
  answer → full abstention) — see the file's own sibling `.NOTES.md` and
  ADR 0009. Not retried a third time this milestone; a real fix needs
  P1-09's systematic comparison, not another single-prompt guess.
- **Underspecified questions get a hedged direct answer, not a
  clarification request.** `abstention_by_subtype`: `absent` 100%,
  `out_of_scope` 100%, `underspecified` 0%. No value in
  `AbstentionReason` represents "asked for clarification" — there is
  nothing for a check to score as a pass even when the model declines,
  because declining alone isn't what gold asks for here.
- **`line_item` ranking, not recall: recall@8 = 1.000, MRR = 0.521.**
  Every gold chunk was retrieved; none ranked near the top. Table-row
  chunks score lower on dense similarity against a natural-language query
  than header prose does (ADR 0008) — present, but buried, which is
  exactly the failure mode a reranker (not currently built) exists to fix
  and `LOW_SUPPORT` (disabled, see above) cannot.
- **Attribution verification checks only named entities.** It resolves a
  supplier name or PO number stated in a sentence to `Chunk.filters` and
  checks the cited chunk agrees (ADR 0009). `q_cd_03`'s fabricated total
  named no supplier in the sentence making the claim, so there was no
  entity to check against, and it passed uncaught. The check catches
  misattribution; it does not catch an unattributed wrong number.
- **Fail-open on missing metadata has a measured cost.** 9 of 20
  documents have no extracted `supplier_id` (genuinely absent from the
  source — see `extract/masters.py`). `q_re_11`'s answer misattributed a
  Phoenix Contact GmbH line to "the SKF Bearings Manufacturing order";
  the cited chunk's document is one of the 9, so the mismatch had no
  `supplier_id` filter to check against and scored `unverifiable`, not
  `mismatch`. The design choice (fail open rather than abstain on missing
  metadata — ADR 0009) was made to avoid a worse problem, false positives
  on correct answers; this is that choice's real, now-observed price, not
  a hypothetical one.
- **`abstention_precision`'s original definition was a measurement bug,
  not a finding**, and is called out here rather than silently corrected:
  it scored an abstention as legitimate only when `question_class ==
  "unanswerable"`, so every one of ben's correct restricted-pair denials
  and every no_reader principal's correct denial counted as imprecision.
  Fixed to score legitimacy by expected behaviour instead (unanswerable,
  or the non-gold-principal side of any restricted/no_reader outcome):
  0.241 → 0.862 on the identical 29 abstained outcomes, same run, same
  behaviour, only the definition changed.
- **What worked cleanly, so it isn't lost in the list of what didn't:**
  `single_hop`/`header_lookup` recall and MRR both 1.000; `restricted`
  filtered pairs 8/8 clean (recall, answer accuracy, and ben correctly
  empty-and-abstained, all 8/8); **0 ACL leaks** across both restricted
  paths and `no_reader`, the one metric this milestone treats as a hard
  gate rather than a scored dimension.

## Wrapped table cells lose their continuation line — RESOLVED (P1-04)

**Found:** P1-03 eval run against the 20-document PO corpus (2026-08-10).
**Resolved:** P1-04, step 1 (2026-08-10). `parse/chain.py`'s
`NativeTextParser` now detects the line-item table structurally
(`_reconstruct_table()`, reusing P1-02's `ROW_PATTERN` row-shape rule --
moved there from `extract/spans.py` so both stages share one definition)
and folds a continuation line into its row by column position before
`Block.text`/`Document.text()` are ever built. Verified against gold: all
278 line descriptions across all 20 documents, including all 14 of the
original truncations, now appear complete in the reconstructed table text.
`extract_rows()` ordinal sequences still match gold exactly on all 20
documents -- the fix changes what's inside a row's cells, not how many rows
there are. Left below for the record of what the defect was and how it was
diagnosed.

`Document.text()` (parse/) does not stitch a table cell's wrapped
continuation back onto its row. When a line-item description overflows its
column width and wraps onto the next physical line inside the same cell --

```
20    PSU-24V-10A    Switched mode power supply 24V     1    EA    198.57 ...
                      10A
```

-- the parsed document text contains only the first line. The continuation
("10A") is dropped before extract/ ever sees it.

Confirmed against the source PDF directly (`pdftotext -layout`), not just
against gold: the wrapped line genuinely is on the page. Gold correctly
joined the two lines into one description; the parser did not. This is
**not** a model misread and **not** a gold error -- extract/ can only work
with what parse/ hands it, and by the time the LLM sees the text, "10A" is
already gone. The two-axis eval design (extraction vs normalisation)
correctly localises this to "the raw text itself is already wrong," but has
no way to further distinguish parse-stage loss from a genuine model drop --
both look identical from extract/ downward.

**Affected:** 14 of 2417 line-description comparisons in the P1-03 eval run
(`data/gold/extraction/po_008.json`, `po_004.json`, `po_000.json`), all the
same recurring part (`PSU-24V-10A`). Scored `wrong` on both the extraction
and normalisation axes at ~1.0 confidence -- a false negative on both: the
routing signals have no way to know the text was truncated before the model
ever ran, so there is nothing here for a lexical or agreement check to
disagree with.

**Fix location, as anticipated:** table/cell reconstruction in parse/'s
block extraction, recognising a continuation line (no leading ordinal, same
column position as the line above it) and merging it into the preceding
row before `Document.text()` is built -- not something extract/ or eval/
could have compensated for after the fact. See P1-04 in `docs/WBS-P1.md`
and ADR entries for the mechanism.

## Self-inflicted regression from the fix above, RESOLVED and verified same day

**Found:** measuring the extraction delta for the fix above (P1-04 step 2,
2026-08-10), before touching the chunker.

`_table_text()`'s first version rendered a reconstructed table by joining
cells with a flat two spaces (`"  ".join(row)`), discarding the original
printed table's column alignment. That is fine when every row's cells are
similar lengths; it is not when they are not -- `po_018.pdf` has
descriptions ranging from "Spare parts kit 4420" to "ControlLogix processor
module", so `Unit Price` and `Extended` landed at a different horizontal
offset on nearly every row, with no consistent gap between them anywhere.

Measured, not assumed: re-ran extraction on the real corpus (see P1-04 step
2's report) and `po_018.pdf` went from 8/8 `extended_price` values correct
(the pre-P1-04 run, reading the original wide-spaced text) to 8/8 missed,
all at confidence 1.0 (`ExtractedField.missing()`), with the flat-join
rendering -- despite the values being genuinely present and correctly
reconstructed. The model could no longer tell where one column ended and
the next began.

**Fixed:** `_table_text()` now pads every column to its own widest value
(header included) before joining, restoring consistent vertical alignment
across rows -- structurally the same information, formatted the way every
other table in the corpus already prints.

**Verified against a live model, not just structurally** ($0.0162, 20
documents, ~$0.0008/document): `po_018.pdf` extraction re-run for real
against the padded text -- 8/8 `extended_price` values correct. Structural
verification (values, row count, alignment present) was not sufficient to
catch this regression in the first place, precisely because it cannot
distinguish "correct information, unreadable layout" from "correct
information, readable layout" -- both diff identically against gold. See
`docs/adr/0007`'s "The flattening regression" section for the general
principle this leaves behind: a change that reformats text a model will
read needs a live-model check, not only a structural one.

The same live run also confirmed both genuine P1-03 extraction errors
(`po_008.pdf`'s dropped line, `po_010.pdf`'s price misattribution) are
gone -- see ADR 0007 for the full numbers. One new, unrelated single-value
error appeared (`po_010.pdf` line 20 `extended_price`, a 10x digit slip
against unambiguous source text) -- caught cleanly at confidence 0.02, not
a false negative, and reads as ordinary model call variance rather than
anything caused by this fix.

# Known limitations

Defects found during eval that are out of scope for the milestone they were
found in, logged here instead of fixed inline so they are not silently lost
or rediscovered from scratch later.

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

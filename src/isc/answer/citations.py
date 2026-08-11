"""Bind [n] citation markers in a generated draft back to the chunk ids
they claim to cite.

A marker is 1-indexed into `hits`, in exactly the order _generate() numbered
its context blocks -- see orchestrator.py's docstring on why `ask()` must
pass the SAME hits list to both calls, never a re-fetched or re-sorted one.
This module only ever reads hits by position; it never re-derives it.

A model can write a marker three shapes, and a real draft will produce all
three:
  - a bare [n]
  - adjacent markers, [1][3] -- each is already a complete bracket group on
    its own, so a regex matching one [n] at a time finds both
  - a single bracket holding a comma-separated group, [1, 3] -- a regex that
    only matches pure-digit bracket contents would not match this AT ALL,
    silently losing every citation in the group, not just narrowing it. The
    pattern below captures the whole digit list inside one bracket and
    splits it, rather than assuming one number per bracket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from isc.extract.validators import PO_NUMBER
from isc.models.answer import Citation
from isc.models.chunk import ScoredChunk

_MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")

# Same shape as retrieve/retriever.py's own _TOKEN -- not imported from
# there, since that one exists to scan a QUESTION for a filter to apply
# before search, and this one exists to scan a generated DRAFT for a claim
# to verify after generation. Same regex, different job; see
# verify_attribution()'s own docstring for why the two must not be
# conflated into one shared "the" token pattern.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

# Sentence boundary: a `.`/`!`/`?` followed by whitespace and a capital
# letter or an opening citation bracket. Deliberately not "any period" --
# this corpus's answers are full of decimal amounts ("392,589.57 SGD [1].
# The PO..."), and splitting on every '.' would cut "589" from "57" mid
# number. The lookahead for a capital/'[' is what keeps a decimal point from
# ever being read as a sentence boundary: nothing in this corpus's numbers
# is followed by whitespace then a capital letter or bracket.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

# A short digit run or a single capital letter, immediately preceded by
# whitespace/start-of-string and followed by '.' then whitespace, is a
# marker -- a numbered-list item ("1. Order Total...") or an initial ("A.
# Tan") -- not a sentence end, even though it is indistinguishable from one
# to _SENTENCE_BOUNDARY alone. Both were found live, not hypothesised: "The
# buyer contact...is A. Tan [1]." split into "...is A." (stranding the
# marker naming a PO number with no citation in the OTHER fragment) and
# "Tan [1]."; "...following orders:\n\n1. Order Total...[4]" split before
# "1.", stranding the supplier name with no citation while every per-order
# total landed, correctly cited, in later fragments. Protected via a
# placeholder rather than more lookbehind alternatives: Python's re requires
# fixed-width lookbehind, and "how much whitespace precedes this" is not
# fixed width, so a numbered-list marker after a blank line ("\n\n1.")
# cannot be excluded by lookbehind the way a single-space-separated initial
# can. Swap out, split, swap back -- the placeholder is never itself
# adjacent to whitespace+capital, so it can never introduce a new boundary.
_LIST_MARKER_OR_INITIAL = re.compile(r"(?<!\S)(\d{1,3}|[A-Z])\.(?=\s)")
_PROTECTED_PERIOD = "\x00"


def _split_sentences(draft: str) -> list[str]:
    protected = _LIST_MARKER_OR_INITIAL.sub(
        lambda m: m.group(0)[:-1] + _PROTECTED_PERIOD, draft,
    )
    return [s.replace(_PROTECTED_PERIOD, ".") for s in _SENTENCE_BOUNDARY.split(protected)]


def bind_citations(draft: str, hits: list[ScoredChunk]) -> list[Citation]:
    """Every marker resolves to hits[n-1] -- never a lookup, never a re-sort.

    Out-of-range markers ([9] against 8 blocks) are dropped, not an error: a
    model miscounting its own context is a generation defect the caller
    handles by abstaining when zero citations survive, not a reason to
    crash mid-answer. Repeats -- the same number cited twice, whether from
    two separate markers or from within one grouped marker -- collapse to a
    single Citation, in the order the number was first cited: that is the
    order a reader encounters the claims in the answer text, which is more
    useful here than retrieval rank.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _MARKER.finditer(draft):
        for token in match.group(1).split(","):
            n = int(token.strip())
            if n not in seen:
                seen.add(n)
                ordered.append(n)

    citations: list[Citation] = []
    for n in ordered:
        if not (1 <= n <= len(hits)):
            continue
        chunk = hits[n - 1].chunk
        citations.append(Citation(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            label=chunk.citation_label(),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        ))
    return citations


def _entities_in(sentence: str, supplier_ids: dict[str, str]) -> list[str]:
    """Supplier names (exact substring against the master list's keys -- the
    model is instructed to quote identifiers exactly, and a
    normalised/fuzzy match here would let a genuinely wrong supplier name
    slip through as "close enough") plus any token shaped like a PO number,
    reusing PO_NUMBER from extract/validators.py rather than a second copy
    of the pattern."""
    found = [name for name in supplier_ids if name in sentence]
    for token in _TOKEN.findall(sentence):
        if PO_NUMBER.match(token):
            found.append(token)
    return found


def _filter_key_for(entity: str, supplier_ids: dict[str, str]) -> tuple[str, str] | tuple[None, None]:
    """Which Chunk.filters key would carry this entity, and the value it
    would carry -- a supplier name resolves to its supplier_id, a PO number
    IS the filters value already. (None, None) means this entity kind has
    no chunk-level filter representation at all: today that never happens
    (both entity kinds _entities_in() extracts resolve), but ADR 0007
    already established that not every field can be one -- a table row's
    own part_number can't, because a chunk covers many rows each with a
    different one. If entity extraction is ever widened to a field like
    that, this is where it falls through to the text-only check below
    rather than crashing or silently skipping verification."""
    if entity in supplier_ids:
        return "supplier_id", supplier_ids[entity]
    if PO_NUMBER.match(entity):
        return "po_number", entity
    return None, None


@dataclass
class AttributionResult:
    """mismatches: real disagreements -- a cited chunk carries the filter
    key and it does not match. unverifiable: the entity's filter key is
    genuinely absent from every cited chunk (e.g. 2/20 documents in this
    corpus have no supplier_id at all), so there is nothing to check it
    against -- counted separately because it is not evidence of anything
    wrong, only evidence of a metadata gap in the source."""

    mismatches: list[str] = field(default_factory=list)
    unverifiable: list[str] = field(default_factory=list)


def verify_attribution(
    draft: str, hits: list[ScoredChunk], supplier_ids: dict[str, str],
) -> AttributionResult:
    """Binding (bind_citations(), above) checks that [n] resolves to a
    chunk that EXISTS. It cannot check that the sentence citing [n]
    describes what is IN that chunk -- a marker pointing at a real,
    permitted chunk is not evidence the prose next to it is true. This is
    the check binding cannot do: for every sentence, does every supplier
    name and PO number IT names actually belong to a chunk IT cites.

    Checked against Chunk.filters (supplier_id, po_number), not raw
    chunk.text -- text matching asks "does this chunk mention Omron",
    which a table or footer chunk structurally never does (chunker.py never
    repeats a document's header fields into its table/prose body chunks).
    filters asks "does this chunk belong to an Omron order", which is the
    question that actually matters and is answered correctly regardless of
    chunk type, since every chunk of a document carries the same filters
    dict. Measured on the P1-07 sample before this change: 3 of 4 flagged
    answers were exactly this -- a correct citation to a table/footer chunk,
    rejected because the entity named in the sentence (restating context
    from the question) was never going to be in that chunk's text no matter
    how correct the answer was. See docs/adr/0009.

    Per-sentence, not per-answer: a chunk cited three sentences ago does not
    license a claim in this one just because both appear somewhere in the
    same draft. This is deliberately the stricter, more naive granularity,
    not a considered final design -- it will still false-positive on a
    sentence that legitimately continues a subject named in a previous
    sentence without re-citing it. Measuring the false-positive rate this
    granularity produces on a real sample is the point of running it before
    deciding whether per-answer (or something between the two) is the right
    unit -- see docs/adr/0009's own numbers.

    Any single mismatch means the WHOLE answer is discarded by the caller,
    not just the offending sentence -- a draft that misattributes one fact
    is not trustworthy on the others it happened to get right.
    unverifiable entities do not block the answer.
    """
    result = AttributionResult()
    for sentence in _split_sentences(draft):
        cited_numbers = {
            int(token.strip())
            for match in _MARKER.finditer(sentence)
            for token in match.group(1).split(",")
        }
        cited_chunks = [hits[n - 1].chunk for n in cited_numbers if 1 <= n <= len(hits)]
        cited_texts = [c.text for c in cited_chunks]

        for entity in _entities_in(sentence, supplier_ids):
            filter_key, filter_value = _filter_key_for(entity, supplier_ids)
            if filter_key is None:
                # No filter represents this entity kind at all (see
                # _filter_key_for) -- the only case that still falls back
                # to raw text.
                if not any(entity in text for text in cited_texts):
                    result.mismatches.append(
                        f"{entity!r} in {sentence.strip()!r} not found in its cited chunk(s)"
                    )
                continue

            carriers = [c for c in cited_chunks if filter_key in c.filters]
            if any(c.filters[filter_key] == filter_value for c in carriers):
                continue  # verified via filters
            if not cited_chunks:
                # Nothing cited at all -- a genuinely uncited claim, not a
                # metadata gap. Must still fail: this is the known
                # per-sentence-granularity false positive (a claim that
                # legitimately continues an earlier citation without
                # re-citing it), not the missing-metadata gap below.
                result.mismatches.append(
                    f"{entity!r} in {sentence.strip()!r} cites no chunk at all"
                )
            elif carriers:
                # A cited chunk DOES carry this filter key, and disagrees --
                # a real mismatch, not a metadata gap.
                result.mismatches.append(
                    f"{entity!r} in {sentence.strip()!r}: no cited chunk's {filter_key} "
                    f"filter matches {filter_value!r}"
                )
            else:
                # Chunks were cited, but none carry this filter key at all --
                # cannot be verified either way. Treat as pass, not fail:
                # failing on missing metadata would abstain on a correct
                # answer because a field was absent from the source PDF.
                result.unverifiable.append(
                    f"{entity!r} in {sentence.strip()!r}: none of its cited chunk(s) carry a "
                    f"{filter_key!r} filter -- genuinely absent from the source, not checkable"
                )
    return result

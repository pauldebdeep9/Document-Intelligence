"""Locate a raw extracted value in the parsed Document and return a Span.

String search, not layout analysis -- pypdf gives no bounding boxes, so this is
the only provenance mechanism available. A hit means "this text appears in
block X on page Y"; nothing stronger. A miss returns None, never the nearest
block: a wrong span sends a human reviewer to the wrong place on the document,
which is worse than sending them nowhere. See docs/adr/0005 for the tradeoffs
this accepts and what a miss means downstream.

Two-tier search space, because a document-wide search space is too coarse for
line-item fields:

  locate(doc, value)               unscoped -- searches every block on every page
  locate(doc, value, scope=line)   scoped -- searches only within `scope`

`extract_rows()` finds line-item rows *positionally*, not by content. An
earlier version anchored each row on its part_number, on the theory that a
part number is distinctive within its own row -- measured against the real
corpus, it was distinctive in only 18% of line items (50/278), because the
parts master has ~10 distinct parts shared across documents with up to 42
lines, so repetition is the norm on exactly the documents most worth scoping.
extract_rows() instead matches the structural shape of a printed row --
leading ordinal, then a token -- which does not depend on how many times a
part number repeats. The caller (extract/extractor.py) is responsible for
verifying the extracted rows actually line up with the parsed record before
trusting them as scopes; see its row-alignment self-check.

Multi-match guard, applied to every search regardless of scope: if a value
matches more than once in its search space, that is not a located value, it is
an ambiguous one, and this does not return a Span -- see SpanOutcome below.
Row scoping narrows the search space a great deal but does not eliminate this
on its own: a quantity of '1' still collides with the leading digit of a price
like '1,801.47', because a comma is not a word character, so the same row can
still contain more than one boundary-safe hit for '1'. The guard is what
actually keeps the promise; scoping just makes the guard fire less often.

Zero matches and multiple matches are opposite claims, not the same "miss":
zero means the value may not be in the document at all -- the cheapest
hallucination signal available. Multiple means the value IS present, just not
uniquely locatable, which says nothing about whether it is correct. Collapsing
both into a bare `None` return would force every caller to either ignore the
distinction or reconstruct it by re-running the search, so locate() returns a
`Located` result carrying both the Span (when found) and a `SpanOutcome` that
says which case happened. Confidence assembly (extractor.py) penalises
NOT_FOUND only.

Zero *contiguous* matches is itself not one claim. A wrapped table cell (a
line-item description whose continuation prints on the next physical line, so
the row's other columns end up sitting between the two pieces once the block
is flattened to text) produces a value that is genuinely, fully present --
every one of its tokens is in the search space -- but no contiguous substring
of it exists to find. Scoring that as NOT_FOUND makes PROVENANCE a signal
about this parser's flattening, not about the model: measured on the real
corpus, 8 of 12 reject-routed fields in one run were this exact case, all
correctly extracted. FRAGMENTED is the honest name for it: not found
contiguously, but every whitespace-separated token of the value (at least 2 of
them -- a single token that fails a contiguous match is just NOT_FOUND, there
is nothing left to call fragmented) appears somewhere in the same search
space. Treated as neutral, exactly like AMBIGUOUS: presence of all the pieces
says nothing about whether they were assembled correctly, so it must not
corroborate either. Only genuine absence -- no contiguous match, and not even
the tokens are all there -- keeps the penalty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from isc.models.document import Document, Span
from isc.parse.chain import ROW_PATTERN

_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


class SpanOutcome(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"      # zero matches, tokens absent too -- may not be in the document
    AMBIGUOUS = "ambiguous"       # 2+ matches -- present, just not uniquely locatable
    FRAGMENTED = "fragmented"     # no contiguous match, but every token is present


@dataclass(frozen=True, slots=True)
class Located:
    """Result of locate(). `span` is populated only when outcome is FOUND --
    check `outcome`, not truthiness of `span` alone, or the four outcomes
    collapse back into indistinguishable "missing" cases."""

    span: Span | None
    outcome: SpanOutcome

    @classmethod
    def found(cls, span: Span) -> Located:
        return cls(span, SpanOutcome.FOUND)

    @classmethod
    def not_found(cls) -> Located:
        return cls(None, SpanOutcome.NOT_FOUND)

    @classmethod
    def ambiguous(cls) -> Located:
        return cls(None, SpanOutcome.AMBIGUOUS)

    @classmethod
    def fragmented(cls) -> Located:
        return cls(None, SpanOutcome.FRAGMENTED)


def locate(doc: Document, value: object, scope: str | None = None) -> Located:
    """`value` is the raw value as extracted (a string for most fields; str()
    is applied for anything else, e.g. a line's quantity/unit_price floats).

    Whitespace is normalised on both sides before matching, because layout-mode
    text pads header fields with runs of spaces and wraps long cells across
    lines. Numbers additionally get thousands-separated and alternate-decimal-
    precision search forms: the extraction prompt instructs the model to
    return '1536.37' while the page prints '1,536.37', and a value the model
    returns as a Python float loses trailing zeros str() had no reason to keep
    -- a quantity of 250.0 never prints as '250.0' on the page, and a price of
    11,572.40 round-trips through float as '11572.4'.

    If `scope` is given, the search space is that text alone (typically one
    physical line of a line-item row) rather than the whole document.
    """
    needle = _needle(value)
    if needle is None:
        return Located.not_found()

    if scope is not None:
        haystack = _normalise_ws(scope)
        n = _count_matches(needle, haystack)
        if n == 0:
            return Located.fragmented() if _is_fragmented(needle, haystack) else Located.not_found()
        if n > 1:
            return Located.ambiguous()
        page_number = _page_containing(doc, scope)
        if page_number is None:
            return Located.not_found()
        return Located.found(Span(document_id=doc.id, page=page_number, text=needle))

    total = 0
    hit_page: int | None = None
    fragmented = False
    for page in doc.pages:
        for block in page.blocks:
            haystack = _normalise_ws(block.text)
            if not haystack:
                continue
            n = _count_matches(needle, haystack)
            if n:
                total += n
                hit_page = page.number
            elif not fragmented and _is_fragmented(needle, haystack):
                fragmented = True
    if total == 0:
        return Located.fragmented() if fragmented else Located.not_found()
    if total > 1:
        return Located.ambiguous()
    assert hit_page is not None  # total == 1 implies the loop set it exactly once
    return Located.found(Span(document_id=doc.id, page=hit_page, text=needle))


def extract_rows(doc: Document) -> list[tuple[int, str]]:
    """Structural row extraction: a physical line beginning with an ordinal
    (the printed 'Item' number) followed by a separate token. This is shape,
    not content -- it self-excludes both ways a line-item block's other lines
    could otherwise be mistaken for a row:

      * the table header ("Item  Part Number  Description ...") starts with a
        word, not a digit, so `(\\d+)` fails at position 0
      * a wrapped description continuation (e.g. '                    10A')
        runs the digits straight into the next token with no space between
        them, so `\\s+` after the ordinal fails to match

    Returns (ordinal, full line text) in document reading order. Purely
    structural -- callers must verify the result actually aligns with the
    record they are scoping before trusting it; this function does not know
    what a "correct" row count or ordinal sequence looks like.
    """
    rows: list[tuple[int, str]] = []
    for page in doc.pages:
        for block in page.blocks:
            for line in block.text.split("\n"):
                m = ROW_PATTERN.match(line)
                if m:
                    rows.append((int(m.group(1)), line))
    return rows


def _needle(value: object) -> str | None:
    if value is None:
        return None
    needle = _normalise_ws(str(value))
    return needle or None


def _normalise_ws(s: str) -> str:
    return " ".join(s.split())


def _count_matches(needle: str, haystack: str) -> int:
    """Plain substring count for text. Numbers count boundary-safe matches
    across every plausible printed form -- literal, thousands-grouped, and
    (for whole numbers) alternate decimal precision -- so a quantity of '1'
    cannot match as a fragment of '18' or a year like '2025'. A comma is not a
    word character, so this still does not stop '1' from matching the leading
    digit of '1,801.47' -- that ambiguity is what the caller's != 1 check
    catches.

    Forms are combined into one alternation, longest first, and matched in a
    single pass rather than summed per form: '500' is itself a boundary-safe
    match of the leading digits of '500.00' (the '.' is not a word character),
    so counting each form's re.findall() separately would double-count one
    real occurrence as two."""
    if not _NUMERIC.match(needle):
        return haystack.count(needle)
    forms = {needle, *_precision_forms(needle)}
    forms |= {grouped for form in forms for grouped in _grouped_forms(form)}
    ordered = sorted(forms, key=len, reverse=True)
    alternation = "|".join(re.escape(f) for f in ordered)
    return len(re.findall(r"(?<!\w)(?:" + alternation + r")(?!\w)", haystack))


def _precision_forms(needle: str) -> set[str]:
    """Alternate decimal precisions for a numeric value, keyed off the value
    itself rather than the string, because str(float) drops trailing zeros a
    fixed-precision source never had reason to drop. A whole number prints as
    '250.0' or '500.0' -- never '250' or '500.00'. A price of exactly
    11,572.40 round-trips through float as '11572.4' -- never the '.40' the
    page actually shows, because 11572.4 == 11572.40 as floats and str()
    picks the shorter form. The corpus prints quantities with no decimal
    point at all and prices at a fixed two decimal places, so both are worth
    generating here; whichever one does not match anything is inert."""
    value = float(needle)
    forms = {f"{value:.2f}"}
    if value == int(value):
        forms.add(str(int(value)))
    forms.discard(needle)
    return forms


def _grouped_forms(needle: str) -> list[str]:
    """Thousands-separated variant of a plain numeric string, e.g.
    '1536.37' -> '1,536.37'. Empty if grouping would not change anything
    (values under 1000)."""
    sign = "-" if needle.startswith("-") else ""
    body = needle[1:] if sign else needle
    whole, _, frac = body.partition(".")
    grouped = f"{sign}{int(whole):,}" + (f".{frac}" if frac else "")
    return [grouped] if grouped != needle else []


def _is_fragmented(needle: str, haystack: str) -> bool:
    """True if every whitespace-separated token of `needle` is present
    somewhere in `haystack`, even though no contiguous match of the whole
    value exists. Requires at least 2 tokens: a single-token value that fails
    a contiguous match has nothing left to call "fragmented" -- it is just
    not there. Each token is checked with the same matching rules as a full
    value (_count_matches), so a numeric token still gets boundary-safe
    treatment rather than a naive substring check."""
    tokens = needle.split(" ")
    if len(tokens) < 2:
        return False
    return all(_count_matches(t, haystack) > 0 for t in tokens)


def _page_containing(doc: Document, scope: str) -> int | None:
    needle = _normalise_ws(scope)
    for page in doc.pages:
        for block in page.blocks:
            if needle in _normalise_ws(block.text):
                return page.number
    return None

"""Unit tests for answer/orchestrator.py and answer/citations.py (P1-07).

FakeChat/FakeRetriever stand in for a live provider and a live index -- same
reasoning as test_index_pipeline.py's FakeEmbedder: no network calls, and
scripted enough to pin exact behaviour rather than plausible behaviour.
"""

from __future__ import annotations

from isc.answer.citations import bind_citations, verify_attribution
from isc.answer.orchestrator import AnswerOrchestrator
from isc.common.config import Settings
from isc.llm.ports import LLMResult, Message
from isc.models.acl import AclSet, Principal
from isc.models.answer import AbstentionReason
from isc.models.chunk import Chunk, ScoredChunk


class FakeChat:
    """One scripted response; records every call it received so a test can
    assert on the exact prompt built, or that it was never called at all."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        return LLMResult(text=self.text, model="fake")


class FakeRetriever:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits

    def retrieve(self, question: str, principal: Principal) -> list[ScoredChunk]:
        return self.hits


def _chunk(cid: str, text: str = "some chunk text", **kw) -> Chunk:
    return Chunk(
        id=cid, document_id=f"doc_{cid}", ordinal=0, text=text,
        acl=AclSet(allow_terms=frozenset({"everyone:*"})),
        **kw,
    )


def _hit(cid: str, score: float = 0.9, **kw) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid, **kw), score=score)


def _principal() -> Principal:
    return Principal(id="u_test")


def _orchestrator(
    chat: FakeChat, hits: list[ScoredChunk] | None = None, settings: Settings | None = None,
) -> AnswerOrchestrator:
    return AnswerOrchestrator(
        retriever=FakeRetriever(hits or []), chat=chat, settings=settings or Settings(),
    )


# -- _generate(): block numbering and citation labels ------------------------

def test_generate_numbers_blocks_in_hits_order_with_citation_labels():
    hits = [
        _hit("chk_a", page_start=1, page_end=1),
        _hit("chk_b", page_start=2, page_end=2, is_table=True, line_range=(350, 420)),
    ]
    chat = FakeChat(text="draft")
    orch = _orchestrator(chat, hits)

    orch._generate("What are the terms?", hits)

    user_msg = chat.calls[0][1].content
    assert "[1] doc_chk_a p.1" in user_msg
    assert "[2] doc_chk_b p.2 (rows 350-420)" in user_msg
    # block order in the prompt follows hits' own order, not a re-sort
    assert user_msg.index("[1]") < user_msg.index("[2]")


# -- _bind_citations() / citations.py ----------------------------------------

def test_bind_citations_maps_marker_n_to_hits_n_minus_1():
    """The specific regression this guards against: a marker resolving to
    *some* chunk is not enough -- it has to be the RIGHT one, position for
    position."""
    hits = [_hit("chk_a"), _hit("chk_b"), _hit("chk_c")]
    citations = bind_citations("The value is X [2].", hits)
    assert len(citations) == 1
    assert citations[0].chunk_id == hits[1].chunk.id
    assert citations[0].chunk_id != hits[0].chunk.id


def test_bind_citations_drops_out_of_range_markers():
    hits = [_hit("chk_a"), _hit("chk_b")]
    citations = bind_citations("See [1] and also [9], which does not exist.", hits)
    assert [c.chunk_id for c in citations] == ["chk_a"]


def test_bind_citations_returns_empty_when_every_marker_is_out_of_range():
    hits = [_hit("chk_a")]
    assert bind_citations("Nothing here [9].", hits) == []


def test_bind_citations_dedupes_a_marker_cited_twice():
    hits = [_hit("chk_a"), _hit("chk_b")]
    citations = bind_citations("First claim [1]. Second claim also [1].", hits)
    assert [c.chunk_id for c in citations] == ["chk_a"]


def test_bind_citations_parses_adjacent_bracket_markers():
    """[1][3] is two complete bracket groups back to back, not one -- both
    must resolve."""
    hits = [_hit("chk_a"), _hit("chk_b"), _hit("chk_c")]
    citations = bind_citations("Combined claim [1][3].", hits)
    assert [c.chunk_id for c in citations] == ["chk_a", "chk_c"]


def test_bind_citations_parses_comma_grouped_markers():
    """A regex matching only a bare single [n] would not match [1, 3] at
    all -- both citations lost, not just one."""
    hits = [_hit("chk_a"), _hit("chk_b"), _hit("chk_c")]
    citations = bind_citations("Combined claim [1, 3].", hits)
    assert [c.chunk_id for c in citations] == ["chk_a", "chk_c"]


def test_bind_citations_orders_by_first_appearance_not_by_number():
    hits = [_hit("chk_a"), _hit("chk_b"), _hit("chk_c")]
    citations = bind_citations("Later fact [3] first, earlier fact [1] second.", hits)
    assert [c.chunk_id for c in citations] == ["chk_c", "chk_a"]


def test_bind_citations_label_and_document_id_come_from_the_resolved_chunk():
    hits = [_hit("chk_a", page_start=3, page_end=4)]
    citations = bind_citations("Claim [1].", hits)
    assert citations[0].document_id == "doc_chk_a"
    assert citations[0].label == hits[0].chunk.citation_label()
    assert citations[0].page_start == 3
    assert citations[0].page_end == 4


# -- ask(): abstention paths, in order ---------------------------------------

def test_ask_abstains_no_results_without_calling_the_model():
    chat = FakeChat(text="should never be reached")
    orch = _orchestrator(chat, hits=[])

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.NO_RESULTS
    assert chat.calls == [], "no hits -- generation must not even be attempted"


def test_ask_abstains_low_support_without_calling_the_model():
    """The gate itself still works when a threshold is configured -- the
    live default is 0.0 (see docs/adr/0008: retrieve()'s score cannot
    support a threshold on this corpus), which is a statement about the
    deployed value, not about whether the mechanism functions. Configured
    explicitly here so this test does not silently start asserting nothing
    if that default ever moves again."""
    weak_hit = _hit("chk_a", score=0.01)
    chat = FakeChat(text="should never be reached")
    settings = Settings()
    settings.retrieval.min_support_score = 0.5
    orch = _orchestrator(chat, hits=[weak_hit], settings=settings)

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.LOW_SUPPORT
    assert chat.calls == [], "weak top score -- generation must not even be attempted"


def test_ask_abstains_insufficient_context_when_model_says_so_explicitly():
    """Distinct from UNGROUNDED_DRAFT: this is the model reading the
    context and correctly declining, not producing an uncited draft. P1-09
    scores the two oppositely -- this one is a PASS on the unanswerable
    slice, UNGROUNDED_DRAFT is a suspected-fabrication FAILURE -- so
    collapsing them would make abstention precision unmeasurable."""
    hits = [_hit("chk_a", score=0.9)]
    chat = FakeChat(text="INSUFFICIENT_CONTEXT")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.INSUFFICIENT_CONTEXT
    assert answer.citations == []
    assert "INSUFFICIENT_CONTEXT" not in answer.text, (
        "the literal marker must never be passed through as answer text"
    )


def test_ask_abstains_insufficient_context_with_surrounding_whitespace():
    hits = [_hit("chk_a", score=0.9)]
    chat = FakeChat(text="  INSUFFICIENT_CONTEXT\n")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.INSUFFICIENT_CONTEXT


def test_insufficient_context_and_ungrounded_draft_have_identical_user_facing_text():
    """Same pattern as test_permission_boundaries.py's
    test_no_results_and_no_permitted_results_are_indistinguishable: the
    recorded reason must differ (that's the whole point, for P1-09), but
    what the caller actually sees must not -- an abstention is an
    abstention from the user's side regardless of which internal path
    produced it."""
    from isc.models.answer import Answer

    a = Answer.abstain("q", AbstentionReason.INSUFFICIENT_CONTEXT)
    b = Answer.abstain("q", AbstentionReason.UNGROUNDED_DRAFT)
    assert a.abstention_reason != b.abstention_reason
    assert a.text == b.text


def test_ask_abstains_ungrounded_when_zero_citations_survive_binding():
    hits = [_hit("chk_a", score=0.9)]
    chat = FakeChat(text="An answer with no citation markers at all.")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.UNGROUNDED_DRAFT


def test_ask_abstains_ungrounded_when_the_only_marker_is_out_of_range():
    hits = [_hit("chk_a", score=0.9)]
    chat = FakeChat(text="An answer citing a block that was never sent [9].")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.UNGROUNDED_DRAFT


def test_ask_returns_grounded_answer_with_citations_on_success():
    hits = [_hit("chk_a", score=0.9), _hit("chk_b", score=0.8)]
    chat = FakeChat(text="The answer is X [1].")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert not answer.abstained
    assert answer.text == "The answer is X [1]."
    assert [c.chunk_id for c in answer.citations] == ["chk_a"]
    assert answer.supporting == hits


def test_ask_never_surfaces_no_permitted_results():
    """The orchestrator has no way to know a retrieval came back empty
    because of ACL filtering rather than a genuine lack of matches --
    retrieve() has already thrown that information away by the time it
    returns []. NO_PERMITTED_RESULTS exists for eval/audit to assign with
    gold knowledge the orchestrator does not have; ask() itself must only
    ever emit NO_RESULTS, so the caller-facing behaviour is identical by
    construction, not by a separate check somewhere."""
    chat = FakeChat(text="should never be reached")
    orch = _orchestrator(chat, hits=[])

    answer = orch.ask("q", _principal())

    assert answer.abstention_reason == AbstentionReason.NO_RESULTS
    assert answer.abstention_reason != AbstentionReason.NO_PERMITTED_RESULTS


def test_default_min_support_score_disables_the_gate():
    """Pins docs/adr/0008's decision: 0.0 is a deliberate disable, not an
    unset placeholder -- a near-zero real score must still pass through to
    generation rather than abstaining LOW_SUPPORT."""
    settings = Settings()
    assert settings.retrieval.min_support_score == 0.0
    near_zero_hit = _hit("chk_a", score=1e-6)
    chat = FakeChat(text="An answer [1].")
    orch = _orchestrator(chat, hits=[near_zero_hit], settings=settings)

    answer = orch.ask("q", _principal())

    assert not answer.abstained


# -- regression guard: _generate() and _bind_citations() must share hits ----

def test_ask_binds_citations_against_the_same_hits_list_used_for_generation():
    """FakeChat always cites [2] regardless of what the context blocks
    actually contain -- this only passes if ask() threads the exact same
    hits object into both _generate() and _bind_citations(). If either end
    independently re-fetched or re-sorted hits, [2] would silently resolve
    to the wrong chunk while still looking like a valid citation."""
    hits = [_hit("chk_a", score=0.9), _hit("chk_b", score=0.8), _hit("chk_c", score=0.7)]
    chat = FakeChat(text="Some grounded claim [2].")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chk_b"
    assert answer.citations[0].chunk_id == hits[1].chunk.id


# -- verify_attribution(): binding checks [n] exists, this checks the prose -
#
# Checked against Chunk.filters (supplier_id / po_number), not raw
# chunk.text -- see docs/adr/0009. A table or footer chunk's TEXT never
# repeats a document's header fields, but its FILTERS carry the same
# po_number/supplier_id as every other chunk of that document.

_SUPPLIER_IDS = {
    "Omron Electronics Asia": "V102337",
    "Keyence Singapore Pte Ltd": "V103014",
    "Kestrel Industrial AG": "V100781",
}


def test_verify_attribution_passes_a_sentence_naming_no_entity():
    """Most sentences name no supplier or PO number at all -- those must
    never fail, or this check would abstain on nearly every real answer."""
    hits = [_hit("chk_a", text="Buyer A. Tan Delivery Address 12 Tuas Avenue 8")]
    draft = "The buyer contact is A. Tan [1]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []
    assert result.unverifiable == []


def test_verify_attribution_passes_when_the_named_supplier_id_matches_its_cited_chunk():
    hits = [_hit("chk_a", text="| Unit Price |\n| 1,536.37 |", filters={"supplier_id": "V102337"})]
    draft = "We paid 1,536.37 SGD per unit from Omron Electronics Asia [1]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []


def test_verify_attribution_passes_a_table_chunk_whose_text_never_mentions_the_supplier():
    """The exact shape found live in the P1-07 sample (q_li_02, q_re_01,
    q_re_10 as alice): a correct answer, correctly cited to a table row
    chunk whose raw text is just a markdown table and never repeats the
    document's supplier name or PO number -- text matching alone rejected
    all three as false positives. filters carries the identity regardless
    of chunk shape, so this must pass."""
    hits = [_hit(
        "chk_a", filters={"po_number": "4522345741", "supplier_id": "V102337"},
        text="| Item | Part Number | Unit Price |\n|---|---|---|\n| 10 | PLC-1756-L83 | 1,536.37 |",
    )]
    draft = ("We paid 1,536.37 SGD per unit for the 250-unit order of ControlLogix "
             "processor modules from Omron Electronics Asia [1].")
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []
    assert result.unverifiable == []


def test_verify_attribution_flags_a_supplier_whose_cited_chunk_disagrees():
    """A real mismatch: the cited chunk DOES carry a supplier_id filter,
    and it names a different supplier than the sentence does."""
    hits = [_hit("chk_a", text="| Unit Price |\n| 609.16 |", filters={"supplier_id": "V103014"})]
    draft = "We paid 609.16 SGD per unit from Omron Electronics Asia [1]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches
    assert "Omron Electronics Asia" in result.mismatches[0]
    assert result.unverifiable == []


def test_verify_attribution_flags_a_po_number_whose_cited_chunk_disagrees():
    hits = [_hit("chk_a", text="Total 392,589.57 SGD", filters={"po_number": "4500000000"})]
    draft = "The total on PO 4522345741 is 392,589.57 SGD [1]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches
    assert result.unverifiable == []


def test_verify_attribution_treats_a_missing_supplier_id_filter_as_unverifiable_not_a_failure():
    """2 of 20 documents in this corpus have no supplier_id at all --
    genuinely absent from the source, not a model or chunking defect. A
    cited chunk with no supplier_id filter key cannot be checked either
    way and must not abstain a correct answer over a metadata gap."""
    hits = [_hit("chk_a", text="| Unit Price |\n| 1,536.37 |", filters={"po_number": "4522345741"})]
    draft = "We paid 1,536.37 SGD per unit from Omron Electronics Asia [1]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []
    assert result.unverifiable
    assert "Omron Electronics Asia" in result.unverifiable[0]


def test_verify_attribution_checks_each_sentence_against_only_its_own_markers():
    """Two sentences, two different citations -- a supplier correctly
    attributed to [1] must not be validated against [2]'s unrelated filters."""
    hits = [
        _hit("chk_a", text="...", filters={"supplier_id": "V102337"}),
        _hit("chk_b", text="...", filters={"supplier_id": "V103014"}),
    ]
    draft = "The Omron order total is 392,589.57 SGD [1]. The Keyence order total is 138,870.01 SGD [2]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []


def test_verify_attribution_does_not_split_on_a_buyer_initial():
    """Found live running the P1-07 sample: 'The buyer contact on PO
    4513180299 is A. Tan [1].' split into '...is A.' and 'Tan [1].' without
    this guard -- the PO number landed in the first fragment, which cites
    nothing, and every single_hop question whose gold answer is a two-part
    name false-positived on it. This corpus's buyer_contact values (A. Tan,
    J. Ruiz, M. Weber, ...) are exactly this shape."""
    hits = [_hit("chk_a", text="Buyer A. Tan", filters={"po_number": "4513180299"})]
    draft = "The buyer contact on PO 4513180299 is A. Tan [1]."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []


def test_verify_attribution_does_not_split_on_a_numbered_list_marker():
    """Found live running all 56 gold questions: 'the following orders:

    1. Order Total: USD 63,926.90 [4]\\n2. ...' split before '1.', stranding
    the supplier name named in the intro clause with no citation at all
    while every per-order total landed, correctly cited, in later
    fragments -- the model-generated cross_document breakdowns in this
    corpus are exactly this shape (numbered list, one citation per line)."""
    hits = [_hit("chk_a", text="Order Total 63,926.90", filters={"supplier_id": "V100781"})]
    draft = ("The total spent with Kestrel Industrial AG is calculated from the following orders:\n\n"
             "1. Order Total: USD 63,926.90 [1]\n\nTotal = USD 63,926.90.")
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches == []


def test_verify_attribution_known_false_positive_on_an_uncited_later_reference():
    """Documented limitation, not desired behaviour -- this is exactly the
    shape flagged before running the sample: a sentence that re-names an
    entity already established by an earlier citation, without re-citing
    it itself, cites no chunk at all -- there is nothing to check filters
    against. The fact is true; this check cannot see that at per-sentence
    granularity, and a genuinely uncited claim still has to fail (this is
    NOT the missing-metadata case, which is unverifiable instead)."""
    hits = [_hit("chk_a", text="...", filters={"supplier_id": "V100781"})]
    draft = "The supplier is Kestrel Industrial AG [1]. Kestrel Industrial AG's payment terms are Net 30."
    result = verify_attribution(draft, hits, _SUPPLIER_IDS)
    assert result.mismatches, "second sentence names the supplier again without citing anything"
    assert result.unverifiable == []


def test_ask_abstains_attribution_mismatch_on_the_real_ben_omron_keyence_regression():
    """Regression fixture for the exact failure found live in the P1-07
    sample: q_re_10 as u_ben, unfiltered corpus-wide retrieval, cited a
    real, permitted chunk from po_001.pdf (Keyence Singapore Pte Ltd,
    supplier_id V103014, line 120: qty 250, unit_price 609.16) while the
    generated sentence named it as "the Omron Electronics Asia order"
    (V102337). Binding alone passed this -- [1] resolved to a real chunk --
    which is exactly the gap verify_attribution() exists to close. Uses the
    real supplier list (default Settings() -> data/masters/suppliers.json),
    not a stub, since the point is this exact pair of real names/ids."""
    hits = [_hit(
        "chk_po001_line120", score=0.9, filters={"po_number": "4513180299", "supplier_id": "V103014"},
        text="| Item | Part Number | Description | Qty | Unit Price |\n|---|---|---|---|---|\n"
             "| 120 | PLC-1756-L83 | ControlLogix processor module | 250 | 609.16 |",
    )]
    chat = FakeChat(
        text="We paid 609.16 SGD per unit for the 250-unit order of ControlLogix "
             "processor modules from Omron Electronics Asia [1]."
    )
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert answer.abstained
    assert answer.abstention_reason == AbstentionReason.ATTRIBUTION_MISMATCH
    assert answer.citations == []


def test_ask_returns_grounded_answer_when_attribution_is_correct():
    hits = [_hit(
        "chk_a", score=0.9, filters={"po_number": "4522345741", "supplier_id": "V102337"},
        text="| Unit Price |\n| 1,536.37 |",
    )]
    chat = FakeChat(text="We paid 1,536.37 SGD per unit from Omron Electronics Asia [1].")
    orch = _orchestrator(chat, hits)

    answer = orch.ask("q", _principal())

    assert not answer.abstained
    assert answer.citations[0].chunk_id == "chk_a"

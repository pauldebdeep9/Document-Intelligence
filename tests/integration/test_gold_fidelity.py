"""Fidelity tests for the retrieval gold set (P1-08).

If these fail, P1-09's recall numbers would be measuring against chunk ids
that no longer exist, or a restriction that does not actually fire -- a
result that looks like a real number but measures nothing. Skipped when the
retrieval gold has not been generated, so a fresh clone still runs green on
`make test` before `python scripts/gen_gold.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isc.common.config import get_settings
from isc.common.ids import corpus_fingerprint
from isc.index.chunker import settings_fingerprint
from isc.models.acl import AclSet, Principal, Sensitivity
from isc.models.records.purchase_order import PurchaseOrder
from isc.retrieve.retriever import Retriever
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "retrieval" / "questions.json"
EXTRACTION_GOLD = ROOT / "data" / "gold" / "extraction"
ACL_DIR = ROOT / "data" / "acl"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"

pytestmark = pytest.mark.skipif(
    not GOLD_PATH.exists(),
    reason="retrieval gold not generated; run `python scripts/gen_gold.py`",
)


def _load() -> dict:
    return json.loads(GOLD_PATH.read_text())


def _users() -> dict[str, Principal]:
    raw = json.loads((ACL_DIR / "users.json").read_text())
    return {
        uid: Principal(
            id=uid, group_ids=frozenset(u["groups"]), site_ids=frozenset(u["sites"]),
            clearance=Sensitivity(u["clearance"]), jurisdictions=frozenset(u["jurisdictions"]),
        )
        for uid, u in raw.items()
    }


def _doc_acl(name: str) -> AclSet:
    raw = json.loads((SYNTHETIC_DIR / f"{name}.acl.json").read_text())
    return AclSet(
        allow_terms=frozenset(raw["allow_terms"]), deny_terms=frozenset(raw["deny_terms"]),
        sensitivity=Sensitivity(raw["sensitivity"]), jurisdictions=frozenset(raw["jurisdictions"]),
    )


def _live_store() -> LocalVectorStore:
    settings = get_settings()
    return LocalVectorStore(settings.paths.data / "vector_store.pkl")


def _live_docstore() -> SqliteDocStore:
    return SqliteDocStore(get_settings().paths.data / "docstore.sqlite")


# -- the check this whole gold set exists to satisfy --------------------------

def test_every_gold_chunk_id_resolves_against_the_index():
    d = _load()
    live_ids = {c.id for c in _live_store()._chunks}
    missing = [
        (q["id"], cid) for q in d["questions"] for cid in q["gold_chunk_ids"] if cid not in live_ids
    ]
    assert not missing, f"gold_chunk_ids missing from the live index: {missing}"


def test_fingerprint_matches_the_current_index():
    d = _load()
    settings = get_settings()
    recorded = d["provenance"]["settings_fingerprint"]
    assert recorded == settings_fingerprint(settings.chunk)
    store_fp = _live_store().settings_fingerprint()
    assert store_fp is not None, "live index has no fingerprint -- was it built by isc index?"
    assert recorded == store_fp


def test_corpus_fingerprint_matches_the_current_corpus():
    """settings_fingerprint (above) is blind to the corpus's own content --
    it is a hash of ChunkSettings alone. corpus_fingerprint covers the gap:
    a hash over every ingested document's content_sha256, independent of
    which documents the gold's questions happen to reference."""
    d = _load()
    recorded = d["provenance"]["corpus_fingerprint"]
    assert recorded == corpus_fingerprint(_live_docstore().content_hashes())


def test_corpus_fingerprint_changes_if_po_019_content_changes():
    """po_019.pdf is the one document no gold question's gold_chunk_ids ever
    names (see docs/adr/0007's "A provenance guard only protects against the
    inputs it actually covers") -- test_every_gold_chunk_id_resolves_against_
    the_index has zero power to catch a content change confined to it.
    corpus_fingerprint has to catch it anyway, regardless of corpus order."""
    docs = _live_docstore()
    ids = docs.list_documents()
    po_019_id = next(
        (doc_id for doc_id in ids if docs.get_document(doc_id).source_uri.endswith("po_019.pdf")),
        None,
    )
    assert po_019_id is not None, "po_019.pdf not found in the live docstore"
    base_hashes = [docs.get_document(doc_id).content_sha256 for doc_id in ids]
    mutated_hashes = [
        "0" * 64 if doc_id == po_019_id else h for doc_id, h in zip(ids, base_hashes)
    ]
    assert corpus_fingerprint(mutated_hashes) != corpus_fingerprint(base_hashes)


# -- structure: counts, uniqueness, principal validity -----------------------

def test_subtype_counts_match_targets():
    d = _load()
    by_subtype = d["counts"]["by_subtype"]
    by_class = d["counts"]["by_class"]
    assert by_subtype["single_hop"] == 10
    assert by_subtype["header_lookup"] == 5
    assert by_subtype["line_item"] == 8
    assert by_subtype["cross_document"] == 8
    assert by_subtype["ambiguous"] == 4
    assert by_subtype["absent"] == 3
    assert by_subtype["out_of_scope"] == 3
    assert by_subtype["underspecified"] == 2
    assert by_subtype["restricted_filtered"] == 8  # alice/ben pairs, PO number in the text
    assert by_subtype["restricted_unfiltered"] == 4  # same pairs, no PO number in the text
    assert by_subtype["no_reader"] == 1  # po_002 -- zero readers in the graph
    assert by_class["answerable"] == 35
    assert by_class["unanswerable"] == 8
    assert by_class["restricted"] == 13  # 8 filtered + 4 unfiltered + po_002
    assert d["counts"]["total"] == 56


def test_every_question_id_is_unique():
    d = _load()
    ids = [q["id"] for q in d["questions"]]
    assert len(ids) == len(set(ids))


def test_principal_can_read_every_source_document():
    """A question assigned to a principal who cannot see its own source
    document is not testing retrieval, it is testing a bug in the gold set.
    Restricted pairs and the no_reader question are checked separately
    below, since their whole point is that some/all principals cannot read."""
    d = _load()
    users = _users()
    for q in d["questions"]:
        if q["question_class"] == "restricted":
            continue
        principal = users[q["principal"]]
        for name in set(q["source_documents"]):
            acl = _doc_acl(name)
            assert principal.may_read(acl), (
                f"{q['id']}: principal {q['principal']} cannot read {name}, its own source document"
            )


# -- unanswerable: no gold chunks, and genuinely not retrievable -------------

def test_unanswerable_questions_have_no_gold_chunks():
    d = _load()
    for q in d["questions"]:
        if q["question_class"] == "unanswerable":
            assert q["gold_chunk_ids"] == []
            assert q["gold_answer"] is None


def test_absent_questions_reference_a_real_absent_field():
    """Cross-checked against the extraction gold's own meta.absent_fields,
    not just trusted from the question's own note."""
    d = _load()
    for q in d["questions"]:
        if q["subtype"] != "absent":
            continue
        name = q["source_documents"][0]
        g = json.loads((EXTRACTION_GOLD / name.replace(".pdf", ".json")).read_text())
        absent_fields = g["meta"]["absent_fields"]
        assert absent_fields, f"{name} has no absent fields at all"
        assert any(f in q["note"] for f in absent_fields), (
            f"{q['id']}: note does not name one of {name}'s real absent_fields {absent_fields}"
        )


def test_out_of_scope_questions_ask_about_fields_the_schema_does_not_have():
    """Structural check that these are genuinely out of scope, not just
    absent from one document: the PurchaseOrder schema itself has no
    delivery/quality/contract-terms field for any document to populate."""
    d = _load()
    schema_fields = set(PurchaseOrder.model_fields)
    out_of_scope = [q for q in d["questions"] if q["subtype"] == "out_of_scope"]
    assert len(out_of_scope) == 3
    banned = {"delivery_confirmed", "quality_inspection", "warranty_terms", "contract_terms"}
    assert banned.isdisjoint(schema_fields)


def test_underspecified_questions_reference_no_document():
    """Genuinely unanswerable by construction: no PO or line is named, so
    there is nothing for retrieval to resolve to."""
    d = _load()
    underspecified = [q for q in d["questions"] if q["subtype"] == "underspecified"]
    assert len(underspecified) == 2
    for q in underspecified:
        assert q["source_documents"] == []


def test_unanswerable_expected_behavior_is_explicit_and_consistent():
    d = _load()
    for q in d["questions"]:
        if q["question_class"] != "unanswerable":
            continue
        if q["subtype"] == "underspecified":
            assert q["expected_behavior"] == "abstain_with_clarification"
        else:
            assert q["expected_behavior"] == "abstain"


# -- restricted: verified both directions, and alice's answer is specific ----

_RESTRICTED_SUBTYPES = {"restricted_filtered", "restricted_unfiltered"}


def test_restricted_pairs_verified_both_directions():
    d = _load()
    users = _users()
    alice, ben = users["u_alice"], users["u_ben"]
    pairs = [q for q in d["questions"] if q["subtype"] in _RESTRICTED_SUBTYPES]
    assert len(pairs) == 12
    for q in pairs:
        name = q["source_documents"][0]
        acl = _doc_acl(name)
        assert alice.may_read(acl), f"{q['id']}: u_alice cannot read {name} -- not answerable"
        assert not ben.may_read(acl), f"{q['id']}: u_ben CAN read {name} -- restriction never fires"
        assert q["principal"] == "u_alice"
        assert q["principal_b"] == "u_ben"
        assert q["expected_b"] == "empty"


def test_restricted_questions_have_a_specific_gold_answer():
    """Abstention is trivially achievable by a system that just retrieves
    badly. Each pair must ask for a concrete value -- alice getting the
    right value and ben getting nothing are then two separately checkable
    assertions, not one vague one."""
    d = _load()
    for q in d["questions"]:
        if q["subtype"] not in _RESTRICTED_SUBTYPES:
            continue
        answer = q["gold_answer"]
        assert answer not in (None, "", [], {}), f"{q['id']}: gold_answer is not a specific value"
        assert len(q["gold_chunk_ids"]) == 1


# -- restricted: filtered vs. unfiltered actually differ at the retriever ----

def test_restricted_filtered_questions_produce_a_po_number_filter():
    """The whole point of the filtered/unfiltered split: without this, all 8
    filtered questions are decided against a single document's ~3-chunk
    pool, and a permission bug that only shows up under corpus-wide
    retrieval (a ranking path that bypasses the filter) would never be
    exercised. Pinned here so a later rephrasing that quietly drops the PO
    number from the text can't silently reclassify a question without the
    counts test catching a mismatch."""
    d = _load()
    retriever = Retriever(store=None, embedder=None, chat=None, settings=None)  # type: ignore[arg-type]
    filtered = [q for q in d["questions"] if q["subtype"] == "restricted_filtered"]
    assert len(filtered) == 8
    for q in filtered:
        assert retriever.infer_filters(q["text"]) != {}, (
            f"{q['id']}: expected a po_number filter, got none -- {q['text']!r}"
        )


def test_restricted_unfiltered_questions_produce_no_filter():
    """The other half of the split: these 4 must resolve to {} from
    infer_filters(), so retrieval genuinely runs against the whole corpus,
    not just a smaller version of the same single-document pool the filtered
    questions already exercise."""
    d = _load()
    retriever = Retriever(store=None, embedder=None, chat=None, settings=None)  # type: ignore[arg-type]
    unfiltered = [q for q in d["questions"] if q["subtype"] == "restricted_unfiltered"]
    assert len(unfiltered) == 4
    for q in unfiltered:
        assert retriever.infer_filters(q["text"]) == {}, (
            f"{q['id']}: expected no filters (corpus-wide retrieval), "
            f"got {retriever.infer_filters(q['text'])} -- {q['text']!r}"
        )


def test_no_reader_question_has_zero_readers_across_the_whole_identity_graph():
    """po_002: export-controlled + EMEA, and no named principal holds both
    the group/site grant and the required clearance+jurisdiction. Every
    principal in the graph must be checked, not just alice/ben."""
    d = _load()
    users = _users()
    no_reader = [q for q in d["questions"] if q["subtype"] == "no_reader"]
    assert len(no_reader) == 1
    q = no_reader[0]
    assert set(q["principals_checked"]) == set(users.keys()), "must check every named principal"
    acl = _doc_acl(q["source_documents"][0])
    readers = [uid for uid in q["principals_checked"] if users[uid].may_read(acl)]
    assert readers == [], f"{q['id']}: unexpectedly has readers: {readers}"
    assert q["gold_chunk_ids"] == []
    assert q["expected"] == "empty"

"""Build the retrieval gold set from the generated corpus.

THE INVARIANT, same as gen_corpus.py's: questions are derived from the gold
EXTRACTION RECORDS (data/gold/extraction/*.json), never from a model. A
model-written question set encodes that model's own reading of the corpus,
which is circular the moment it is used to evaluate that same model's
retrieval.

Question classes and their proportions:
  answerable    ~70%  single_hop, header_lookup, line_item, cross_document,
                      ambiguous -- each tagged with its subtype
  unanswerable  ~15%  plausible questions the corpus cannot answer; abstention
                      is the only correct behaviour. Subtyped absent /
                      out_of_scope / underspecified
  restricted    ~15%  answerable for one principal, invisible to another --
                      the pair is u_alice/u_ben (data/acl/adversarial_pairs.json:
                      "confidential clearance vs internal clearance"), verified
                      both directions

Every question is asked as a specific principal, checked against the real ACL
graph (data/acl/users.json + the corpus's per-document allow_terms) so a
question is never assigned to someone who cannot see its own source document
-- see _readable_by() and the planning check this script's tests re-run.

Chunk ids are resolved by re-chunking each document with the SAME pure
function index/pipeline.py uses (chunk_document), not by hand-transcribing
ids: chunk_id() is a deterministic function of (document_id, ordinal, text),
so re-deriving it here and cross-checking against the live LocalVectorStore
(see _verify_against_index) is both easier and more honest than guessing.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from isc.common.config import Settings, get_settings
from isc.common.errors import ChunkSettingsMismatch
from isc.common.ids import corpus_fingerprint
from isc.index.chunker import chunk_document, filters_from_record, settings_fingerprint
from isc.models.acl import AclSet, Principal, Sensitivity
from isc.models.chunk import Chunk
from isc.models.records import invoice, purchase_order  # noqa: F401 -- registry import side effect
from isc.models.records.base import registry
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore

# 2000-01-01T00:00:00Z -- the same SOURCE_DATE_EPOCH base gen_corpus.py pins
# PDF timestamps to (see _pin_pdf_timestamps there). Duplicated rather than
# imported: the two scripts are siblings, not a package, and the convention
# to match is the formula (base + seed), not a shared runtime dependency.
_PDF_EPOCH_BASE = 946684800


# ---------------------------------------------------------------------------
# Loading: gold records, the ACL graph, and a fresh re-chunk of every document
# ---------------------------------------------------------------------------

def _load_extraction_gold(gold_dir: Path) -> dict[str, dict[str, Any]]:
    """PO filename ("po_000.pdf") -> its full extraction gold record."""
    by_name: dict[str, dict[str, Any]] = {}
    for p in sorted(gold_dir.glob("*.json")):
        g = json.loads(p.read_text())
        by_name[g["document"]] = g
    return by_name


def _load_users(acl_dir: Path) -> dict[str, Principal]:
    raw = json.loads((acl_dir / "users.json").read_text())
    return {
        uid: Principal(
            id=uid, group_ids=frozenset(u["groups"]), site_ids=frozenset(u["sites"]),
            clearance=Sensitivity(u["clearance"]), jurisdictions=frozenset(u["jurisdictions"]),
        )
        for uid, u in raw.items()
    }


def _build_chunk_index(
    docs: SqliteDocStore, settings: Settings, gold_by_name: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[Chunk]], dict[str, str]]:
    """PO filename -> its ordered chunks, and PO filename -> real document_id.

    Mirrors index/pipeline.py's own per-document logic exactly (same registry
    lookup, same filters_from_record call, same chunk_document call) so the
    chunk ids produced here are the ones actually in the built index, not an
    approximation of them.
    """
    ponum_to_name = {g["raw"]["po_number"]: name for name, g in gold_by_name.items()}
    chunks_by_name: dict[str, list[Chunk]] = {}
    doc_id_by_name: dict[str, str] = {}

    for doc_id in docs.list_documents():
        doc = docs.get_document(doc_id)
        if doc is None:
            continue
        record_payload = docs.get_record(doc_id)
        filters: dict[str, str] = {}
        if record_payload is not None:
            record_cls = registry.get(doc.doc_type)
            record = record_cls.model_validate(record_payload)
            filters = filters_from_record(record)
        po_number = filters.get("po_number")
        name = ponum_to_name.get(po_number) if po_number else None
        if name is None:
            continue  # not a PO this gold set indexes (e.g. a non-PO doc_type)
        chunks_by_name[name] = chunk_document(doc, settings.chunk, filters=filters)
        doc_id_by_name[name] = doc_id

    missing = set(gold_by_name) - set(chunks_by_name)
    if missing:
        raise RuntimeError(
            f"{len(missing)} PO(s) in extraction gold have no matching indexed "
            f"document -- run `isc ingest && isc parse && isc extract` first: {sorted(missing)}"
        )
    return chunks_by_name, doc_id_by_name


def _readable_by(acl: AclSet, users: dict[str, Principal]) -> list[str]:
    return [uid for uid, p in users.items() if p.may_read(acl)]


def _doc_acl(synthetic_dir: Path, name: str) -> AclSet:
    """name is the PO filename including its .pdf suffix, e.g. "po_000.pdf" --
    the sidecar is "po_000.pdf.acl.json", not "po_000.acl.json"."""
    raw = json.loads((synthetic_dir / f"{name}.acl.json").read_text())
    return AclSet(
        allow_terms=frozenset(raw["allow_terms"]), deny_terms=frozenset(raw["deny_terms"]),
        sensitivity=Sensitivity(raw["sensitivity"]), jurisdictions=frozenset(raw["jurisdictions"]),
    )


# ---------------------------------------------------------------------------
# Chunk lookup helpers -- every gold_chunk_id below is resolved through these,
# never typed in by hand.
# ---------------------------------------------------------------------------

def _header_chunk(chunks: dict[str, list[Chunk]], name: str) -> str:
    """The document's first chunk -- title + key/value header block."""
    return chunks[name][0].id


def _line_chunk(chunks: dict[str, list[Chunk]], name: str, line_number: int) -> str:
    for c in chunks[name]:
        if c.line_range is not None and c.line_range[0] <= line_number <= c.line_range[1]:
            return c.id
    raise ValueError(f"no chunk in {name} covers line {line_number}")


def _line(gold: dict[str, Any], name: str, line_number: int) -> dict[str, Any]:
    for ln in gold[name]["raw"]["lines"]:
        if ln["line_number"] == line_number:
            return ln
    raise ValueError(f"{name} has no line {line_number}")


def _money(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def _fmt_money(v: Decimal) -> str:
    return f"{v:,.2f}"


# ---------------------------------------------------------------------------
# Question builders -- one per subtype. Values are always read out of the
# loaded gold record, never re-typed, so a question and its gold_answer
# cannot silently drift apart.
# ---------------------------------------------------------------------------

def _single_hop(qid: str, name: str, field: str, text: str, principal: str,
                 chunks: dict[str, list[Chunk]], gold: dict[str, Any]) -> dict[str, Any]:
    g = gold[name]
    return {
        "id": qid, "text": text.format(po=g["raw"]["po_number"]),
        "question_class": "answerable", "subtype": "single_hop",
        "gold_chunk_ids": [_header_chunk(chunks, name)],
        "gold_answer": g["raw"][field],
        "source_documents": [name], "principal": principal,
    }


def _header_lookup(qid: str, name: str, field: str, text: str, principal: str,
                    chunks: dict[str, list[Chunk]], gold: dict[str, Any]) -> dict[str, Any]:
    d = _single_hop(qid, name, field, text, principal, chunks, gold)
    d["subtype"] = "header_lookup"
    return d


def _line_item(qid: str, name: str, line_number: int, field: str, text: str, principal: str,
               chunks: dict[str, list[Chunk]], gold: dict[str, Any]) -> dict[str, Any]:
    g = gold[name]
    ln = _line(gold, name, line_number)
    return {
        "id": qid, "text": text.format(po=g["raw"]["po_number"], line=line_number),
        "question_class": "answerable", "subtype": "line_item",
        "gold_chunk_ids": [_line_chunk(chunks, name, line_number)],
        "gold_answer": ln[field],
        "source_documents": [name], "principal": principal,
    }


def _cross_doc_total(qid: str, text: str, names: list[str], currency: str, principal: str,
                      chunks: dict[str, list[Chunk]], gold: dict[str, Any]) -> dict[str, Any]:
    breakdown = []
    total = Decimal("0")
    for name in names:
        g = gold[name]
        assert g["raw"]["currency"] == currency, f"{name} is not {currency}"
        amt = _money(g["raw"]["total_amount"])
        total += amt
        breakdown.append({"document": name, "total_amount": g["raw"]["total_amount"]})
    return {
        "id": qid, "text": text,
        "question_class": "answerable", "subtype": "cross_document",
        "gold_chunk_ids": [_header_chunk(chunks, name) for name in names],
        "gold_answer": {"total": _fmt_money(total), "currency": currency, "breakdown": breakdown},
        "source_documents": names, "principal": principal,
    }


def _cross_doc_part(qid: str, text: str, part_number: str, refs: list[tuple[str, int]],
                     principal: str, chunks: dict[str, list[Chunk]],
                     gold: dict[str, Any]) -> dict[str, Any]:
    answer = []
    gold_ids = []
    names = []
    for name, line_number in refs:
        ln = _line(gold, name, line_number)
        assert ln["part_number"] == part_number, f"{name} line {line_number} is not {part_number}"
        answer.append({
            "document": name, "line_number": line_number,
            "unit_price": ln["unit_price"], "currency": gold[name]["raw"]["currency"],
        })
        gold_ids.append(_line_chunk(chunks, name, line_number))
        names.append(name)
    return {
        "id": qid, "text": text,
        "question_class": "answerable", "subtype": "cross_document",
        "gold_chunk_ids": gold_ids, "gold_answer": answer,
        "source_documents": names, "principal": principal,
    }


def _ambiguous(qid: str, text: str, gold_answer: Any, refs: list[tuple[str, str]],
               principal: str, chunks: dict[str, list[Chunk]]) -> dict[str, Any]:
    """refs: list of (po_name, "header" | line_number-as-str)."""
    gold_ids = []
    names = []
    for name, where in refs:
        gold_ids.append(_header_chunk(chunks, name) if where == "header"
                         else _line_chunk(chunks, name, int(where)))
        names.append(name)
    return {
        "id": qid, "text": text,
        "question_class": "answerable", "subtype": "ambiguous",
        "gold_chunk_ids": gold_ids, "gold_answer": gold_answer,
        "source_documents": names, "principal": principal,
        "note": "spans two distinct, confusable suppliers -- correct behaviour "
                "is to surface both, not silently pick one",
    }


_EXPECTED_BEHAVIOR = {
    # absent/out_of_scope: the question is well-specified, the corpus just
    # does not contain the answer -- nothing to clarify, plain abstention is
    # the whole of the correct behaviour.
    "absent": "abstain",
    "out_of_scope": "abstain",
    # underspecified: abstention alone is indistinguishable from a system
    # that just failed to retrieve. Asking which PO/part/line is meant is
    # also a defensible response, but "either is fine" is not scoreable --
    # this gold set picks abstain-with-clarification as THE target
    # behaviour, stated explicitly rather than left implied.
    "underspecified": "abstain_with_clarification",
}


def _unanswerable(qid: str, text: str, subtype: str, note: str, principal: str,
                   source_documents: list[str]) -> dict[str, Any]:
    assert subtype in _EXPECTED_BEHAVIOR
    return {
        "id": qid, "text": text,
        "question_class": "unanswerable", "subtype": subtype,
        "gold_chunk_ids": [], "gold_answer": None,
        "source_documents": source_documents, "principal": principal, "note": note,
        "expected_behavior": _EXPECTED_BEHAVIOR[subtype],
    }


def _restricted(qid: str, name: str, field: str | None, line_number: int | None, text: str,
                 subtype: str, chunks: dict[str, list[Chunk]], gold: dict[str, Any]) -> dict[str, Any]:
    """subtype is "restricted_filtered" (text carries the PO number, so
    infer_filters() narrows retrieval to that one document before ACL is
    even applied -- ben's empty result is decided against a pool the size of
    one document's own chunks) or "restricted_unfiltered" (text deliberately
    carries no PO number, so infer_filters() returns {} and retrieval runs
    against the WHOLE corpus -- ben's empty result has to hold up against
    every chunk he is entitled to read, not just the ones from this one
    document). Split into two subtypes, not left as one, so P1-09 can report
    permission enforcement separately for each path: a leak that only shows
    up when a ranking path bypasses the filter (corpus-wide retrieval) would
    pass every filtered question and look clean. See
    test_restricted_filtered_questions_produce_a_po_number_filter /
    test_restricted_unfiltered_questions_produce_no_filter in
    test_gold_fidelity.py, which pin this by construction rather than by
    convention."""
    g = gold[name]
    if line_number is not None:
        ln = _line(gold, name, line_number)
        gold_chunk_ids = [_line_chunk(chunks, name, line_number)]
        answer = ln[field]  # type: ignore[index]
    else:
        gold_chunk_ids = [_header_chunk(chunks, name)]
        answer = g["raw"][field]  # type: ignore[index]
    return {
        "id": qid, "text": text.format(po=g["raw"]["po_number"]),
        "question_class": "restricted", "subtype": subtype,
        "gold_chunk_ids": gold_chunk_ids, "gold_answer": answer,
        "source_documents": [name], "principal": "u_alice",
        "principal_b": "u_ben",
        "expected_b": "empty",  # ben's search must come back with zero chunks --
        # indistinguishable from the document not existing at all, not a
        # partial or redacted view. See test_gold_fidelity.py.
        # None for a header-field question. Carried through so
        # test_gold_fidelity.py can re-derive exactly which line's quantity
        # a restricted_unfiltered question's disambiguation depends on,
        # without re-parsing it back out of the question text -- see
        # test_restricted_unfiltered_questions_resolve_to_one_alice_readable_document.
        "line_number": line_number,
    }


def _no_reader(qid: str, name: str, field: str, text: str, all_principals: list[str],
               gold: dict[str, Any]) -> dict[str, Any]:
    """po_002: export-controlled + EMEA, and no named principal in the
    identity graph holds both the group/site grant AND the required
    clearance+jurisdiction. Not a two-party alice/ben pair -- the strongest
    possible ACL assertion, since there is no reader at all. Every principal
    must abstain, and every principal's retrieval must come back empty."""
    g = gold[name]
    return {
        "id": qid, "text": text.format(po=g["raw"]["po_number"]),
        "question_class": "restricted", "subtype": "no_reader",
        "gold_chunk_ids": [], "gold_answer": None,
        "source_documents": [name], "principal": None,
        "principals_checked": all_principals,
        "expected": "empty",  # every principal in principals_checked must get
        # zero chunks -- not just ben, everyone, since this document has no
        # valid reader in the identity graph at all.
        "note": f"{name} is export_controlled with allow_terms requiring "
                "group:buyers-emea + site:site_de07; no named principal holds "
                "both that grant and export_controlled clearance with a "
                "matching jurisdiction. Exercises the export-control path "
                f"nothing else in this gold set touches: field {field!r} has "
                f"a real value ({g['raw'][field]!r}) but is unreachable by anyone.",
    }


# ---------------------------------------------------------------------------
# The question set
# ---------------------------------------------------------------------------

def _build_questions(chunks: dict[str, list[Chunk]], gold: dict[str, Any]) -> list[dict[str, Any]]:
    q: list[dict[str, Any]] = []

    # -- single_hop (~10): one fact from one chunk, outside the header_lookup
    # field set (incoterms/payment_terms/ship_to_site), each on a document a
    # named principal actually has an ACL grant to read.
    q.append(_single_hop("q_sh_01", "po_001.pdf", "buyer_contact",
                          "Who is the buyer contact on PO {po}?", "u_alice", chunks, gold))
    q.append(_single_hop("q_sh_02", "po_004.pdf", "po_date",
                          "What is the PO date on PO {po}?", "u_chen", chunks, gold))
    q.append(_single_hop("q_sh_03", "po_004.pdf", "supplier_name",
                          "Who is the supplier on PO {po}?", "u_chen", chunks, gold))
    q.append(_single_hop("q_sh_04", "po_005.pdf", "currency",
                          "What currency is PO {po} denominated in?", "u_chen", chunks, gold))
    q.append(_single_hop("q_sh_05", "po_007.pdf", "buyer_contact",
                          "Who is the buyer contact on PO {po}?", "u_chen", chunks, gold))
    q.append(_single_hop("q_sh_06", "po_010.pdf", "po_date",
                          "What is the PO date on PO {po}?", "u_ewan", chunks, gold))
    q.append(_single_hop("q_sh_07", "po_012.pdf", "buyer_contact",
                          "Who is the buyer contact on PO {po}?", "u_ewan", chunks, gold))
    q.append(_single_hop("q_sh_08", "po_015.pdf", "total_amount",
                          "What is the total order amount on PO {po}?", "u_gita", chunks, gold))
    q.append(_single_hop("q_sh_09", "po_017.pdf", "buyer_contact",
                          "Who is the buyer contact on PO {po}?", "u_alice", chunks, gold))
    q.append(_single_hop("q_sh_10", "po_006.pdf", "supplier_id",
                          "What is the supplier's vendor code on PO {po}?",
                          "u_alice", chunks, gold))

    # -- header_lookup (~5): incoterms / payment terms / ship-to site
    q.append(_header_lookup("q_hl_01", "po_006.pdf", "incoterms",
                             "What incoterms apply to PO {po}?", "u_alice", chunks, gold))
    q.append(_header_lookup("q_hl_02", "po_016.pdf", "payment_terms",
                             "What are the payment terms on PO {po}?", "u_ewan", chunks, gold))
    q.append(_header_lookup("q_hl_03", "po_011.pdf", "ship_to_site",
                             "What is the ship-to site for PO {po}?", "u_chen", chunks, gold))
    q.append(_header_lookup("q_hl_04", "po_013.pdf", "incoterms",
                             "What incoterms apply to PO {po}?", "u_ewan", chunks, gold))
    q.append(_header_lookup("q_hl_05", "po_003.pdf", "payment_terms",
                             "What are the payment terms on PO {po}?", "u_chen", chunks, gold))

    # -- line_item (~8): at least 4 in a table split across chunks (po_008,
    # po_010 both split into 4 parts each; these four questions deliberately
    # pair one early-part line with one late-part line in the SAME document,
    # so retrieving the wrong part of the table is a distinguishable failure.
    q.append(_line_item("q_li_01", "po_008.pdf", 10, "unit_price",
                         "What is the unit price of line {line} on PO {po}?",
                         "u_alice", chunks, gold))
    q.append(_line_item("q_li_02", "po_008.pdf", 420, "unit_price",
                         "What is the unit price of line {line} on PO {po}?",
                         "u_alice", chunks, gold))
    q.append(_line_item("q_li_03", "po_010.pdf", 20, "description",
                         "What is the description of line {line} on PO {po}?",
                         "u_ewan", chunks, gold))
    q.append(_line_item("q_li_04", "po_010.pdf", 410, "unit_price",
                         "What is the unit price of line {line} on PO {po}?",
                         "u_ewan", chunks, gold))
    q.append(_line_item("q_li_05", "po_006.pdf", 20, "unit_price",
                         "What is the unit price of line {line} on PO {po}?",
                         "u_alice", chunks, gold))
    q.append(_line_item("q_li_06", "po_011.pdf", 20, "unit_price",
                         "What is the unit price of line {line} on PO {po}?",
                         "u_chen", chunks, gold))
    q.append(_line_item("q_li_07", "po_012.pdf", 20, "quantity",
                         "What quantity was ordered on line {line} of PO {po}?",
                         "u_ewan", chunks, gold))
    q.append(_line_item("q_li_08", "po_016.pdf", 30, "unit_price",
                         "What is the unit price of line {line} on PO {po}?",
                         "u_ewan", chunks, gold))

    # -- cross_document (~8): single-chunk retrieval cannot answer these.
    # Four "total spend with one supplier" (same-currency document pairs
    # only -- summing across currencies would need FX data this corpus does
    # not have) and four "same part number, different price" pairs.
    q.append(_cross_doc_total(
        "q_cd_01", "What did we spend with Omron Electronics Asia in total, in SGD?",
        ["po_000.pdf", "po_017.pdf"], "SGD", "u_alice", chunks, gold))
    q.append(_cross_doc_total(
        "q_cd_02", "What did we spend with Keyence Singapore Pte Ltd in total, in SGD?",
        ["po_001.pdf", "po_018.pdf"], "SGD", "u_alice", chunks, gold))
    q.append(_cross_doc_total(
        "q_cd_03", "What did we spend with Keyence Singapore Pte Ltd in total, in USD?",
        ["po_010.pdf", "po_013.pdf"], "USD", "u_ewan", chunks, gold))
    q.append(_cross_doc_total(
        "q_cd_04", "What did we spend with Kestrel Industrial AG in total, in USD?",
        ["po_009.pdf", "po_016.pdf"], "USD", "u_ewan", chunks, gold))
    q.append(_cross_doc_part(
        "q_cd_05", "What did we pay for part PLC-1756-L83 across our purchase orders?",
        "PLC-1756-L83", [("po_000.pdf", 10), ("po_014.pdf", 30)], "u_alice", chunks, gold))
    q.append(_cross_doc_part(
        "q_cd_06", "What did we pay for part TRM-BLK-2P5 across our purchase orders?",
        "TRM-BLK-2P5", [("po_001.pdf", 30), ("po_017.pdf", 80)], "u_alice", chunks, gold))
    q.append(_cross_doc_part(
        "q_cd_07", "What did we pay for part ENC-INC-1024 across our purchase orders?",
        "ENC-INC-1024", [("po_003.pdf", 30), ("po_004.pdf", 80)], "u_chen", chunks, gold))
    q.append(_cross_doc_part(
        "q_cd_08", "What did we pay for part TRM-BLK-2P5 across our purchase orders?",
        "TRM-BLK-2P5", [("po_013.pdf", 10), ("po_009.pdf", 10)], "u_ewan", chunks, gold))

    # -- ambiguous (~4): Kestrel Industrial AG (V100781) vs Kestrel Industrial
    # Pneumatics GmbH (V100782) -- the corpus's one genuinely confusable
    # supplier pair actually represented on both sides (Fastenal's other
    # entity, "...Supply Pte Ltd", never appears in this 20-document corpus,
    # so it cannot anchor an ambiguous question here; see this script's
    # report). po_004 (Pneumatics GmbH) and po_007 (AG) are both readable by
    # u_chen, so one principal can genuinely be asked all four.
    q.append(_ambiguous(
        "q_am_01", "What did we order from Kestrel Industrial?",
        {"note": "two distinct Kestrel Industrial entities placed orders",
         "suppliers": [
             {"supplier_name": gold["po_004.pdf"]["raw"]["supplier_name"],
              "document": "po_004.pdf"},
             {"supplier_name": gold["po_007.pdf"]["raw"]["supplier_name"],
              "document": "po_007.pdf"},
         ]},
        [("po_004.pdf", "header"), ("po_007.pdf", "header")], "u_chen", chunks))
    q.append(_ambiguous(
        "q_am_02", "What is Kestrel Industrial's vendor code?",
        [{"supplier_name": gold["po_004.pdf"]["raw"]["supplier_name"],
          "supplier_id": gold["po_004.pdf"]["raw"]["supplier_id"], "document": "po_004.pdf"},
         {"supplier_name": gold["po_007.pdf"]["raw"]["supplier_name"],
          "supplier_id": gold["po_007.pdf"]["raw"]["supplier_id"], "document": "po_007.pdf"}],
        [("po_004.pdf", "header"), ("po_007.pdf", "header")], "u_chen", chunks))
    q.append(_ambiguous(
        "q_am_03", "What are the payment terms on the Kestrel Industrial order?",
        [{"supplier_name": gold["po_004.pdf"]["raw"]["supplier_name"],
          "payment_terms": gold["po_004.pdf"]["raw"]["payment_terms"], "document": "po_004.pdf"},
         {"supplier_name": gold["po_007.pdf"]["raw"]["supplier_name"],
          "payment_terms": gold["po_007.pdf"]["raw"]["payment_terms"], "document": "po_007.pdf"}],
        [("po_004.pdf", "header"), ("po_007.pdf", "header")], "u_chen", chunks))
    q.append(_ambiguous(
        "q_am_04", "How much did we spend with Kestrel Industrial in total?",
        [{"supplier_name": gold["po_004.pdf"]["raw"]["supplier_name"],
          "total_amount": gold["po_004.pdf"]["raw"]["total_amount"],
          "currency": gold["po_004.pdf"]["raw"]["currency"], "document": "po_004.pdf"},
         {"supplier_name": gold["po_007.pdf"]["raw"]["supplier_name"],
          "total_amount": gold["po_007.pdf"]["raw"]["total_amount"],
          "currency": gold["po_007.pdf"]["raw"]["currency"], "document": "po_007.pdf"}],
        [("po_004.pdf", "header"), ("po_007.pdf", "header")], "u_chen", chunks))

    # -- unanswerable (~8): must be plausible. Three subtypes.
    q.append(_unanswerable(
        "q_ua_01", f"Who is the buyer contact on PO {gold['po_004.pdf']['raw']['po_number']}?",
        "absent", "buyer_contact is not present on po_004.pdf (see meta.absent_fields)",
        "u_chen", ["po_004.pdf"]))
    q.append(_unanswerable(
        "q_ua_02", f"What are the payment terms on PO {gold['po_005.pdf']['raw']['po_number']}?",
        "absent", "payment_terms is not present on po_005.pdf", "u_chen", ["po_005.pdf"]))
    q.append(_unanswerable(
        "q_ua_03", f"What incoterms apply to PO {gold['po_014.pdf']['raw']['po_number']}?",
        "absent", "incoterms is not present on po_014.pdf", "u_alice", ["po_014.pdf"]))
    q.append(_unanswerable(
        "q_ua_04",
        f"Has the shipment for PO {gold['po_000.pdf']['raw']['po_number']} been "
        "confirmed as received?",
        "out_of_scope", "the corpus has no delivery/receiving records for any PO",
        "u_alice", ["po_000.pdf"]))
    q.append(_unanswerable(
        "q_ua_05",
        f"Did the goods on PO {gold['po_009.pdf']['raw']['po_number']} pass incoming "
        "quality inspection?",
        "out_of_scope", "the corpus has no quality-inspection records", "u_ewan", ["po_009.pdf"]))
    q.append(_unanswerable(
        "q_ua_06",
        f"What warranty terms apply under the Master Supply Agreement referenced on "
        f"PO {gold['po_007.pdf']['raw']['po_number']}?",
        "out_of_scope", "the PO references an MSA by name but its terms are not in the corpus",
        "u_chen", ["po_007.pdf"]))
    q.append(_unanswerable(
        "q_ua_07", "What was the unit price?",
        "underspecified", "no PO or part number given -- every PO has multiple, different prices",
        "u_alice", []))
    q.append(_unanswerable(
        "q_ua_08", "When is delivery promised?",
        "underspecified", "no PO or line given -- every PO has multiple, different promised dates",
        "u_alice", []))

    # -- restricted_filtered (~8) + restricted_unfiltered (~4): u_alice /
    # u_ben, same group+site, differing clearance. po_000/po_014/po_018 are
    # the only PURCHASE_ORDER documents in this corpus where alice's
    # confidential clearance (vs. ben's internal) is the sole reason she can
    # read it and he cannot -- verified via _readable_by() against the real
    # per-document ACL, not assumed from the sensitivity label alone.
    #
    # Every question here asks for a SPECIFIC value (a price, a date, a part
    # number) rather than a vague one: abstention is trivially achievable by
    # a system that just retrieves badly, so the only way this actually
    # exercises the restriction is if alice's expected answer is precise
    # enough that "ben got nothing" and "alice got the right value" are two
    # separately checkable assertions, not one.
    q.append(_restricted("q_re_01", "po_000.pdf", "total_amount", None,
                          "What is the total order amount on PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_02", "po_000.pdf", "unit_price", 10,
                          "What is the unit price of line 10 on PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_03", "po_014.pdf", "po_date", None,
                          "What is the PO date on PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_04", "po_014.pdf", "unit_price", 10,
                          "What is the unit price of line 10 on PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_05", "po_014.pdf", "unit_price", 40,
                          "What is the unit price of line 40 on PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_06", "po_018.pdf", "part_number", 40,
                          "What part number is on line 40 of PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_07", "po_018.pdf", "unit_price", 20,
                          "What is the unit price of line 20 on PO {po}?",
                          "restricted_filtered", chunks, gold))
    q.append(_restricted("q_re_08", "po_018.pdf", "total_amount", None,
                          "What is the total order amount on PO {po}?",
                          "restricted_filtered", chunks, gold))

    # -- restricted_unfiltered (4): same alice/ben pair, same specific-value
    # requirement, but the question text deliberately carries NO PO number --
    # infer_filters() returns {} for every one of these (pinned by
    # test_restricted_unfiltered_questions_produce_no_filter), so retrieval
    # runs against the whole corpus rather than a single document's ~3-chunk
    # pool. All 8 restricted_filtered questions above are decided with the
    # filter already narrowing the field; a ranking path that bypasses the
    # filter (or a permission check applied only after filtering) could pass
    # every one of them and still leak corpus-wide. Phrased on supplier name
    # + a line's description rather than the PO number or part number --
    # both of which repeat verbatim across many other documents in this
    # corpus.
    #
    # Supplier name ALONE is not always enough to disambiguate, and this was
    # found the hard way, not assumed: Omron Electronics Asia and Keyence
    # Singapore Pte Ltd each place multiple orders in this corpus, and
    # u_alice -- not just u_ben -- can read more than one of them (Omron:
    # po_000 confidential + po_017 internal; Keyence Singapore: po_018
    # confidential + po_001 internal). A first version of q_re_10 asked "the
    # ControlLogix processor module on the Omron Electronics Asia order" as
    # if there were only one -- u_ben answered it correctly and confidently
    # from po_017, which he can legitimately read; not an ACL leak, but not
    # a question with a unique gold answer either. SKF Bearings Manufacturing
    # (q_re_11/12's supplier) places only one order in this corpus, so it
    # never had this problem. Where supplier alone collides, the line's
    # quantity is folded into the question text as the second disambiguating
    # fact -- verified, not assumed, against every OTHER alice-readable
    # document from the same supplier: see
    # test_restricted_unfiltered_questions_resolve_to_one_alice_readable_document.
    q.append(_restricted(
        "q_re_10", "po_000.pdf", "unit_price", 10,
        "What did we pay per unit for the 250-unit order of ControlLogix "
        "processor modules from Omron Electronics Asia?",
        "restricted_unfiltered", chunks, gold))
    q.append(_restricted(
        "q_re_11", "po_014.pdf", "unit_price", 40,
        "What did we pay per unit for the 24VDC relay module on the SKF "
        "Bearings Manufacturing order?",
        "restricted_unfiltered", chunks, gold))
    q.append(_restricted(
        "q_re_12", "po_014.pdf", "po_date", None,
        "When was the SKF Bearings Manufacturing purchase order dated?",
        "restricted_unfiltered", chunks, gold))
    q.append(_restricted(
        "q_re_13", "po_018.pdf", "unit_price", 30,
        "What did we pay per unit for the 100-unit order of incremental "
        "encoders from Keyence Singapore?",
        "restricted_unfiltered", chunks, gold))

    # -- no_reader (1): po_002 -- export_controlled + EMEA, zero readers in
    # the named identity graph. Not excluded as a gap; recorded as the
    # strongest possible ACL assertion, and the only question in this gold
    # set that exercises the export-control path.
    q.append(_no_reader(
        "q_re_09", "po_002.pdf", "total_amount",
        "What is the total order amount on PO {po}?",
        ["u_alice", "u_ben", "u_chen", "u_dara", "u_ewan", "u_frank", "u_gita"], gold))

    return q


# ---------------------------------------------------------------------------
# Verification -- against the live index, not assumed from the re-chunk above
# ---------------------------------------------------------------------------

def _verify_against_index(questions: list[dict[str, Any]], store: LocalVectorStore,
                           fingerprint: str) -> None:
    store_fp = store.settings_fingerprint()
    if store_fp is not None and store_fp != fingerprint:
        raise ChunkSettingsMismatch(
            f"live index at {store.path} has fingerprint {store_fp!r}; this gold set "
            f"was built against {fingerprint!r}. Rebuild the index or regenerate gold."
        )
    live_ids = {c.id for c in store._chunks}
    missing: list[tuple[str, str]] = []
    for q in questions:
        for cid in q["gold_chunk_ids"]:
            if cid not in live_ids:
                missing.append((q["id"], cid))
    if missing:
        raise RuntimeError(
            f"{len(missing)} gold_chunk_id(s) do not resolve against the live index "
            f"at {store.path}: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )


def _verify_restricted_pairs(questions: list[dict[str, Any]], users: dict[str, Principal],
                              synthetic_dir: Path, gold: dict[str, Any]) -> None:
    for q in questions:
        if q["question_class"] != "restricted":
            continue
        name = q["source_documents"][0]
        acl = _doc_acl(synthetic_dir, name)
        if q["subtype"] == "no_reader":
            readers = [uid for uid in q["principals_checked"] if users[uid].may_read(acl)]
            if readers:
                raise RuntimeError(f"{q['id']}: {name} unexpectedly has readers: {readers}")
            continue
        alice, ben = users["u_alice"], users["u_ben"]
        if not alice.may_read(acl):
            raise RuntimeError(f"{q['id']}: u_alice cannot read {name} -- not actually answerable")
        if ben.may_read(acl):
            raise RuntimeError(f"{q['id']}: u_ben CAN read {name} -- restriction does not fire")


def _verify_restricted_unfiltered_disambiguation(
    questions: list[dict[str, Any]], users: dict[str, Principal],
    synthetic_dir: Path, gold: dict[str, Any],
) -> None:
    """A restricted_unfiltered question carries no PO number by design, so
    its supplier name is the only thing standing between retrieval and the
    right document -- if u_alice herself can read a SECOND document from
    that supplier, the question does not have a unique answer even though
    nothing about the ACL is wrong. This is exactly how q_re_10 broke the
    first time: Omron Electronics Asia places 4 orders in this corpus, and
    u_alice can read 2 of them (po_000 confidential, po_017 internal) --
    u_ben answered from po_017, which he can legitimately read, not a leak.

    Supplier name alone is enough when only one alice-readable document
    shares it (true for SKF Bearings Manufacturing, q_re_11/12's supplier --
    it places a single order in this corpus). Where it collides, the
    question folds the line's quantity in as a second disambiguating fact
    (see q_re_10/13's text) -- checked here against every OTHER
    alice-readable same-supplier document's matching-part lines, not
    assumed from the question's own wording.
    """
    alice = users["u_alice"]
    for q in questions:
        if q["subtype"] != "restricted_unfiltered":
            continue
        name = q["source_documents"][0]
        supplier = gold[name]["raw"]["supplier_name"]
        same_supplier = [
            other for other in gold
            if gold[other]["raw"]["supplier_name"] == supplier
            and alice.may_read(_doc_acl(synthetic_dir, other))
        ]
        others = [o for o in same_supplier if o != name]
        if not others:
            continue  # supplier alone is unique among what u_alice can read

        line_number = q["line_number"]
        if line_number is None:
            raise RuntimeError(
                f"{q['id']}: supplier {supplier!r} is alice-readable across "
                f"{sorted(same_supplier)}, and this is a header-field question "
                "with no line-level fact available to disambiguate it"
            )
        target_line = _line(gold, name, line_number)
        part, qty = target_line["part_number"], target_line["quantity"]
        colliding = [
            other for other in others
            if any(ln["part_number"] == part and ln["quantity"] == qty
                   for ln in gold[other]["raw"]["lines"])
        ]
        if colliding:
            raise RuntimeError(
                f"{q['id']}: part {part!r} qty {qty!r} also appears on "
                f"alice-readable {sorted(colliding)} from the same supplier "
                f"{supplier!r} -- the question does not uniquely resolve"
            )
        # The quantity has to disambiguate in the TEXT, not just in the
        # data behind it -- a question whose wording never mentions it
        # reads exactly like the original, broken q_re_10, even if this
        # specific line's quantity happens to be structurally unique.
        if str(qty) not in q["text"]:
            raise RuntimeError(
                f"{q['id']}: supplier {supplier!r} needs quantity {qty!r} to "
                f"disambiguate from {sorted(others)}, but the question text "
                f"never states it: {q['text']!r}"
            )


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--extraction-gold", type=Path, default=Path("data/gold/extraction"))
    ap.add_argument("--acl", type=Path, default=Path("data/acl"))
    ap.add_argument("--out", type=Path, default=Path("data/gold/retrieval/questions.json"))
    ap.add_argument("--seed", type=int, default=2608,
                     help="the seed the corpus (gen_corpus.py) was generated with")
    args = ap.parse_args()

    settings = get_settings()
    fingerprint = settings_fingerprint(settings.chunk)

    gold = _load_extraction_gold(args.extraction_gold)
    users = _load_users(args.acl)
    docs = SqliteDocStore(settings.paths.data / "docstore.sqlite")
    # Every ingested document, not just the POs this gold set indexes (see
    # _build_chunk_index) -- corpus_fingerprint()'s whole point is catching
    # drift regardless of which documents the gold happens to reference.
    corpus_fp = corpus_fingerprint(docs.content_hashes())
    chunks, doc_ids = _build_chunk_index(docs, settings, gold)

    questions = _build_questions(chunks, gold)

    store = LocalVectorStore(settings.paths.data / "vector_store.pkl")
    _verify_against_index(questions, store, fingerprint)
    _verify_restricted_pairs(questions, users, args.corpus, gold)
    _verify_restricted_unfiltered_disambiguation(questions, users, args.corpus, gold)

    by_class: dict[str, int] = {}
    by_subtype: dict[str, int] = {}
    for q in questions:
        by_class[q["question_class"]] = by_class.get(q["question_class"], 0) + 1
        by_subtype[q["subtype"]] = by_subtype.get(q["subtype"], 0) + 1

    out = {
        "provenance": {
            "seed": args.seed,
            "settings_fingerprint": fingerprint,
            "corpus_fingerprint": corpus_fp,
            # Deterministic and seed-derived, matching gen_corpus.py's own
            # SOURCE_DATE_EPOCH convention -- not datetime.now(), which would
            # make this file (and the "generate twice, diff bytes" test)
            # non-deterministic exactly the way the PDF timestamps used to be.
            "corpus_generated_at": datetime.fromtimestamp(
                _PDF_EPOCH_BASE + args.seed, tz=UTC
            ).isoformat(),
        },
        "counts": {"by_class": by_class, "by_subtype": by_subtype, "total": len(questions)},
        "questions": questions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")

    print(f"wrote {len(questions)} questions -> {args.out}")
    print("  by class:", by_class)
    print("  by subtype:", by_subtype)


if __name__ == "__main__":
    main()

"""Guards the corpus invariant: gold is truth, the PDF is rendered from it.

These tests are the reason the corpus can be trusted as an eval baseline. If
they fail, every downstream extraction metric is meaningless — a gold set that
disagrees with its own documents measures nothing.

Skipped when the corpus has not been generated, so a fresh clone still runs
green on `make test` before `make corpus`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
GOLD = ROOT / "data" / "gold" / "extraction"
ACL = ROOT / "data" / "acl"

pytestmark = pytest.mark.skipif(
    not list(GOLD.glob("*.json")), reason="corpus not generated; run `make corpus`"
)


def _gold_files() -> list[Path]:
    return sorted(GOLD.glob("*.json"))


def _flat_text(pdf: Path) -> str:
    from pypdf import PdfReader
    return " ".join(
        "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages).split()
    )


@pytest.mark.parametrize("gold_path", _gold_files(), ids=lambda p: p.stem)
def test_every_gold_value_appears_in_the_pdf(gold_path: Path):
    g = json.loads(gold_path.read_text())
    flat = _flat_text(SYN / g["document"])
    for field in ("po_number", "po_date", "supplier_name", "supplier_id", "incoterms",
                  "payment_terms", "currency", "buyer_contact", "total_amount"):
        value = g["raw"].get(field)
        if value is None:
            continue
        assert str(value) in flat, f"{g['document']}: gold {field}={value!r} not on the page"
    for line in g["raw"]["lines"]:
        assert line["part_number"] in flat


@pytest.mark.parametrize("gold_path", _gold_files(), ids=lambda p: p.stem)
def test_absent_fields_are_absent_from_gold(gold_path: Path):
    """'Correctly absent' is only a valid outcome if the gold really is null."""
    g = json.loads(gold_path.read_text())
    for field in g["meta"]["absent_fields"]:
        assert g["raw"].get(field) is None
        assert g["normalised"].get(field) is None


@pytest.mark.parametrize("gold_path", _gold_files(), ids=lambda p: p.stem)
def test_page_count_is_measured_not_predicted(gold_path: Path):
    from pypdf import PdfReader
    g = json.loads(gold_path.read_text())
    actual = len(PdfReader(str(SYN / g["document"])).pages)
    assert g["meta"]["n_pages"] == actual
    assert g["meta"]["wraps_pages"] == (actual > 1)


def test_every_document_has_an_acl_sidecar():
    from isc.models.acl import AclSet, Sensitivity
    pdfs = sorted(SYN.glob("*.pdf"))
    assert pdfs, "no documents generated"
    for pdf in pdfs:
        sidecar = pdf.with_suffix(".pdf.acl.json")
        assert sidecar.exists(), f"{pdf.name} has no ACL sidecar"
        raw = json.loads(sidecar.read_text())
        acl = AclSet(
            allow_terms=frozenset(raw["allow_terms"]),
            deny_terms=frozenset(raw["deny_terms"]),
            sensitivity=Sensitivity(raw["sensitivity"]),
            jurisdictions=frozenset(raw["jurisdictions"]),
        )
        assert acl.allow_terms


def test_corpus_covers_every_sensitivity_level():
    """Stratification, not sampling. A missing level silently disables a branch
    of the ACL suite."""
    levels = {
        json.loads(p.read_text())["sensitivity"]
        for p in SYN.glob("*.acl.json")
    }
    assert {"internal", "confidential", "export_controlled"} <= levels


def test_corpus_covers_every_region():
    sites = {json.loads(p.read_text())["meta"]["site_id"] for p in _gold_files()}
    assert len(sites) >= 3, f"only {sites} represented; principals will have no corpus"


def test_corpus_has_multi_page_documents():
    assert any(json.loads(p.read_text())["meta"]["wraps_pages"] for p in _gold_files())


def test_corpus_has_ambiguous_dates():
    """The most common silent extraction error must be represented."""
    assert any(json.loads(p.read_text())["meta"]["ambiguous_date_fields"]
               for p in _gold_files())


@pytest.mark.acl
def test_adversarial_pairs_fire_against_the_real_corpus():
    """A pair that cannot fire is zero coverage and must not read as passing."""
    from isc.ingest.local_source import LocalDirectorySource
    from isc.models.acl import Principal, Sensitivity

    users = json.loads((ACL / "users.json").read_text())
    pairs = json.loads((ACL / "adversarial_pairs.json").read_text())
    present = {json.loads(p.read_text())["doc_type"] for p in _gold_files()}
    items = list(LocalDirectorySource(SYN).items())

    def principal(uid: str) -> Principal:
        u = users[uid]
        return Principal(id=uid, group_ids=frozenset(u["groups"]),
                         site_ids=frozenset(u["sites"]),
                         clearance=Sensitivity(u["clearance"]),
                         jurisdictions=frozenset(u["jurisdictions"]))

    fired = 0
    for pair in pairs:
        if not set(pair["requires"]) <= present:
            continue  # PENDING: doc type not in the corpus yet
        a, b = principal(pair["can_see"]), principal(pair["cannot_see"])
        only_a = [i for i in items if a.may_read(i.acl) and not b.may_read(i.acl)]
        assert only_a, f"pair {pair['can_see']}>{pair['cannot_see']} never fires: {pair['why']}"
        fired += 1

    assert fired >= 4, f"only {fired} adversarial pairs exercised by this corpus"

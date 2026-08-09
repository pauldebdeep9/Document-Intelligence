"""Guards P1-01's fidelity claim: every gold header value and line part_number
the corpus claims to contain must actually appear in the parsed Document.text().

Line descriptions are deliberately excluded. They wrap across rendered table
rows ("Switched mode power supply 24V" / "10A"), and the wrapped continuation
lands in the flattened text after the rest of that row's columns, not adjacent
to the description it belongs to. Matching that needs column-aware reflow,
which is table reconstruction's job (P1-04), not this parser's. Testing it
here would either fail spuriously on wrapped rows or push reflow logic into
the wrong item.

Skipped when the corpus has not been generated, same convention as
test_corpus_fidelity.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isc.models.acl import AclSet
from isc.models.document import Document
from isc.parse.chain import NativeTextParser

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
GOLD = ROOT / "data" / "gold" / "extraction"

pytestmark = pytest.mark.skipif(
    not list(GOLD.glob("*.json")), reason="corpus not generated; run `make corpus`"
)

# Same field set as test_corpus_fidelity.py's PDF-level fidelity check. No
# line descriptions -- see module docstring.
_FIELDS = ("po_number", "po_date", "supplier_name", "supplier_id", "incoterms",
           "payment_terms", "currency", "buyer_contact", "total_amount")


def _gold_files() -> list[Path]:
    return sorted(GOLD.glob("*.json"))


def _norm(s: object) -> str:
    return " ".join(str(s).split())


def _parse(gold: dict) -> Document:
    pdf = SYN / gold["document"]
    doc = Document(id="doc_fidelity", source_uri=gold["document"], content_sha256="x",
                    acl=AclSet(allow_terms=frozenset({"everyone:*"})))
    return NativeTextParser().parse(doc, pdf.read_bytes())


@pytest.mark.parametrize("gold_path", _gold_files(), ids=lambda p: p.stem)
def test_every_gold_header_value_appears_in_parsed_text(gold_path: Path):
    g = json.loads(gold_path.read_text())
    flat = _norm(_parse(g).text())
    for field in _FIELDS:
        value = g["raw"].get(field)
        if value is None:
            continue
        assert _norm(value) in flat, f"{g['document']}: gold {field}={value!r} not in parsed text"


@pytest.mark.parametrize("gold_path", _gold_files(), ids=lambda p: p.stem)
def test_every_line_part_number_appears_in_parsed_text(gold_path: Path):
    g = json.loads(gold_path.read_text())
    flat = _norm(_parse(g).text())
    for line in g["raw"]["lines"]:
        assert _norm(line["part_number"]) in flat, \
            f"{g['document']}: part_number {line['part_number']!r} not in parsed text"

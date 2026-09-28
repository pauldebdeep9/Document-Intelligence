"""scripts/gen_gold.py's _cross_doc_part (D8, AG-10) on a synthetic document.

The real extraction gold has no line with a null unit price, so the "priced
lines only" half of D8 cannot fail against it -- a generator that kept
unpriced lines would write identical gold. This builds the one case the
corpus lacks: a readable document holding the part twice, once unpriced.
"""

from __future__ import annotations

import json

from isc.models.acl import AclSet, Principal, Sensitivity
from isc.models.chunk import Chunk
from scripts.gen_gold import _cross_doc_part

PART = "ZZ-100-A"
DOC = "po_900.pdf"


def test_part_price_gold_skips_unpriced_lines(tmp_path):
    (tmp_path / f"{DOC}.acl.json").write_text(json.dumps({
        "allow_terms": ["group:buyers"], "deny_terms": [],
        "sensitivity": "internal", "jurisdictions": [],
    }))
    users = {"u_x": Principal(id="u_x", group_ids=frozenset({"buyers"}),
                              clearance=Sensitivity.INTERNAL)}
    gold = {DOC: {"raw": {"currency": "SGD", "lines": [
        {"line_number": 10, "part_number": PART, "unit_price": "5.00"},
        {"line_number": 20, "part_number": PART, "unit_price": None},
    ]}}}
    table = Chunk(id="chk_table", document_id="d_900", ordinal=1, text="| 10 | 20 |",
                  acl=AclSet(allow_terms=frozenset({"group:buyers"})), is_table=True,
                  line_range=(10, 20))

    q = _cross_doc_part("q_x", f"What did we pay for part {PART}?", PART, "u_x",
                        {DOC: [table]}, gold, users, tmp_path)

    assert q["gold_answer"] == [
        {"document": DOC, "line_number": 10, "unit_price": "5.00", "currency": "SGD"}]
    assert q["gold_chunk_ids"] == ["chk_table"]
    assert q["source_documents"] == [DOC]

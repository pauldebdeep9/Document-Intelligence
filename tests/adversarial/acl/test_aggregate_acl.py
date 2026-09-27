"""Adversarial ACL suite for the aggregate (records) path.

An aggregate leaks in ways a chunk list does not: a total that includes a
restricted order discloses its amount, and a count discloses its existence,
even when no restricted text is shown. Same rules as the rest of this
directory: none of these may be marked xfail, skipped, or weakened.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from isc.aggregate.answerer import RecordAnswerer
from isc.aggregate.source import visible_records
from isc.models.acl import Principal, Sensitivity
from isc.models.answer import AbstentionReason
from isc.storage.sqlite_docstore import SqliteDocStore
from tests.aggregate_world import PlanChat, Po, build, plan, settings

pytestmark = pytest.mark.acl

D = Decimal
KEYENCE = "Keyence Singapore Pte Ltd"
QUESTION = f"What did we spend with {KEYENCE} in total, in SGD?"
PLAN = plan("total_spend", KEYENCE, currency="SGD")

# alice (confidential) can read both; ben (internal) only the internal one.
ALICE = Principal(id="u_alice", group_ids=frozenset({"buyers-apac"}),
                  clearance=Sensitivity.CONFIDENTIAL)
BEN = Principal(id="u_ben", group_ids=frozenset({"buyers-apac"}),
                clearance=Sensitivity.INTERNAL)
FRANK = Principal(id="u_frank", group_ids=frozenset({"contractors"}),
                  clearance=Sensitivity.PUBLIC)

OPEN = Po("d_open", "4500000001", KEYENCE, "V103014", "SGD", D("1000.00"),
          allow={"group:buyers-apac"}, sensitivity=Sensitivity.INTERNAL)
HIDDEN = Po("d_hidden", "4500000002", KEYENCE, "V103014", "SGD", D("2675022.29"),
            allow={"group:buyers-apac"}, sensitivity=Sensitivity.CONFIDENTIAL)
# Extracted but never indexed, and unreadable by ben: must not even surface
# as "not checked" for him.
HIDDEN_UNEXTRACTED = Po("d_hidden2", "4500000003", KEYENCE, "V103014", "SGD", D("5.00"),
                        allow={"group:buyers-apac"}, sensitivity=Sensitivity.CONFIDENTIAL,
                        extracted=False)


def _answer(tmp_path, principal, pos=(OPEN, HIDDEN, HIDDEN_UNEXTRACTED)):
    docs, store = build(tmp_path, list(pos))
    return RecordAnswerer(PlanChat(PLAN), docs, store, settings()).try_answer(QUESTION, principal)


def assert_nothing_hidden_contributes(answer, principal, hidden: list[Po]) -> None:
    """The gate. Checked structurally (every supporting chunk passes this
    principal's own may_read) AND by content (no hidden PO number or amount,
    and no sum that includes one, appears anywhere in the text)."""
    for sc in answer.supporting:
        assert principal.may_read(sc.chunk.acl), f"leaked chunk {sc.chunk.id}"
    for po in hidden:
        assert po.po_number not in answer.text, f"hidden PO {po.po_number} named"
        if po.total is not None:
            assert f"{po.total:,.2f}" not in answer.text, f"hidden amount {po.total} shown"
            assert f"{OPEN.total + po.total:,.2f}" not in answer.text, "hidden amount summed in"


def test_restricted_order_never_contributes(tmp_path):
    ans = _answer(tmp_path, BEN)
    assert not ans.abstained
    assert_nothing_hidden_contributes(ans, BEN, [HIDDEN, HIDDEN_UNEXTRACTED])
    assert "across the 1 purchase order visible to you: 1,000.00 SGD" in ans.text
    assert "Not checked" not in ans.text   # the hidden unextracted doc is not counted


def test_the_same_question_as_the_cleared_principal_includes_it(tmp_path):
    ans = _answer(tmp_path, ALICE)
    assert "2,676,022.29 SGD" in ans.text
    assert "Not checked: 1 purchase order" in ans.text


def test_nothing_visible_is_indistinguishable_from_nothing_existing(tmp_path):
    """frank can read none of these orders. His answer must be identical to
    the answer for a supplier with no orders at all -- otherwise the
    difference discloses that restricted orders exist."""
    hidden_only = _answer(tmp_path / "a", FRANK)
    nothing = _answer(tmp_path / "b", FRANK, pos=())
    for ans in (hidden_only, nothing):
        assert ans.abstained and ans.abstention_reason is AbstentionReason.NO_RESULTS
        assert ans.supporting == []
    assert hidden_only.text == nothing.text
    assert hidden_only.user_facing_reason() == nothing.user_facing_reason()


def test_record_of_an_unreadable_document_is_never_loaded(tmp_path):
    """Structural, not per-dataset: the permission check happens before the
    record is read, so a bug downstream of source.py cannot leak what was
    never in memory."""
    docs, store = build(tmp_path, [OPEN, HIDDEN])
    loaded: list[str] = []

    class SpyDocs(SqliteDocStore):
        def get_record(self, doc_id):
            loaded.append(doc_id)
            return super().get_record(doc_id)

    spy = SpyDocs(tmp_path / "docstore.sqlite")
    view = visible_records(BEN, spy, store)
    assert [r.document.id for r in view.records] == ["d_open"]
    assert "d_hidden" not in loaded


def test_export_controlled_order_needs_a_matching_jurisdiction(tmp_path):
    ec = Po("d_ec", "4500000009", KEYENCE, "V103014", "SGD", D("77.00"),
            allow={"group:engineering", "group:trade-compliance"},
            sensitivity=Sensitivity.EXPORT_CONTROLLED, jurisdictions=frozenset({"SG"}))
    gita = Principal(id="u_gita", group_ids=frozenset({"trade-compliance"}),
                     clearance=Sensitivity.EXPORT_CONTROLLED, jurisdictions=frozenset({"SG"}))
    ewan = Principal(id="u_ewan", group_ids=frozenset({"engineering"}),
                     clearance=Sensitivity.CONFIDENTIAL, jurisdictions=frozenset({"US"}))
    assert "77.00 SGD" in _answer(tmp_path / "g", gita, pos=[ec]).text
    denied = _answer(tmp_path / "e", ewan, pos=[ec])
    assert denied.abstained and "77.00" not in denied.text


def test_the_gate_goes_red_when_the_source_skips_the_check(tmp_path, monkeypatch):
    """A gate that has never failed tests nothing. Mutate the records path
    so it loads records as a fully-cleared principal instead of the asker
    (the realistic bug: a service identity doing the query), and the gate
    above must fail -- on the structural check first, before any text is
    even compared."""
    import isc.aggregate.answerer as answerer

    real = answerer.visible_records
    monkeypatch.setattr(answerer, "visible_records",
                        lambda principal, docs, store: real(ALICE, docs, store))
    leaky = _answer(tmp_path, BEN)
    with pytest.raises(AssertionError, match="leaked chunk"):
        assert_nothing_hidden_contributes(leaky, BEN, [HIDDEN])
    assert HIDDEN.po_number in leaky.text   # and the content check would have caught it too

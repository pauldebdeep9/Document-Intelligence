"""Adversarial ACL suite.

This file exists from the first commit, before retrieval is implemented, because
'we will add permissions later' is the failure mode it is designed to prevent.
Every test here is a hard invariant: none of them may be marked xfail, skipped,
or weakened to make a feature land.
"""

from __future__ import annotations

import pytest

from isc.common.errors import AclViolation
from isc.models.acl import AclSet, Grant, Principal, PrincipalType, Sensitivity
from isc.models.chunk import Chunk
from tests.conftest import make_chunk

pytestmark = pytest.mark.acl


class TestFailClosed:
    def test_empty_allow_terms_rejected_at_construction(self):
        with pytest.raises(ValueError):
            AclSet(allow_terms=frozenset())

    def test_chunk_cannot_be_built_without_acl(self):
        with pytest.raises(Exception):
            Chunk(id="c", document_id="d", ordinal=0, text="x")  # type: ignore[call-arg]

    def test_everyone_must_be_explicit(self):
        """'Public' is a deliberate grant, never the result of an omission."""
        acl = AclSet(allow_terms=frozenset({"everyone:*"}))
        assert Principal(id="anyone").may_read(acl)


class TestGroupExpansion:
    def test_group_grant_reaches_member(self, alice):
        acl = AclSet(allow_terms=frozenset({"group:buyers-apac"}))
        assert alice.may_read(acl)

    def test_group_grant_does_not_reach_non_member(self, frank):
        acl = AclSet(allow_terms=frozenset({"group:buyers-apac"}))
        assert not frank.may_read(acl)

    def test_expansion_materialises_user_terms(self):
        acl = AclSet.from_grants(
            [Grant(principal_id="buyers-apac", principal_type=PrincipalType.GROUP)],
            {"buyers-apac": {"u_alice", "u_ben"}},
        )
        assert "user:u_alice" in acl.allow_terms
        assert "group:buyers-apac" in acl.allow_terms


class TestDenyPrecedence:
    def test_deny_beats_allow(self, alice):
        acl = AclSet(allow_terms=frozenset({"group:buyers-apac"}),
                     deny_terms=frozenset({"user:u_alice"}))
        assert not alice.may_read(acl)

    def test_deny_on_group_beats_user_allow(self, alice):
        acl = AclSet(allow_terms=frozenset({"user:u_alice"}),
                     deny_terms=frozenset({"group:quality-sg"}))
        assert not alice.may_read(acl)


class TestSensitivity:
    def test_clearance_below_label_blocks(self, frank):
        acl = AclSet(allow_terms=frozenset({"group:contractors"}),
                     sensitivity=Sensitivity.CONFIDENTIAL)
        assert not frank.may_read(acl)

    def test_export_control_requires_matching_jurisdiction(self, alice, gita):
        acl = AclSet(allow_terms=frozenset({"everyone:*"}),
                     sensitivity=Sensitivity.EXPORT_CONTROLLED,
                     jurisdictions=frozenset({"US"}))
        assert gita.may_read(acl)
        assert not alice.may_read(acl)   # SG only

    def test_export_controlled_with_no_jurisdictions_is_readable_by_nobody(self, gita):
        acl = AclSet(allow_terms=frozenset({"everyone:*"}),
                     sensitivity=Sensitivity.EXPORT_CONTROLLED)
        assert not gita.may_read(acl)


class TestRetrievalIsolation:
    """The behavioural test: filtering happens before ranking, not after."""

    def test_restricted_chunk_never_returned(self, alice, frank, vec):
        from pathlib import Path
        from isc.storage.local_vector import LocalVectorStore

        store = LocalVectorStore(Path("/tmp/acl_test_store.pkl"))
        secret = make_chunk("c_secret", "confidential pricing for supplier X",
                            {"group:buyers-apac"})
        public = make_chunk("c_public", "general shipping policy", {"everyone:*"})
        store.add([secret, public], [[1.0, 0, 0], [0.0, 1.0, 0]])

        # Query that is a near-perfect match for the secret chunk.
        got = store.search([1.0, 0, 0], frank, k=10)
        assert all(s.chunk.id != "c_secret" for s in got)
        assert [s.chunk.id for s in store.search([1.0, 0, 0], alice, k=10)][0] == "c_secret"

    def test_lexical_path_filters_too(self, frank, alice):
        """The BM25 path is a separate code path and a separate chance to leak."""
        from pathlib import Path
        from isc.storage.local_vector import LocalVectorStore

        store = LocalVectorStore(Path("/tmp/acl_test_store2.pkl"))
        secret = make_chunk("c_s", "part ABC-991 unit price 12.50", {"group:buyers-apac"})
        store.add([secret], [[1.0, 0, 0]])
        assert store.search_lexical("ABC-991", frank) == []
        assert len(store.search_lexical("ABC-991", alice)) == 1

    def test_no_results_and_no_permitted_results_are_indistinguishable(self):
        """Confirming a document exists is itself a disclosure."""
        from isc.models.answer import AbstentionReason, Answer

        a = Answer.abstain("q", AbstentionReason.NO_PERMITTED_RESULTS)
        b = Answer.abstain("q", AbstentionReason.NO_RESULTS)
        assert a.text == b.text
        assert a.user_facing_reason() == b.user_facing_reason()


class TestIndexTimeGuard:
    def test_index_rejects_chunk_without_acl_terms(self):
        """Last line of defence: even a hand-constructed chunk cannot enter."""
        from pathlib import Path
        from isc.storage.local_vector import LocalVectorStore
        from isc.models.chunk import Chunk

        store = LocalVectorStore(Path("/tmp/acl_test_store3.pkl"))
        chunk = make_chunk("c_ok", "text", {"everyone:*"})
        object.__setattr__(chunk.acl, "allow_terms", frozenset())
        with pytest.raises(AclViolation):
            store.add([chunk], [[1.0, 0, 0]])

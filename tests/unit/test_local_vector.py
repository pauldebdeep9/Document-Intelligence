"""Unit tests for storage/local_vector.py's settings-fingerprint guard
(P1-04): chunk ids are a function of chunking settings, so mixing chunks
built under different settings into the same store must fail loudly rather
than silently mixing incompatible boundaries.
"""

from __future__ import annotations

import pytest

from isc.common.errors import ChunkSettingsMismatch
from isc.models.acl import AclSet
from isc.models.chunk import Chunk
from isc.storage.local_vector import LocalVectorStore


def _chunk(cid: str, text: str = "some chunk text") -> Chunk:
    return Chunk(id=cid, document_id="doc_1", ordinal=0, text=text,
                 acl=AclSet(allow_terms=frozenset({"everyone:*"})))


def test_first_add_records_the_fingerprint(tmp_path):
    store = LocalVectorStore(tmp_path / "store.pkl")
    store.add([_chunk("c1")], [[0.1, 0.2]], settings_fingerprint="fp_a")
    assert store.settings_fingerprint() == "fp_a"


def test_matching_fingerprint_on_a_later_add_is_fine(tmp_path):
    store = LocalVectorStore(tmp_path / "store.pkl")
    store.add([_chunk("c1")], [[0.1, 0.2]], settings_fingerprint="fp_a")
    store.add([_chunk("c2")], [[0.3, 0.4]], settings_fingerprint="fp_a")
    assert store.count() == 2


def test_mismatched_fingerprint_raises_loudly(tmp_path):
    store = LocalVectorStore(tmp_path / "store.pkl")
    store.add([_chunk("c1")], [[0.1, 0.2]], settings_fingerprint="fp_a")
    with pytest.raises(ChunkSettingsMismatch):
        store.add([_chunk("c2")], [[0.3, 0.4]], settings_fingerprint="fp_b")
    # the failed add must not have partially applied
    assert store.count() == 1


def test_fingerprint_survives_save_and_load(tmp_path):
    path = tmp_path / "store.pkl"
    store = LocalVectorStore(path)
    store.add([_chunk("c1")], [[0.1, 0.2]], settings_fingerprint="fp_a")
    store.save()

    reloaded = LocalVectorStore(path)
    assert reloaded.settings_fingerprint() == "fp_a"
    with pytest.raises(ChunkSettingsMismatch):
        reloaded.add([_chunk("c2")], [[0.3, 0.4]], settings_fingerprint="fp_b")


def test_no_fingerprint_supplied_is_a_noop_guard(tmp_path):
    """Callers that do not pass one (or a store saved before this guard
    existed) are not retroactively broken -- None means unchecked, not
    'matches everything', and is never compared against a real value."""
    store = LocalVectorStore(tmp_path / "store.pkl")
    store.add([_chunk("c1")], [[0.1, 0.2]])
    store.add([_chunk("c2")], [[0.3, 0.4]])
    assert store.count() == 2
    assert store.settings_fingerprint() is None

"""Unit tests for storage/local_vector.py's settings-fingerprint guard
(P1-04): chunk ids are a function of chunking settings, so mixing chunks
built under different settings into the same store must fail loudly rather
than silently mixing incompatible boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from isc.common.errors import ChunkSettingsMismatch
from isc.models.acl import AclSet, Principal
from isc.models.chunk import Chunk
from isc.storage.local_vector import LocalVectorStore

_everyone = Principal(id="u_x")


def _chunk(cid: str, text: str = "some chunk text", document_id: str = "doc_1") -> Chunk:
    return Chunk(id=cid, document_id=document_id, ordinal=0, text=text,
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


# -- upsert-by-id (P1-05 fix: re-running index must not duplicate chunks) --

def test_adding_the_same_chunks_twice_does_not_duplicate(tmp_path):
    store = LocalVectorStore(tmp_path / "store.pkl")
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    first = store.add(chunks, vectors)
    assert (first.inserted, first.replaced) == (3, 0)
    assert store.count() == 3

    second = store.add(chunks, vectors)
    assert (second.inserted, second.replaced) == (0, 3)
    assert store.count() == 3


def test_add_with_an_existing_id_replaces_vector_and_text(tmp_path):
    store = LocalVectorStore(tmp_path / "store.pkl")
    store.add([_chunk("c1", text="original text")], [[0.1, 0.2]])

    result = store.add([_chunk("c1", text="updated text")], [[0.9, 0.1]])

    assert (result.inserted, result.replaced) == (0, 1)
    assert store.count() == 1
    assert store._chunks[0].text == "updated text"
    expected = np.array([0.9, 0.1]) / np.linalg.norm([0.9, 0.1])
    np.testing.assert_allclose(store._vectors[0], expected, atol=1e-6)


def test_remove_document_leaves_no_orphans_after_a_smaller_re_chunk(tmp_path):
    """A document that used to chunk into 3 pieces and now chunks into 1:
    upsert alone would leave the other 2 (no longer produced) ids stranded in
    the store forever. remove_document() is what the pipeline calls first."""
    store = LocalVectorStore(tmp_path / "store.pkl")
    store.add(
        [_chunk("c1", document_id="doc_1"), _chunk("c2", document_id="doc_1"),
         _chunk("c3", document_id="doc_1"), _chunk("other", document_id="doc_2")],
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
    )
    assert store.count() == 4

    removed = store.remove_document("doc_1")
    assert removed == 3
    store.add([_chunk("c1", document_id="doc_1")], [[0.1, 0.2]])

    assert store.count() == 2
    assert {c.id for c in store._chunks} == {"c1", "other"}


def test_bm25_df_is_correct_after_a_replace(tmp_path):
    """The fiddly part: a replaced chunk's old token contribution to df must
    disappear, and its new contribution must appear, without touching the
    other chunk's df contribution at all."""
    store = LocalVectorStore(tmp_path / "store.pkl")
    c1 = _chunk("c1", document_id="doc_1", text="a zzzunique widget assembly")
    c2 = _chunk("c2", document_id="doc_2", text="another widget assembly")
    store.add([c1, c2], [[0.1, 0.2], [0.3, 0.4]])

    assert store._df["zzzunique"] == 1
    assert store._df["widget"] == 2
    idf_widget_before = _idf(store, "widget")

    hits = store.search_lexical("zzzunique", _everyone)
    assert [h.chunk.id for h in hits] == ["c1"]

    c1_replacement = _chunk("c1", document_id="doc_1", text="a replaced widget assembly")
    store.add([c1_replacement], [[0.6, 0.6]])

    # the replaced text's unique term is gone from df and from search...
    assert store._df["zzzunique"] == 0
    assert store.search_lexical("zzzunique", _everyone) == []
    # ...while the term shared with the untouched chunk is exactly as before:
    # same df, same idf, and c2 (never touched) still matches.
    assert store._df["widget"] == 2
    assert _idf(store, "widget") == idf_widget_before
    hits = store.search_lexical("widget", _everyone)
    assert {h.chunk.id for h in hits} == {"c1", "c2"}


def _idf(store: LocalVectorStore, term: str) -> float:
    import math
    n = store.count()
    df = store._df[term]
    return math.log(1 + (n - df + 0.5) / (df + 0.5))

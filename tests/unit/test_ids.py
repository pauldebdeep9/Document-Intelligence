"""Unit tests for common/ids.py's corpus_fingerprint() -- the corpus-content
counterpart to index/chunker.py's settings_fingerprint(). See test_chunker.py
for that one's equivalent tests.
"""

from __future__ import annotations

from isc.common.ids import corpus_fingerprint


def test_corpus_fingerprint_is_deterministic():
    hashes = ["aaa", "bbb", "ccc"]
    assert corpus_fingerprint(hashes) == corpus_fingerprint(list(hashes))


def test_corpus_fingerprint_is_order_independent():
    """SqliteDocStore.content_hashes() makes no ordering promise -- the
    fingerprint must not depend on the order documents happen to come back
    in."""
    assert corpus_fingerprint(["aaa", "bbb", "ccc"]) == corpus_fingerprint(["ccc", "aaa", "bbb"])


def test_corpus_fingerprint_changes_if_any_single_hash_changes():
    """The property the chunk-id check can't have: it has to catch a change
    confined to ONE document, regardless of position -- including the last
    element, standing in for a document like po_019.pdf that no gold
    question's gold_chunk_ids happens to name (see docs/adr/0007)."""
    base = ["aaa", "bbb", "ccc"]
    for i in range(len(base)):
        changed = list(base)
        changed[i] = "zzz"
        assert corpus_fingerprint(changed) != corpus_fingerprint(base), (
            f"changing element {i} did not change the fingerprint"
        )


def test_corpus_fingerprint_changes_with_corpus_size():
    assert corpus_fingerprint(["aaa", "bbb"]) != corpus_fingerprint(["aaa", "bbb", "bbb"])


def test_corpus_fingerprint_of_empty_corpus_is_stable():
    assert corpus_fingerprint([]) == corpus_fingerprint([])

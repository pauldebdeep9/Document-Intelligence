from __future__ import annotations

from isc.models.chunk import ScoredChunk
from isc.retrieve.fusion import rrf
from tests.conftest import make_chunk


def _sc(cid: str, rank: int, dense: float | None = None, lex: float | None = None):
    return ScoredChunk(chunk=make_chunk(cid, f"text {cid}", {"everyone:*"}),
                       score=0.0, dense_score=dense, lexical_score=lex, rank=rank)


def test_agreement_beats_single_list_dominance():
    """A chunk ranked 2nd by both retrievers should beat one ranked 1st by only one."""
    dense = [_sc("a", 0, dense=0.9), _sc("b", 1, dense=0.8)]
    lexical = [_sc("c", 0, lex=5.0), _sc("b", 1, lex=4.0)]
    out = rrf([dense, lexical], k=60, top_n=3)
    assert out[0].chunk.id == "b"


def test_ranks_are_renumbered():
    out = rrf([[_sc("a", 0)], [_sc("b", 0)]], top_n=2)
    assert [s.rank for s in out] == [0, 1]


def test_empty_input_is_safe():
    assert rrf([], top_n=5) == []

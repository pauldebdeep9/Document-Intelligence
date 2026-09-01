import math

import pytest

from isc.models import Chunk, SourceEvidence
from isc.retrieval import cosine_similarity, top_k_chunks


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_matches_known_value() -> None:
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(
        1 / math.sqrt(2)
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [([], [1.0]), ([1.0], [])],
)
def test_cosine_similarity_rejects_empty_vector(
    left: list[float],
    right: list[float],
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        cosine_similarity(left, right)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0, 2.0], [1.0])


def test_cosine_similarity_rejects_zero_left_vector() -> None:
    with pytest.raises(ValueError, match="non-zero magnitude"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_cosine_similarity_rejects_zero_right_vector() -> None:
    with pytest.raises(ValueError, match="non-zero magnitude"):
        cosine_similarity([1.0, 0.0], [0.0, 0.0])


@pytest.mark.parametrize("component", [float("nan"), float("inf"), float("-inf")])
def test_cosine_similarity_rejects_non_finite_component(component: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        cosine_similarity([1.0, component], [1.0, 0.0])


def test_top_k_chunks_ranks_by_descending_cosine_similarity() -> None:
    chunks = [
        Chunk(chunk_id="middle", page_number=1, text="Middle"),
        Chunk(chunk_id="lowest", page_number=1, text="Lowest"),
        Chunk(chunk_id="highest", page_number=1, text="Highest"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[1.0, 1.0], [-1.0, 0.0], [1.0, 0.0]],
        [1.0, 0.0],
    )

    assert [item.chunk_id for item in evidence] == ["highest", "middle", "lowest"]


def test_top_k_chunks_defaults_to_three_results() -> None:
    chunks = [
        Chunk(chunk_id=f"c{index}", page_number=1, text=str(index))
        for index in range(4)
    ]

    evidence = top_k_chunks(
        chunks,
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
        [1.0, 0.0],
    )

    assert [item.chunk_id for item in evidence] == ["c0", "c1", "c2"]


def test_top_k_chunks_returns_corpus_when_k_is_larger() -> None:
    chunks = [
        Chunk(chunk_id="c1", page_number=1, text="One"),
        Chunk(chunk_id="c2", page_number=1, text="Two"),
        Chunk(chunk_id="c3", page_number=1, text="Three"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [1.0, 0.0],
        k=10,
    )

    assert len(evidence) == 3


def test_top_k_chunks_supports_custom_k() -> None:
    chunks = [
        Chunk(chunk_id="low", page_number=1, text="Low"),
        Chunk(chunk_id="high", page_number=1, text="High"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[0.0, 1.0], [1.0, 0.0]],
        [1.0, 0.0],
        k=1,
    )

    assert [item.chunk_id for item in evidence] == ["high"]


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_chunks_rejects_invalid_k(k: int) -> None:
    chunks = [Chunk(chunk_id="c1", page_number=1, text="One")]

    with pytest.raises(ValueError, match="greater than zero"):
        top_k_chunks(chunks, [[1.0]], [1.0], k=k)


def test_top_k_chunks_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="No chunks available"):
        top_k_chunks([], [], [1.0])


def test_top_k_chunks_rejects_chunk_embedding_count_mismatch() -> None:
    chunks = [Chunk(chunk_id="c1", page_number=1, text="One")]

    with pytest.raises(ValueError, match="counts must match"):
        top_k_chunks(chunks, [], [1.0])


def test_top_k_chunks_preserves_input_order_for_score_ties() -> None:
    chunks = [
        Chunk(chunk_id="first", page_number=2, text="First"),
        Chunk(chunk_id="second", page_number=1, text="Second"),
        Chunk(chunk_id="third", page_number=3, text="Third"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[1.0, 1.0], [2.0, 2.0], [0.0, 1.0]],
        [1.0, 0.0],
    )

    assert [item.chunk_id for item in evidence[:2]] == ["first", "second"]


def test_top_k_chunks_copies_exact_evidence_fields_and_score() -> None:
    text = "  Supplier: Müller Components\nCurrency: €\n"
    chunk = Chunk(chunk_id="page-004-chunk-002", page_number=4, text=text)

    evidence = top_k_chunks([chunk], [[1.0, 1.0]], [1.0, 0.0])

    assert isinstance(evidence[0], SourceEvidence)
    assert evidence[0].chunk_id == chunk.chunk_id
    assert evidence[0].page_number == chunk.page_number
    assert evidence[0].text == chunk.text
    assert evidence[0].score == pytest.approx(1 / math.sqrt(2))


def test_top_k_chunks_retains_negative_scores() -> None:
    chunks = [
        Chunk(chunk_id="positive", page_number=1, text="Positive"),
        Chunk(chunk_id="negative", page_number=1, text="Negative"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[1.0, 0.0], [-1.0, 0.0]],
        [1.0, 0.0],
        k=2,
    )

    assert [item.chunk_id for item in evidence] == ["positive", "negative"]
    assert evidence[1].score == pytest.approx(-1.0)


def test_top_k_chunks_returns_all_when_k_equals_corpus_size() -> None:
    chunks = [
        Chunk(chunk_id="c1", page_number=1, text="One"),
        Chunk(chunk_id="c2", page_number=1, text="Two"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[0.0, 1.0], [1.0, 0.0]],
        [1.0, 0.0],
        k=2,
    )

    assert [item.chunk_id for item in evidence] == ["c2", "c1"]


def test_top_k_chunks_ranks_across_pages_without_grouping() -> None:
    chunks = [
        Chunk(chunk_id="page-ten", page_number=10, text="Lower score"),
        Chunk(chunk_id="page-one", page_number=1, text="Highest score"),
        Chunk(chunk_id="page-five", page_number=5, text="Middle score"),
    ]

    evidence = top_k_chunks(
        chunks,
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [1.0, 0.0],
    )

    assert [(item.chunk_id, item.page_number) for item in evidence] == [
        ("page-one", 1),
        ("page-five", 5),
        ("page-ten", 10),
    ]


def test_top_k_chunks_is_deterministic() -> None:
    chunks = [
        Chunk(chunk_id="c1", page_number=1, text="One"),
        Chunk(chunk_id="c2", page_number=2, text="Two"),
    ]
    embeddings = [[1.0, 1.0], [1.0, 0.0]]
    query = [1.0, 0.0]

    assert top_k_chunks(chunks, embeddings, query) == top_k_chunks(
        chunks,
        embeddings,
        query,
    )

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.corpus.generate import PDF_DIR
from evals.gold.authoring import build_goldset, write_goldset
from evals.gold.schema import (
    Anchor,
    ChunkingConfig,
    ExtractionGold,
    GoldItem,
    GoldSet,
    QuestionClass,
    RetrievalGold,
    validate_anchors,
)
from evals.gold.split import Split, assert_splits_are_disjoint, assign_split, load_gold
from isc.models import PurchaseOrder

_EXPECTED_DEV = {"po-003", "po-005", "po-006", "po-007", "po-011", "po-012"}
_EXPECTED_TEST = {"po-002", "po-008", "po-010"}
_EXPECTED_HELD = {"po-001", "po-004", "po-009"}

_NON_ABSENT_CLASSES: list[QuestionClass] = [
    "header_field",
    "line_item",
    "cross_page",
    "near_duplicate",
]


def _make_retrieval(
    *,
    question_class: QuestionClass,
    anchors: list[Anchor],
    question_id: str = "q1",
    doc_id: str = "po-001",
) -> RetrievalGold:
    return RetrievalGold(
        question_id=question_id,
        doc_id=doc_id,
        question="irrelevant",
        question_class=question_class,
        anchors=anchors,
    )


def _make_gold_item(doc_id: str, retrieval: list[RetrievalGold]) -> GoldItem:
    return GoldItem(
        doc_id=doc_id,
        extraction=ExtractionGold(doc_id=doc_id, expected=PurchaseOrder(po_number="X")),
        retrieval=retrieval,
    )


def _make_goldset(items: list[GoldItem]) -> GoldSet:
    return GoldSet(
        version="test",
        created_utc="2026-01-01T00:00:00+00:00",
        items=items,
        chunking=ChunkingConfig(chunk_size=1200, overlap=200),
    )


# --- schema model contracts ----------------------------------------------------------------


def test_anchor_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        Anchor(page_number=1, text="   ")


@pytest.mark.parametrize("question_class", _NON_ABSENT_CLASSES)
def test_retrieval_gold_rejects_empty_anchors_for_non_absent_classes(
    question_class: QuestionClass,
) -> None:
    with pytest.raises(ValueError, match="anchors may only be empty"):
        _make_retrieval(question_class=question_class, anchors=[])


def test_retrieval_gold_allows_empty_anchors_when_absent() -> None:
    retrieval = _make_retrieval(question_class="absent", anchors=[])
    assert retrieval.anchors == []


def test_retrieval_gold_allows_non_empty_anchors_when_absent() -> None:
    retrieval = _make_retrieval(
        question_class="absent",
        anchors=[Anchor(page_number=1, text="some evidence")],
    )
    assert len(retrieval.anchors) == 1


def test_gold_item_rejects_mismatched_extraction_doc_id() -> None:
    with pytest.raises(ValueError, match="extraction.doc_id"):
        GoldItem(
            doc_id="po-001",
            extraction=ExtractionGold(doc_id="po-002", expected=PurchaseOrder()),
            retrieval=[],
        )


def test_gold_item_rejects_mismatched_retrieval_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id mismatch"):
        _make_gold_item(
            "po-001",
            [
                _make_retrieval(
                    question_class="absent",
                    anchors=[],
                    doc_id="po-002",
                ),
            ],
        )


def test_goldset_rejects_duplicate_doc_ids() -> None:
    item = _make_gold_item("po-001", [])
    with pytest.raises(ValueError, match="duplicate doc_ids"):
        _make_goldset([item, item.model_copy()])


def test_goldset_rejects_duplicate_question_ids() -> None:
    item_a = _make_gold_item(
        "po-001", [_make_retrieval(question_class="absent", anchors=[], doc_id="po-001")]
    )
    item_b = _make_gold_item(
        "po-002", [_make_retrieval(question_class="absent", anchors=[], doc_id="po-002")]
    )
    with pytest.raises(ValueError, match="duplicate question_ids"):
        _make_goldset([item_a, item_b])


# --- validate_anchors ------------------------------------------------------------------------


def test_validate_anchors_passes_for_the_full_authored_goldset_at_1200_200() -> None:
    goldset = build_goldset()
    validate_anchors(goldset, chunk_size=1200, overlap=200)


def test_validate_anchors_passes_for_the_full_authored_goldset_at_600_100() -> None:
    goldset = build_goldset()
    validate_anchors(goldset, chunk_size=600, overlap=100)


def test_validate_anchors_rejects_anchor_text_not_present_on_the_page() -> None:
    item = _make_gold_item(
        "po-001",
        [
            _make_retrieval(
                question_id="bad-q",
                doc_id="po-001",
                question_class="header_field",
                anchors=[Anchor(page_number=1, text="this text does not appear anywhere")],
            ),
        ],
    )
    goldset = _make_goldset([item])

    with pytest.raises(ValueError, match="bad-q"):
        validate_anchors(goldset, chunk_size=1200, overlap=200, pdf_dir=PDF_DIR)


def test_validate_anchors_rejects_anchor_on_the_wrong_page_number() -> None:
    # "Total Amount: 2960.00" is real po-006 text, but it's on page 3, not page 1.
    item = _make_gold_item(
        "po-006",
        [
            _make_retrieval(
                question_id="wrong-page-q",
                doc_id="po-006",
                question_class="header_field",
                anchors=[Anchor(page_number=1, text="Total Amount: 2960.00")],
            ),
        ],
    )
    goldset = _make_goldset([item])

    with pytest.raises(ValueError, match="wrong-page-q"):
        validate_anchors(goldset, chunk_size=1200, overlap=200, pdf_dir=PDF_DIR)


# --- split.assign_split ----------------------------------------------------------------------


def test_assign_split_is_deterministic_within_a_process() -> None:
    assert assign_split("po-001") == assign_split("po-001")


def test_assign_split_matches_expected_distribution_for_current_corpus() -> None:
    doc_ids = [f"po-{n:03d}" for n in range(1, 13)]
    dev = {d for d in doc_ids if assign_split(d) == Split.DEV}
    test = {d for d in doc_ids if assign_split(d) == Split.TEST}
    held = {d for d in doc_ids if assign_split(d) == Split.HELD}

    assert dev == _EXPECTED_DEV
    assert test == _EXPECTED_TEST
    assert held == _EXPECTED_HELD


def test_assign_split_is_stable_across_processes() -> None:
    script = (
        "from evals.gold.split import assign_split; "
        "print(assign_split('po-006').value)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.stdout.strip() == assign_split("po-006").value


# --- split.assert_splits_are_disjoint -------------------------------------------------------


def test_assert_splits_are_disjoint_passes_for_all_current_doc_ids() -> None:
    doc_ids = [f"po-{n:03d}" for n in range(1, 13)]
    assert_splits_are_disjoint(doc_ids)


def test_assert_splits_are_disjoint_passes_for_an_empty_list() -> None:
    assert_splits_are_disjoint([])


def test_assert_splits_are_disjoint_raises_when_a_doc_id_maps_inconsistently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.gold.split as split_module

    results = iter([Split.DEV, Split.TEST])
    monkeypatch.setattr(split_module, "assign_split", lambda doc_id: next(results))

    with pytest.raises(ValueError, match="po-001"):
        split_module.assert_splits_are_disjoint(["po-001", "po-001"])


# --- split.load_gold --------------------------------------------------------------------------


@pytest.fixture
def small_goldset_path(tmp_path: Path) -> Path:
    items = [
        _make_gold_item(
            doc_id,
            [
                _make_retrieval(
                    question_class="absent",
                    anchors=[],
                    question_id=f"{doc_id}-q1",
                    doc_id=doc_id,
                ),
            ],
        )
        for doc_id in _EXPECTED_DEV | _EXPECTED_TEST | _EXPECTED_HELD
    ]
    goldset_path = tmp_path / "goldset.json"
    write_goldset(_make_goldset(items), goldset_path)
    return goldset_path


def test_load_gold_dev_works_with_no_extra_arguments(small_goldset_path: Path) -> None:
    goldset = load_gold(Split.DEV, goldset_path=small_goldset_path)

    assert {item.doc_id for item in goldset.items} == _EXPECTED_DEV


def test_load_gold_test_raises_without_allow_test(small_goldset_path: Path) -> None:
    with pytest.raises(ValueError, match="allow_test"):
        load_gold(Split.TEST, goldset_path=small_goldset_path)


def test_load_gold_test_succeeds_with_allow_test(small_goldset_path: Path) -> None:
    goldset = load_gold(Split.TEST, allow_test=True, goldset_path=small_goldset_path)

    assert {item.doc_id for item in goldset.items} == _EXPECTED_TEST


@pytest.mark.parametrize(
    ("allow_held", "run_label"),
    [(False, ""), (False, "run-1"), (True, "")],
)
def test_load_gold_held_raises_without_both_allow_held_and_run_label(
    allow_held: bool,
    run_label: str,
    small_goldset_path: Path,
) -> None:
    with pytest.raises(ValueError, match="allow_held"):
        load_gold(
            Split.HELD,
            allow_held=allow_held,
            run_label=run_label,
            goldset_path=small_goldset_path,
        )


def test_load_gold_held_succeeds_with_allow_held_and_run_label(
    small_goldset_path: Path,
) -> None:
    goldset = load_gold(
        Split.HELD,
        allow_held=True,
        run_label="run-1",
        goldset_path=small_goldset_path,
    )

    assert {item.doc_id for item in goldset.items} == _EXPECTED_HELD


# --- authoring.build_goldset -------------------------------------------------------------------


def test_build_goldset_has_all_twelve_doc_ids() -> None:
    goldset = build_goldset()

    assert {item.doc_id for item in goldset.items} == {f"po-{n:03d}" for n in range(1, 13)}


def test_build_goldset_expected_purchase_order_matches_corpus_ground_truth() -> None:
    goldset = build_goldset()

    po_004 = next(item for item in goldset.items if item.doc_id == "po-004")
    assert po_004.extraction.expected.line_items[0].part_number == "4500123456"
    assert po_004.extraction.expected.supplier_name == "Titan Fabrication"


def test_build_goldset_covers_every_question_class() -> None:
    goldset = build_goldset()

    classes = {
        retrieval.question_class for item in goldset.items for retrieval in item.retrieval
    }
    assert classes == {"header_field", "line_item", "cross_page", "absent", "near_duplicate"}


def test_build_goldset_question_count_is_within_the_target_range() -> None:
    goldset = build_goldset()

    total = sum(len(item.retrieval) for item in goldset.items)
    assert 30 <= total <= 40


# --- authoring.write_goldset -------------------------------------------------------------------


def test_write_goldset_creates_a_readable_file(tmp_path: Path) -> None:
    goldset = build_goldset()
    output_path = tmp_path / "out.json"

    write_goldset(goldset, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["version"] == goldset.version


def test_write_goldset_round_trips_through_goldset_model(tmp_path: Path) -> None:
    goldset = build_goldset()
    output_path = tmp_path / "out.json"

    write_goldset(goldset, output_path)
    round_tripped = GoldSet.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert round_tripped == goldset


def test_write_goldset_creates_parent_directories(tmp_path: Path) -> None:
    goldset = build_goldset()
    output_path = tmp_path / "nested" / "dir" / "out.json"

    write_goldset(goldset, output_path)

    assert output_path.exists()


# --- authoring.main ----------------------------------------------------------------------------


def test_main_writes_goldset_to_the_patched_default_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.gold.authoring as authoring_module

    monkeypatch.setattr(authoring_module, "GOLDSET_PATH", tmp_path / "goldset.json")

    authoring_module.main()

    assert (tmp_path / "goldset.json").exists()


def test_main_prints_the_item_and_question_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import evals.gold.authoring as authoring_module

    monkeypatch.setattr(authoring_module, "GOLDSET_PATH", tmp_path / "goldset.json")

    authoring_module.main()

    captured = capsys.readouterr()
    assert "12 items" in captured.out


def test_main_is_rerunnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import evals.gold.authoring as authoring_module

    monkeypatch.setattr(authoring_module, "GOLDSET_PATH", tmp_path / "goldset.json")

    authoring_module.main()
    authoring_module.main()

    written = GoldSet.model_validate_json((tmp_path / "goldset.json").read_text())
    assert len(written.items) == 12

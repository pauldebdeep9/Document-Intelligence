import hashlib
import json
from pathlib import Path

import pytest

from evals.corpus.generate import generate_corpus, render_pdf, write_corpus_json
from evals.corpus.specs import DOCUMENT_SPECS, DocumentSpec
from isc.chunking import chunk_pages
from isc.models import PurchaseOrder
from isc.pdf import extract_pdf_pages

_EXPECTED_DOC_IDS = {f"po-{n:03d}" for n in range(1, 13)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- DOCUMENT_SPECS / DocumentSpec -------------------------------------------------------


def test_document_specs_has_exactly_the_expected_twelve_doc_ids() -> None:
    doc_ids = {spec.doc_id for spec in DOCUMENT_SPECS}

    assert len(DOCUMENT_SPECS) == 12
    assert doc_ids == _EXPECTED_DOC_IDS


def test_document_spec_doc_ids_are_unique() -> None:
    doc_ids = [spec.doc_id for spec in DOCUMENT_SPECS]

    assert len(doc_ids) == len(set(doc_ids))


def test_document_spec_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="extra"):
        DocumentSpec(
            doc_id="po-999",
            purchase_order=PurchaseOrder(),
            pages=[],
            notes="",
            unexpected_field="not allowed",  # type: ignore[call-arg]
        )


# --- render_pdf ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", DOCUMENT_SPECS, ids=[spec.doc_id for spec in DOCUMENT_SPECS])
def test_render_pdf_output_is_readable_and_matches_declared_page_count(
    spec: DocumentSpec,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / f"{spec.doc_id}.pdf"

    render_pdf(spec, output_path)
    pages = extract_pdf_pages(output_path)

    assert len(pages) == len(spec.pages)
    assert [page.page_number for page in pages] == list(range(1, len(spec.pages) + 1))


def test_render_pdf_po007_blank_page_survives_extraction_as_blank(tmp_path: Path) -> None:
    spec = next(spec for spec in DOCUMENT_SPECS if spec.doc_id == "po-007")
    output_path = tmp_path / "po-007.pdf"

    render_pdf(spec, output_path)
    pages = extract_pdf_pages(output_path)

    assert pages[1].page_number == 2
    assert pages[1].text.strip() == ""


def test_render_pdf_is_byte_identical_across_two_calls(tmp_path: Path) -> None:
    spec = next(spec for spec in DOCUMENT_SPECS if spec.doc_id == "po-001")
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"

    render_pdf(spec, first_path)
    render_pdf(spec, second_path)

    assert _sha256(first_path) == _sha256(second_path)


# --- write_corpus_json ---------------------------------------------------------------------


def test_write_corpus_json_contains_all_twelve_doc_ids(tmp_path: Path) -> None:
    output_path = tmp_path / "corpus.json"

    write_corpus_json(DOCUMENT_SPECS, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert {entry["doc_id"] for entry in payload} == _EXPECTED_DOC_IDS


def test_write_corpus_json_round_trips_through_document_spec(tmp_path: Path) -> None:
    output_path = tmp_path / "corpus.json"

    write_corpus_json(DOCUMENT_SPECS, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    round_tripped = [DocumentSpec.model_validate(entry) for entry in payload]

    assert round_tripped == DOCUMENT_SPECS


def test_write_corpus_json_is_byte_identical_across_two_calls(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_corpus_json(DOCUMENT_SPECS, first_path)
    write_corpus_json(DOCUMENT_SPECS, second_path)

    assert _sha256(first_path) == _sha256(second_path)


# --- generate_corpus -----------------------------------------------------------------------


def test_generate_corpus_writes_twelve_pdfs_and_corpus_json(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    corpus_path = tmp_path / "corpus.json"

    generate_corpus(pdf_dir=pdf_dir, corpus_path=corpus_path)

    assert sorted(p.name for p in pdf_dir.glob("*.pdf")) == sorted(
        f"{doc_id}.pdf" for doc_id in _EXPECTED_DOC_IDS
    )
    assert corpus_path.exists()


def test_generate_corpus_output_is_byte_identical_across_two_runs(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    generate_corpus(pdf_dir=first_dir / "pdfs", corpus_path=first_dir / "corpus.json")
    generate_corpus(pdf_dir=second_dir / "pdfs", corpus_path=second_dir / "corpus.json")

    for doc_id in _EXPECTED_DOC_IDS:
        first_hash = _sha256(first_dir / "pdfs" / f"{doc_id}.pdf")
        second_hash = _sha256(second_dir / "pdfs" / f"{doc_id}.pdf")
        assert first_hash == second_hash, f"{doc_id}.pdf differs across runs"

    assert _sha256(first_dir / "corpus.json") == _sha256(second_dir / "corpus.json")


def test_generate_corpus_po010_description_crosses_a_chunk_boundary(tmp_path: Path) -> None:
    # po-010's description is built from short, individually unique "titanium-bracket-
    # segment-NNNN" tokens (see evals/corpus/specs.py) specifically so that a short, atomic
    # token can be used as a boundary marker here. A wrapped multi-line PDF render turns the
    # single space between two tokens into a newline when pypdf extracts it back, so an
    # arbitrary-length slice of the original description (e.g. description[:50]) can straddle
    # that substitution and silently fail to match anything — a short token never can, since
    # word-wrap only ever breaks *between* tokens, not inside one.
    spec = next(spec for spec in DOCUMENT_SPECS if spec.doc_id == "po-010")
    description = spec.purchase_order.line_items[0].description
    assert description is not None
    first_token = "titanium-bracket-segment-0001"
    last_token = "titanium-bracket-segment-0050"
    assert description.startswith(first_token)
    assert description.rstrip().endswith(last_token)

    pdf_dir = tmp_path / "pdfs"
    generate_corpus(pdf_dir=pdf_dir, corpus_path=tmp_path / "corpus.json")

    pages = extract_pdf_pages(pdf_dir / "po-010.pdf")
    chunks = chunk_pages(pages)

    assert not any(description in chunk.text for chunk in chunks)
    first_chunk_ids = {chunk.chunk_id for chunk in chunks if first_token in chunk.text}
    last_chunk_ids = {chunk.chunk_id for chunk in chunks if last_token in chunk.text}
    assert first_chunk_ids
    assert last_chunk_ids
    assert first_chunk_ids != last_chunk_ids


# --- main ------------------------------------------------------------------------------------


def test_main_creates_all_expected_files_at_patched_default_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.corpus.generate as generate_module

    monkeypatch.setattr(generate_module, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(generate_module, "CORPUS_PATH", tmp_path / "corpus.json")

    generate_module.main()

    assert sorted(p.name for p in (tmp_path / "pdfs").glob("*.pdf")) == sorted(
        f"{doc_id}.pdf" for doc_id in _EXPECTED_DOC_IDS
    )
    assert (tmp_path / "corpus.json").exists()


def test_main_prints_a_summary_message_with_the_spec_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import evals.corpus.generate as generate_module

    monkeypatch.setattr(generate_module, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(generate_module, "CORPUS_PATH", tmp_path / "corpus.json")

    generate_module.main()

    captured = capsys.readouterr()
    assert "12" in captured.out


def test_main_is_rerunnable_producing_identical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.corpus.generate as generate_module

    monkeypatch.setattr(generate_module, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(generate_module, "CORPUS_PATH", tmp_path / "corpus.json")

    generate_module.main()
    first_hashes = {p.name: _sha256(p) for p in (tmp_path / "pdfs").glob("*.pdf")}
    first_corpus_hash = _sha256(tmp_path / "corpus.json")

    generate_module.main()
    second_hashes = {p.name: _sha256(p) for p in (tmp_path / "pdfs").glob("*.pdf")}
    second_corpus_hash = _sha256(tmp_path / "corpus.json")

    assert first_hashes == second_hashes
    assert first_corpus_hash == second_corpus_hash

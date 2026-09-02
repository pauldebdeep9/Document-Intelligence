from evals.gold.schema import (
    Anchor,
    ChunkingConfig,
    ExtractionGold,
    GoldItem,
    GoldSet,
    RetrievalGold,
)
from evals.gold.split import Split
from evals.report import build_report
from evals.runner import DocumentRecord, ItemRecord, RunConfig, RunRecord
from isc.models import PurchaseOrder

_INSUFFICIENT_ANSWER = "I don't have enough information in the provided sources."


def _goldset() -> GoldSet:
    return GoldSet(
        version="test",
        chunking=ChunkingConfig(chunk_size=1200, overlap=200),
        items=[
            GoldItem(
                doc_id="doc-a",
                extraction=ExtractionGold(
                    doc_id="doc-a",
                    expected=PurchaseOrder(
                        po_number="PO-A001", supplier_name="Alpha Corp", currency="USD"
                    ),
                ),
                retrieval=[
                    RetrievalGold(
                        question_id="doc-a-q1",
                        doc_id="doc-a",
                        question="What is the PO number?",
                        question_class="header_field",
                        anchors=[Anchor(page_number=1, text="PO Number: PO-A001")],
                    ),
                    RetrievalGold(
                        question_id="doc-a-q2",
                        doc_id="doc-a",
                        question="Who is the supplier?",
                        question_class="header_field",
                        anchors=[Anchor(page_number=1, text="Supplier: Alpha Corp")],
                    ),
                    RetrievalGold(
                        question_id="doc-a-q3",
                        doc_id="doc-a",
                        question="What are the freight terms?",
                        question_class="absent",
                        anchors=[],
                    ),
                ],
            ),
            GoldItem(
                doc_id="doc-b",
                extraction=ExtractionGold(
                    doc_id="doc-b", expected=PurchaseOrder(po_number="PO-B001")
                ),
                retrieval=[
                    RetrievalGold(
                        question_id="doc-b-q1",
                        doc_id="doc-b",
                        question="What is part number X's price?",
                        question_class="line_item",
                        anchors=[Anchor(page_number=1, text="Unit Price: 9.00")],
                    ),
                    RetrievalGold(
                        question_id="doc-b-q2",
                        doc_id="doc-b",
                        question="What is the warranty period?",
                        question_class="absent",
                        anchors=[],
                    ),
                    RetrievalGold(
                        question_id="doc-b-q3",
                        doc_id="doc-b",
                        question="What is the delivery date?",
                        question_class="absent",
                        anchors=[],
                    ),
                ],
            ),
        ],
    )


def _run_record() -> RunRecord:
    return RunRecord(
        run_id="report-test-run",
        created_utc="2026-01-01T00:00:00+00:00",
        git_commit_sha="deadbeef",
        config=RunConfig(
            chat_model="chat-model",
            embedding_model="embed-model",
            chunk_size=1200,
            overlap=200,
            k=3,
            goldset_version="test",
            split=Split.DEV,
            document_count=2,
            pooled_chunk_count=4,
        ),
        documents=[
            DocumentRecord(
                doc_id="doc-a",
                # po_number correct, supplier_name wrong, currency missed, ship_to_site
                # hallucinated.
                purchase_order=PurchaseOrder(
                    po_number="PO-A001",
                    supplier_name="Alpha Corporation",
                    currency=None,
                    ship_to_site="Unexpected Site",
                ),
                chunk_count=2,
            ),
            DocumentRecord(
                doc_id="doc-b",
                purchase_order=PurchaseOrder(po_number="PO-B001"),
                chunk_count=2,
            ),
        ],
        items=[
            # header_field, hit at rank 1.
            ItemRecord(
                question_id="doc-a-q1",
                doc_id="doc-a",
                question_class="header_field",
                retrieved_chunk_ids=["doc-a:c1"],
                retrieved_doc_ids=["doc-a"],
                retrieved_page_numbers=[1],
                retrieved_texts=["PO Number: PO-A001"],
                retrieved_scores=[0.9],
                answer="PO-A001",
                source_chunk_ids=["doc-a:c1"],
                error=None,
            ),
            # header_field, hit at rank 2 (wrong chunk first).
            ItemRecord(
                question_id="doc-a-q2",
                doc_id="doc-a",
                question_class="header_field",
                retrieved_chunk_ids=["doc-a:c2", "doc-a:c3"],
                retrieved_doc_ids=["doc-a", "doc-a"],
                retrieved_page_numbers=[1, 1],
                retrieved_texts=["Total Amount: 1.00", "Supplier: Alpha Corp"],
                retrieved_scores=[0.8, 0.7],
                answer="Alpha Corp",
                source_chunk_ids=["doc-a:c3"],
                error=None,
            ),
            # absent, compliant.
            ItemRecord(
                question_id="doc-a-q3",
                doc_id="doc-a",
                question_class="absent",
                retrieved_chunk_ids=[],
                retrieved_doc_ids=[],
                retrieved_page_numbers=[],
                retrieved_texts=[],
                retrieved_scores=[],
                answer=_INSUFFICIENT_ANSWER,
                source_chunk_ids=[],
                error=None,
            ),
            # line_item, total miss.
            ItemRecord(
                question_id="doc-b-q1",
                doc_id="doc-b",
                question_class="line_item",
                retrieved_chunk_ids=["doc-a:c1"],
                retrieved_doc_ids=["doc-a"],
                retrieved_page_numbers=[1],
                retrieved_texts=["PO Number: PO-A001"],
                retrieved_scores=[0.5],
                answer="I don't know",
                source_chunk_ids=[],
                error=None,
            ),
            # absent, NON-compliant (answered instead of declining).
            ItemRecord(
                question_id="doc-b-q2",
                doc_id="doc-b",
                question_class="absent",
                retrieved_chunk_ids=[],
                retrieved_doc_ids=[],
                retrieved_page_numbers=[],
                retrieved_texts=[],
                retrieved_scores=[],
                answer="12 months",
                source_chunk_ids=[],
                error=None,
            ),
            # absent, but errored -> must not appear in compliance counts.
            ItemRecord(
                question_id="doc-b-q3",
                doc_id="doc-b",
                question_class="absent",
                retrieved_chunk_ids=[],
                retrieved_doc_ids=[],
                retrieved_page_numbers=[],
                retrieved_texts=[],
                retrieved_scores=[],
                answer=None,
                source_chunk_ids=[],
                error="RuntimeError: provider unavailable",
            ),
        ],
    )


def test_build_report_excludes_absent_questions_from_retrieval_slices_and_states_count() -> None:
    report = build_report(_run_record(), _goldset())

    # 2 successfully-scored absent items (doc-a-q3, doc-b-q2) — the errored one
    # (doc-b-q3) is excluded from every slice, including this count, per its own rule.
    assert "2 absent questions excluded from retrieval slices" in report
    assert "absent: hit@k" not in report


def test_build_report_contains_no_percent_character() -> None:
    report = build_report(_run_record(), _goldset())

    assert "%" not in report


def test_build_report_computes_hit_at_k_and_mrr_per_question_class() -> None:
    report = build_report(_run_record(), _goldset())

    assert "header_field: hit@k 2/2, MRR 0.750" in report
    assert "line_item: hit@k 0/1, MRR 0.000" in report


def test_build_report_computes_field_verdict_counts_per_field() -> None:
    report = build_report(_run_record(), _goldset())

    assert "po_number: correct 2/2" in report
    assert "supplier_name: correct 1/2, wrong 1/2" in report
    assert "currency: correct 1/2" in report and "missed 1/2" in report
    assert "ship_to_site: correct 1/2" in report and "hallucinated 1/2" in report


def test_build_report_reports_insufficiency_compliance_for_absent_slice() -> None:
    report = build_report(_run_record(), _goldset())

    assert "compliant: 1/2" in report
    assert "doc-b-q2" in report
    assert "12 months" in report


def test_build_report_lists_errored_items_explicitly() -> None:
    report = build_report(_run_record(), _goldset())

    assert "1 item(s) errored" in report
    assert "doc-b-q3" in report
    assert "RuntimeError: provider unavailable" in report


def test_build_report_excludes_errored_items_from_metric_slices() -> None:
    report = build_report(_run_record(), _goldset())

    # doc-b-q3 is an absent item that errored; it must not be counted as compliant or
    # non-compliant, only as an error. 2/2 (not 2/3) proves this.
    assert "compliant: 1/2" in report


def test_build_report_returns_a_string() -> None:
    report = build_report(_run_record(), _goldset())

    assert isinstance(report, str)
    assert len(report) > 0

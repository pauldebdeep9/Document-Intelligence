"""Minimal terminal runner for the Document Intelligence proof of concept."""

import argparse
import os
from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from isc.models import PipelineResult, PurchaseOrder, SourceEvidence
from isc.pipeline import process_document


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse optional non-interactive CLI arguments for the demo runner."""
    parser = argparse.ArgumentParser(
        description="Run the Document Intelligence PoC pipeline.",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        help="Path to the PDF to process. Prompted for interactively if omitted.",
    )
    parser.add_argument(
        "--question",
        dest="question",
        help="Question to answer from the PDF. Prompted for interactively if omitted.",
    )
    return parser.parse_args(argv)


def _require_env(name: str) -> str:
    """Return a required environment value or raise a clear configuration error."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Required environment variable is missing or blank: {name}")
    return value


def _print_purchase_order(purchase_order: PurchaseOrder) -> None:
    """Display the structured Purchase Order in a compact terminal format."""
    print("\nPurchase Order")
    print("--------------")
    fields = (
        ("PO Number", purchase_order.po_number),
        ("PO Date", purchase_order.po_date),
        ("Supplier", purchase_order.supplier_name),
        ("Ship-to Site", purchase_order.ship_to_site),
        ("Payment Terms", purchase_order.payment_terms),
        ("Currency", purchase_order.currency),
        ("Total Amount", purchase_order.total_amount),
    )
    for label, value in fields:
        print(f"{label}: {value if value is not None else '-'}")

    print("Line Items:")
    if not purchase_order.line_items:
        print("  None")
    for number, item in enumerate(purchase_order.line_items, start=1):
        print(f"  {number}.")
        print(f"     Part Number: {item.part_number or '-'}")
        print(f"     Description: {item.description or '-'}")
        print(f"     Quantity: {item.quantity if item.quantity is not None else '-'}")
        print(f"     Unit Price: {item.unit_price if item.unit_price is not None else '-'}")


def _print_sources(sources: list[SourceEvidence]) -> None:
    """Display application-owned evidence returned by the pipeline."""
    print("\nSources")
    print("-------")
    if not sources:
        print("No supporting sources returned.")
        return

    for source in sources:
        print(f"Chunk: {source.chunk_id}")
        print(f"Page: {source.page_number}")
        print(f"Cosine retrieval similarity: {source.score:.6f}")
        print("Text:")
        print(source.text)


def _print_result(result: PipelineResult) -> None:
    """Display all human-facing pipeline output."""
    _print_purchase_order(result.purchase_order)
    print("\nAnswer")
    print("------")
    print(result.answer)
    _print_sources(result.sources)


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration, run the PoC pipeline, and display its result."""
    load_dotenv()
    args = _parse_args(argv)
    try:
        api_key = _require_env("OPENAI_API_KEY")
        chat_model = _require_env("OPENAI_CHAT_MODEL")
        embedding_model = _require_env("OPENAI_EMBEDDING_MODEL")

        pdf_path = (args.pdf_path if args.pdf_path is not None else input("PDF path: ")).strip()
        if not pdf_path:
            raise ValueError("PDF path must not be blank")
        question = args.question if args.question is not None else input("Question: ")
        if not question.strip():
            raise ValueError("Question must not be blank")

        client = OpenAI(api_key=api_key)
        result = process_document(
            pdf_path=pdf_path,
            question=question,
            client=client,
            chat_model=chat_model,
            embedding_model=embedding_model,
        )
    except (FileNotFoundError, ValueError, OpenAIError) as exc:
        print(f"\nError: {exc}")
        return

    _print_result(result)


if __name__ == "__main__":
    main()

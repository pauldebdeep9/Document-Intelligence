"""Builds data/gold/goldset.json from corpus.json plus hand-written retrieval questions.

Every anchor below was copied from the real output of `extract_pdf_pages` run against the
actual generated PDFs, not from corpus.json's `pages` field — those can differ (e.g. a
wrapped line joins with "\\n" at the wrap point, not the original space; trailing whitespace
at end-of-line does not survive layout-mode extraction). `validate_anchors` re-checks all of
this against the real PDFs regardless; these are simply pre-checked so that check is expected
to pass, not a substitute for it.
"""

import json
from pathlib import Path

from evals.corpus.generate import CORPUS_PATH
from evals.gold.schema import (
    Anchor,
    ChunkingConfig,
    ExtractionGold,
    GoldItem,
    GoldSet,
    RetrievalGold,
    validate_anchors,
)
from isc.models import PurchaseOrder

GOLDSET_PATH = Path("data/gold/goldset.json")

CHUNK_SIZE = 1200
OVERLAP = 200

_VERSION = "1.1.0"

# doc_id -> hand-written retrieval questions for that document.
_QUESTIONS: dict[str, list[RetrievalGold]] = {
    "po-001": [
        RetrievalGold(
            question_id="po-001-q1",
            doc_id="po-001",
            question="What is the PO number on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="PO Number: PO-1001")],
        ),
        RetrievalGold(
            question_id="po-001-q2",
            doc_id="po-001",
            question="What are the payment terms on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Payment Terms: Net 30")],
        ),
        RetrievalGold(
            question_id="po-001-q3",
            doc_id="po-001",
            question="What is the unit price of part number GSK-1140?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "2. Part Number: GSK-1140\n"
                        "   Description: Silicone gasket, 140mm diameter\n"
                        "   Quantity: 5\n"
                        "   Unit Price: 97.00"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-001-q4",
            doc_id="po-001",
            question="What is the quantity ordered for part number SHF-3390?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "3. Part Number: SHF-3390\n"
                        "   Description: Stainless steel shaft, 390mm length\n"
                        "   Quantity: 100"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-001-q5",
            doc_id="po-001",
            question="What are the freight terms (e.g., FOB, CIF) for this purchase order?",
            question_class="absent",
            anchors=[],
        ),
    ],
    "po-002": [
        RetrievalGold(
            question_id="po-002-q1",
            doc_id="po-002",
            question="What is the ship-to site on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Ship-to Site: Dockside Warehouse 2")],
        ),
        RetrievalGold(
            question_id="po-002-q2",
            doc_id="po-002",
            question="What is the quantity ordered for part number WSR-0820?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "2. Part Number: WSR-0820\n"
                        "   Description: Flat washer, M8\n"
                        "   Quantity: 800\n"
                        "   Unit Price: 0.20"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-002-q3",
            doc_id="po-002",
            question="What are the payment terms on this purchase order?",
            question_class="absent",
            anchors=[],
        ),
        RetrievalGold(
            question_id="po-002-q4",
            doc_id="po-002",
            question="What is the currency for this purchase order?",
            question_class="absent",
            anchors=[],
        ),
        RetrievalGold(
            question_id="po-002-q5",
            doc_id="po-002",
            question="What is the unit price of part number HXN-0812?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "1. Part Number: HXN-0812\n"
                        "   Description: Hex nut, M8\n"
                        "   Quantity: 400\n"
                        "   Unit Price: 1.20"
                    ),
                ),
            ],
        ),
    ],
    "po-003": [
        RetrievalGold(
            question_id="po-003-q1",
            doc_id="po-003",
            question="What is the PO date on this purchase order, exactly as printed?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="PO Date: 03/04/2026")],
        ),
        RetrievalGold(
            question_id="po-003-q2",
            doc_id="po-003",
            question="Who is the supplier on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Supplier: Coastal Hydraulics")],
        ),
        RetrievalGold(
            question_id="po-003-q3",
            doc_id="po-003",
            question="What is the unit price of the hydraulic hose fitting?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "1. Part Number: HYD-4471\n"
                        "   Description: Hydraulic hose fitting, 1/2in NPT\n"
                        "   Quantity: 25\n"
                        "   Unit Price: 8.50"
                    ),
                ),
            ],
        ),
    ],
    # po-004-q1/po-005-q1 originally shared byte-identical question text ("...on this purchase
    # order?"), which made the pair undecidable from the query alone: nothing in the text said
    # which document was meant, so a hit or miss was luck, not discrimination (measured at
    # k=3, the two candidates scored 0.6478 vs 0.6459 apart - see FINDING-002.md). Each question
    # now names its own po_number so it is answerable from its own document alone, while still
    # requiring the model to read the differing part number rather than just restating it - do
    # not "simplify" this back to the original shared wording.
    "po-004": [
        RetrievalGold(
            question_id="po-004-q1",
            doc_id="po-004",
            question=(
                "What is the exact part number for the laser-cut steel bracket on purchase "
                "order PO-1004?"
            ),
            question_class="near_duplicate",
            anchors=[Anchor(page_number=1, text="1. Part Number: 4500123456")],
        ),
        RetrievalGold(
            question_id="po-004-q2",
            doc_id="po-004",
            question="What quantity was ordered for part number 4500123456?",
            question_class="near_duplicate",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "1. Part Number: 4500123456\n"
                        "   Description: Laser-cut steel bracket, 3mm gauge\n"
                        "   Quantity: 60"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-004-q3",
            doc_id="po-004",
            question="What is the requested delivery date for this purchase order?",
            question_class="absent",
            anchors=[],
        ),
    ],
    "po-005": [
        RetrievalGold(
            question_id="po-005-q1",
            doc_id="po-005",
            question=(
                "What is the exact part number for the laser-cut steel bracket on purchase "
                "order PO-1005?"
            ),
            question_class="near_duplicate",
            anchors=[Anchor(page_number=1, text="1. Part Number: 4500123457")],
        ),
        RetrievalGold(
            question_id="po-005-q2",
            doc_id="po-005",
            question="What quantity was ordered for part number 4500123457?",
            question_class="near_duplicate",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "1. Part Number: 4500123457\n"
                        "   Description: Laser-cut steel bracket, 3mm gauge\n"
                        "   Quantity: 60"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-005-q3",
            doc_id="po-005",
            question="What is the warranty period for the part ordered on this purchase order?",
            question_class="absent",
            anchors=[],
        ),
    ],
    "po-006": [
        RetrievalGold(
            question_id="po-006-q1",
            doc_id="po-006",
            question=(
                "What are the part numbers of the 30mm deep groove bearing and the 25mm "
                "angular contact bearing on this purchase order?"
            ),
            question_class="cross_page",
            anchors=[
                Anchor(
                    page_number=1,
                    text="1. Part Number: BRG-1001\n   Description: Deep groove ball bearing, 30mm",
                ),
                Anchor(
                    page_number=2,
                    text="3. Part Number: BRG-1003\n   Description: Angular contact bearing, 25mm",
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-006-q2",
            doc_id="po-006",
            question=(
                "What is the quantity ordered for part number BRG-1002, and separately for "
                "part number BRG-1004?"
            ),
            question_class="cross_page",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "2. Part Number: BRG-1002\n"
                        "   Description: Deep groove ball bearing, 40mm\n"
                        "   Quantity: 40\n"
                        "   Unit Price: 15.00"
                    ),
                ),
                Anchor(
                    page_number=2,
                    text=(
                        "4. Part Number: BRG-1004\n"
                        "   Description: Angular contact bearing, 35mm\n"
                        "   Quantity: 34\n"
                        "   Unit Price: 20.00"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-006-q3",
            doc_id="po-006",
            question=(
                "What are the part numbers of the first and last line items in this purchase "
                "order's table, and what is the purchase order's total amount?"
            ),
            question_class="cross_page",
            anchors=[
                Anchor(page_number=1, text="1. Part Number: BRG-1001"),
                Anchor(page_number=2, text="4. Part Number: BRG-1004"),
                Anchor(page_number=3, text="Total Amount: 2960.00"),
            ],
        ),
        RetrievalGold(
            question_id="po-006-q4",
            doc_id="po-006",
            question="Who is the buyer's contact person for this purchase order?",
            question_class="absent",
            anchors=[],
        ),
        RetrievalGold(
            question_id="po-006-q5",
            doc_id="po-006",
            question="What are the payment terms on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Payment Terms: Net 60")],
        ),
    ],
    "po-007": [
        RetrievalGold(
            question_id="po-007-q1",
            doc_id="po-007",
            question="Who is the supplier on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Supplier: Northfield Industrial Supply")],
        ),
        RetrievalGold(
            question_id="po-007-q2",
            doc_id="po-007",
            question="What is the unit price of the check valve?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "2. Part Number: VLV-2202\n"
                        "   Description: Check valve, 1in NPT\n"
                        "   Quantity: 9\n"
                        "   Unit Price: 35.00"
                    ),
                ),
            ],
        ),
    ],
    "po-008": [
        RetrievalGold(
            question_id="po-008-q1",
            doc_id="po-008",
            question="What total amount is printed on this purchase order?",
            question_class="line_item",
            anchors=[Anchor(page_number=1, text="Total Amount: 750.00")],
        ),
        RetrievalGold(
            question_id="po-008-q2",
            doc_id="po-008",
            question="What are the payment terms on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Payment Terms: Net 30")],
        ),
        RetrievalGold(
            question_id="po-008-q3",
            doc_id="po-008",
            question="What is the description of part number CBL-500?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text="1. Part Number: CBL-500\n   Description: THHN copper wire, 500ft spool",
                ),
            ],
        ),
    ],
    "po-009": [
        RetrievalGold(
            question_id="po-009-q1",
            doc_id="po-009",
            question="Who is the supplier on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Supplier: Smith & Sons Machining")],
        ),
        RetrievalGold(
            question_id="po-009-q2",
            doc_id="po-009",
            question="What is the quantity ordered for the CNC-machined spacer?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "1. Part Number: MCH-0090\n"
                        "   Description: CNC-machined spacer, brass\n"
                        "   Quantity: 20"
                    ),
                ),
            ],
        ),
    ],
    "po-010": [
        RetrievalGold(
            question_id="po-010-q1",
            doc_id="po-010",
            question="What is the unit price for part number BRK-7701?",
            question_class="line_item",
            anchors=[Anchor(page_number=1, text="   Quantity: 40\n   Unit Price: 110.00")],
        ),
        RetrievalGold(
            question_id="po-010-q2",
            doc_id="po-010",
            question="What is the ship-to site on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Ship-to Site: Hangar 4")],
        ),
        RetrievalGold(
            question_id="po-010-q3",
            doc_id="po-010",
            question=(
                "What certification standard applies to the bracket ordered on this "
                "purchase order?"
            ),
            question_class="absent",
            anchors=[],
        ),
    ],
    "po-011": [
        RetrievalGold(
            question_id="po-011-q1",
            doc_id="po-011",
            question="What currency symbol is printed on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Currency: $")],
        ),
        RetrievalGold(
            question_id="po-011-q2",
            doc_id="po-011",
            question="What is the quantity ordered for the carbide end mill?",
            question_class="line_item",
            anchors=[
                Anchor(
                    page_number=1,
                    text=(
                        "1. Part Number: TL-3305\n"
                        "   Description: Carbide end mill, 1/4in, 4-flute\n"
                        "   Quantity: 24"
                    ),
                ),
            ],
        ),
        RetrievalGold(
            question_id="po-011-q3",
            doc_id="po-011",
            question="What is the ISO 4217 currency code for this purchase order?",
            question_class="absent",
            anchors=[],
        ),
    ],
    "po-012": [
        RetrievalGold(
            question_id="po-012-q1",
            doc_id="po-012",
            question="Who is the supplier on this purchase order?",
            question_class="header_field",
            anchors=[Anchor(page_number=1, text="Supplier: Bramwell Logistics")],
        ),
        RetrievalGold(
            question_id="po-012-q2",
            doc_id="po-012",
            question="What is the part number of the first line item on this purchase order?",
            question_class="absent",
            anchors=[],
        ),
        RetrievalGold(
            question_id="po-012-q3",
            doc_id="po-012",
            question="What is the total amount on this purchase order?",
            question_class="absent",
            anchors=[],
        ),
    ],
}


def build_goldset(corpus_path: Path = CORPUS_PATH) -> GoldSet:
    """Load corpus.json and attach each document's hand-written retrieval questions."""
    specs = json.loads(corpus_path.read_text(encoding="utf-8"))

    items = []
    for spec in specs:
        doc_id = spec["doc_id"]
        expected = PurchaseOrder.model_validate(spec["purchase_order"])
        items.append(
            GoldItem(
                doc_id=doc_id,
                extraction=ExtractionGold(doc_id=doc_id, expected=expected),
                retrieval=_QUESTIONS[doc_id],
            )
        )

    return GoldSet(
        version=_VERSION,
        items=items,
        chunking=ChunkingConfig(chunk_size=CHUNK_SIZE, overlap=OVERLAP),
    )


def write_goldset(goldset: GoldSet, output_path: Path = GOLDSET_PATH) -> None:
    """Write the goldset as indented JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(goldset.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build the gold set, validate its anchors, and write it. Writes nothing on failure."""
    goldset = build_goldset(corpus_path=CORPUS_PATH)
    validate_anchors(goldset, CHUNK_SIZE, OVERLAP)
    write_goldset(goldset, output_path=GOLDSET_PATH)
    question_count = sum(len(item.retrieval) for item in goldset.items)
    print(f"Wrote {len(goldset.items)} items / {question_count} questions to {GOLDSET_PATH}")


if __name__ == "__main__":
    main()

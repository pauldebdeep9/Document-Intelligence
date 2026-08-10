"""Synthetic purchase order corpus.

THE INVARIANT: the gold record is generated first and the PDF is rendered from
it. Never the other way round. Annotating generated documents afterwards
reintroduces exactly the labelling error synthetic data exists to eliminate, and
a wrong gold set makes every downstream eval number meaningless.

Concretely this means `_make_record()` decides the truth, `_render_pdf()` is a
pure function of that truth, and `gold/extraction/<id>.json` is written from the
same object the renderer consumed.

Gold is stored in two forms:
  raw        strings exactly as rendered on the page. Scores the LLM extraction,
             which the prompt instructs to return values verbatim.
  normalised typed values (ISO dates, Decimal amounts). Scores the wrapped
             record after extract/ applies validators.
Collapsing these would make a date-format bug indistinguishable from a
normalisation bug.

Deliberate difficulty, controlled by seed so runs are reproducible:
  * Date format varies by site (SG d/m/Y, US m/d/Y, DE d.m.Y). Roughly a third
    of dates are ambiguous (day <= 12), which is the single most common silent
    extraction error in a global supply chain.
  * Optional fields are omitted at a fixed rate, so "correctly absent" is a
    populated outcome class rather than a theoretical one.
  * Line counts run long enough to wrap onto a second page.
  * Confusable supplier pairs and unmastered parts come from the master data.
  * A subset omits the printed extended price, testing that the extractor
    returns null rather than computing it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from pypdf import PdfReader
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

INCOTERMS = ["FOB", "DDP", "EXW", "CIP", "DAP"]
PAYMENT_TERMS = ["Net 30", "Net 45", "Net 60", "2/10 Net 30"]
CURRENCY_BY_REGION = {"APAC": "SGD", "AMER": "USD", "EMEA": "EUR"}
BUYERS = ["A. Tan", "M. Weber", "J. Ruiz", "K. Nakamura", "P. Osei"]

# reportlab stamps /CreationDate and /ModDate from the wall clock unless told
# otherwise, and folds that timestamp into the PDF's /ID -- so two runs with
# the identical --seed produced byte-different PDFs (different content hash,
# different doc_id) despite every gold field matching. SOURCE_DATE_EPOCH is
# reportlab's (and the wider reproducible-builds ecosystem's) documented hook
# for this; deriving it from the seed, not hardcoding one constant, means two
# different seeds stay visibly distinguishable while the same seed is always
# byte-identical. 946684800 = 2000-01-01T00:00:00Z, an arbitrary fixed base.
_PDF_EPOCH_BASE = 946684800


def _pin_pdf_timestamps(seed: int) -> None:
    os.environ["SOURCE_DATE_EPOCH"] = str(_PDF_EPOCH_BASE + seed)

# Omission rates for optional fields. Tuned so most documents have at least one
# absent field; otherwise 'correct_absent' never fires in the harness.
OMIT = {
    "supplier_id": 0.25,
    "incoterms": 0.20,
    "payment_terms": 0.15,
    "buyer_contact": 0.30,
    "total_amount": 0.10,
}


# --------------------------------------------------------------------------
# 1. Truth
# --------------------------------------------------------------------------

def _make_record(rng: random.Random, site: dict[str, Any],
                 masters: dict[str, Any]) -> dict[str, Any]:
    """Build the gold record. This is the source of truth for the document.

    `site` is assigned by the caller rather than sampled here: site drives both
    the group grant and the date format, so random assignment at n=20 leaves
    whole regions with two documents and the corresponding principal with almost
    nothing to retrieve.
    """
    supplier = rng.choice(masters["suppliers"])
    currency = CURRENCY_BY_REGION[site["region"]]
    po_date = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 540))

    # 30+ lines genuinely overflow A4 and force the header row to reprint.
    # An earlier version used 14/18 and asserted they wrapped; they did not.
    n_lines = rng.choice([2, 3, 4, 5, 8, 30, 42])
    pool = masters["parts"] + masters["unmastered_parts"]
    # Whether extended prices are printed is a property of the DOCUMENT, not of
    # individual lines: a real ERP either prints the column or it does not.
    # Rolling per line made 15/20 documents partially unpriced and left the
    # arithmetic cross-check with almost no clean cases to validate against.
    omit_extended = rng.random() < 0.20
    lines = []
    for i in range(n_lines):
        part = rng.choice(pool)
        qty = Decimal(rng.choice([1, 2, 5, 10, 24, 50, 100, 250]))
        unit_price = (Decimal(rng.randrange(150, 240000)) / 100).quantize(Decimal("0.01"))
        extended = (qty * unit_price).quantize(Decimal("0.01"))
        print_extended = not omit_extended
        lines.append({
            "line_number": (i + 1) * 10,
            "part_number": part["part_number"],
            "description": part["description"],
            "quantity": qty,
            "unit_of_measure": part["uom"],
            "unit_price": unit_price,
            "extended_price": extended if print_extended else None,
            "promised_date": po_date + timedelta(days=rng.randrange(14, 120)),
        })

    total = sum((ln["extended_price"] or Decimal(0) for ln in lines), Decimal(0))
    # If any line omitted its extended price, the printed total still reflects
    # the true sum — so the arithmetic cross-check must tolerate that case
    # rather than flagging it as a conflict.
    true_total = sum(
        (Decimal(ln["quantity"]) * ln["unit_price"] for ln in lines), Decimal(0)
    ).quantize(Decimal("0.01"))

    record: dict[str, Any] = {
        "po_number": f"45{rng.randrange(10_000_000, 99_999_999):08d}"[:10],
        "po_date": po_date,
        "supplier_name": supplier["name"],
        "supplier_id": supplier["supplier_id"],
        "ship_to_site": site["name"],
        "incoterms": rng.choice(INCOTERMS),
        "payment_terms": rng.choice(PAYMENT_TERMS),
        "currency": currency,
        "total_amount": true_total,
        "buyer_contact": rng.choice(BUYERS),
        "lines": lines,
        "_site": site,
        "_supplier": supplier,
        "_all_lines_priced": all(ln["extended_price"] is not None for ln in lines),
    }

    # Omit optional fields. Omission is recorded as None in the gold, which the
    # harness scores as 'correct_absent' when extraction also returns null.
    for field_name, rate in OMIT.items():
        if rng.random() < rate:
            record[field_name] = None
    return record


def _fmt_date(d: date, fmt: str) -> str:
    return d.strftime(fmt)


def _is_ambiguous(d: date, fmt: str) -> bool:
    """d/m/Y and m/d/Y are indistinguishable when the day is 12 or lower."""
    return fmt in {"%d/%m/%Y", "%m/%d/%Y"} and d.day <= 12


def _money(v: Decimal) -> str:
    return f"{v:,.2f}"


# --------------------------------------------------------------------------
# 2. Rendering — a pure function of the record
# --------------------------------------------------------------------------

def _render_pdf(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """Render the PDF and return the raw (as-printed) gold strings."""
    site = record["_site"]
    dfmt = site["date_format"]
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    label = ParagraphStyle("label", parent=small, textColor=colors.grey)

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Purchase Order {record['po_number']}",
        # invariant=True (on top of SOURCE_DATE_EPOCH, set by
        # _pin_pdf_timestamps) also suppresses reportlab's per-build PDF
        # comments. creator/producer are pinned explicitly so a reportlab
        # version bump -- which could change its default producer string --
        # can never change the corpus hash on its own.
        invariant=True,
        creator="isc-docint gen_corpus.py",
        producer="isc-docint synthetic corpus",
    )
    flow: list[Any] = []
    raw: dict[str, Any] = {}

    flow.append(Paragraph("<b>PURCHASE ORDER</b>", styles["Title"]))
    flow.append(Spacer(1, 4 * mm))

    # Header block, rendered as a two-column key/value grid.
    raw["po_number"] = record["po_number"]
    raw["po_date"] = _fmt_date(record["po_date"], dfmt) if record["po_date"] else None
    raw["supplier_name"] = record["supplier_name"]
    raw["supplier_id"] = record["supplier_id"]
    raw["ship_to_site"] = record["ship_to_site"]
    raw["incoterms"] = record["incoterms"]
    raw["payment_terms"] = record["payment_terms"]
    raw["currency"] = record["currency"]
    raw["buyer_contact"] = record["buyer_contact"]
    raw["total_amount"] = _money(record["total_amount"]) if record["total_amount"] else None

    header_rows: list[list[Any]] = [
        [Paragraph("PO Number", label), Paragraph(f"<b>{raw['po_number']}</b>", small),
         Paragraph("Order Date", label), Paragraph(raw["po_date"] or "", small)],
        [Paragraph("Supplier", label), Paragraph(raw["supplier_name"], small),
         Paragraph("Vendor Code", label), Paragraph(raw["supplier_id"] or "", small)],
        [Paragraph("Ship To", label), Paragraph(raw["ship_to_site"], small),
         Paragraph("Currency", label), Paragraph(raw["currency"], small)],
        [Paragraph("Incoterms", label), Paragraph(raw["incoterms"] or "", small),
         Paragraph("Payment Terms", label), Paragraph(raw["payment_terms"] or "", small)],
        [Paragraph("Buyer", label), Paragraph(raw["buyer_contact"] or "", small),
         Paragraph("Delivery Address", label), Paragraph(site["address"], small)],
    ]
    ht = Table(header_rows, colWidths=[26 * mm, 58 * mm, 26 * mm, 64 * mm])
    ht.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#DDDDDD")),
    ]))
    flow.append(ht)
    flow.append(Spacer(1, 6 * mm))

    # Line item table. repeatRows=1 reprints the header on page 2, which is what
    # makes the wrapped-table case realistic for table reconstruction.
    head = ["Item", "Part Number", "Description", "Qty", "UoM",
            "Unit Price", "Extended", "Promised"]
    body: list[list[Any]] = [[Paragraph(f"<b>{h}</b>", small) for h in head]]
    raw_lines = []
    for ln in record["lines"]:
        ext = _money(ln["extended_price"]) if ln["extended_price"] is not None else ""
        promised = _fmt_date(ln["promised_date"], dfmt)
        body.append([
            Paragraph(str(ln["line_number"]), small),
            Paragraph(ln["part_number"], small),
            Paragraph(ln["description"], small),
            Paragraph(f"{ln['quantity']:,}", small),
            Paragraph(ln["unit_of_measure"], small),
            Paragraph(_money(ln["unit_price"]), small),
            Paragraph(ext, small),
            Paragraph(promised, small),
        ])
        raw_lines.append({
            "line_number": ln["line_number"],
            "part_number": ln["part_number"],
            "description": ln["description"],
            "quantity": f"{ln['quantity']:,}",
            "unit_of_measure": ln["unit_of_measure"],
            "unit_price": _money(ln["unit_price"]),
            "extended_price": ext or None,
            "promised_date": promised,
        })
    raw["lines"] = raw_lines

    lt = Table(body, colWidths=[11*mm, 26*mm, 47*mm, 12*mm, 11*mm, 22*mm, 24*mm, 21*mm],
               repeatRows=1)
    lt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
        ("ALIGN", (3, 1), (6, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    flow.append(lt)
    flow.append(Spacer(1, 4 * mm))

    if raw["total_amount"] is not None:
        tt = Table([[Paragraph("<b>Order Total</b>", small),
                     Paragraph(f"<b>{record['currency']} {raw['total_amount']}</b>", small)]],
                   colWidths=[150 * mm, 24 * mm])
        tt.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "RIGHT")]))
        flow.append(tt)

    flow.append(Spacer(1, 6 * mm))
    flow.append(Paragraph(
        "This order is subject to the Master Supply Agreement in force between the "
        "parties. Goods must be accompanied by a Certificate of Conformance. "
        "Rockwell Automation ISC — internal synthetic document, not a real order.",
        label))

    doc.build(flow)
    return raw


# --------------------------------------------------------------------------
# 3. Permissions
# --------------------------------------------------------------------------

GROUP_BY_REGION = {"APAC": "buyers-apac", "EMEA": "buyers-emea", "AMER": "engineering"}


JURISDICTION_BY_REGION = {"APAC": "SG", "AMER": "US", "EMEA": "DE"}


def sensitivity_plan(n: int) -> list[str]:
    """Stratified, not sampled.

    At n=20 a 10% random draw for export_controlled produces zero documents
    roughly one run in eight — and a corpus with no export-controlled documents
    silently disables the jurisdiction branch of the ACL suite. Small corpora
    must guarantee coverage of the classes the tests depend on, not hope for it.
    """
    n_export = max(2, round(n * 0.10))
    n_conf = max(3, round(n * 0.35))
    plan = (["export_controlled"] * n_export
            + ["confidential"] * n_conf
            + ["internal"] * (n - n_export - n_conf))
    return plan[:n]


def _make_acl(record: dict[str, Any], sensitivity: str) -> dict[str, Any]:
    """Assign permissions so the adversarial pairs in the identity graph hold.

    Region drives the group grant, so an APAC buyer cannot read EMEA orders.
    The confidential subset defeats an internal-only clearance even with the
    right group. The export-controlled subset additionally requires a matching
    jurisdiction, which no principal outside trade-compliance holds.
    """
    site = record["_site"]
    return {
        "allow_terms": [f"group:{GROUP_BY_REGION[site['region']]}", f"site:{site['site_id']}"],
        "deny_terms": [],
        "sensitivity": sensitivity,
        "jurisdictions": (
            [JURISDICTION_BY_REGION[site["region"]]]
            if sensitivity == "export_controlled" else []
        ),
    }


# --------------------------------------------------------------------------
# 4. Gold
# --------------------------------------------------------------------------

def _normalised(record: dict[str, Any]) -> dict[str, Any]:
    def _d(v: Any) -> Any:
        return v.isoformat() if isinstance(v, date) else (str(v) if isinstance(v, Decimal) else v)

    out = {k: _d(v) for k, v in record.items() if not k.startswith("_") and k != "lines"}
    out["lines"] = [{k: _d(v) for k, v in ln.items()} for ln in record["lines"]]
    return out


def _gold(record: dict[str, Any], raw: dict[str, Any], doc_name: str,
          n_pages: int) -> dict[str, Any]:
    site = record["_site"]
    dfmt = site["date_format"]
    # Every date field on the record, not just the header: a line item's
    # promised_date is exactly as ambiguous as po_date under the same site
    # format, and there are far more of them per document. Undercounting
    # here understated the corpus's own difficulty -- a P1-03 eval run
    # found 42 ambiguous-date extraction errors against this field
    # reporting only 3, because it never looked at line items at all.
    ambiguous = [
        f for f, v in (("po_date", record["po_date"]),)
        if v is not None and _is_ambiguous(v, dfmt)
    ]
    ambiguous += [
        f"lines[{ln['line_number']}].promised_date"
        for ln in record["lines"]
        if ln["promised_date"] is not None and _is_ambiguous(ln["promised_date"], dfmt)
    ]
    return {
        "document": doc_name,
        "doc_type": "purchase_order",
        "raw": raw,
        "normalised": _normalised(record),
        "meta": {
            "site_id": site["site_id"],
            "region": site["region"],
            "date_format": dfmt,
            "ambiguous_date_fields": ambiguous,
            "n_lines": len(record["lines"]),
            # Measured from the rendered PDF, not predicted from line count.
            # Predicting it was wrong: 18 lines fits A4, so the corpus claimed
            # multi-page coverage it did not have.
            "n_pages": n_pages,
            "wraps_pages": n_pages > 1,
            "all_lines_priced": record["_all_lines_priced"],
            "supplier_in_master": True,
            "unmastered_parts": sorted({
                ln["part_number"] for ln in record["lines"]
                if ln["part_number"].startswith(("TMP-", "SPR-"))
            }),
            "absent_fields": sorted(
                k for k in OMIT if record.get(k) is None
            ),
        },
    }


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--types", default="purchase_order")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--gold", type=Path, default=Path("data/gold/extraction"))
    ap.add_argument("--masters", type=Path, default=Path("data/masters"))
    ap.add_argument("--seed", type=int, default=2608)
    args = ap.parse_args()

    if args.types != "purchase_order":
        raise SystemExit(f"only purchase_order is implemented; got {args.types!r}")

    masters = {
        "suppliers": json.loads((args.masters / "suppliers.json").read_text()),
        "parts": json.loads((args.masters / "parts.json").read_text()),
        "sites": json.loads((args.masters / "sites.json").read_text()),
    }
    masters["unmastered_parts"] = json.loads(
        (args.masters / "unmastered_parts.json").read_text()
    )

    args.out.mkdir(parents=True, exist_ok=True)
    args.gold.mkdir(parents=True, exist_ok=True)
    _pin_pdf_timestamps(args.seed)
    rng = random.Random(args.seed)

    plan = sensitivity_plan(args.n)
    rng.shuffle(plan)
    # Round-robin so every region gets even coverage and every buyer principal
    # has a comparable corpus to retrieve from.
    sites = [masters["sites"][i % len(masters["sites"])] for i in range(args.n)]
    rng.shuffle(sites)

    manifest = []
    for i in range(args.n):
        name = f"po_{i:03d}"
        record = _make_record(rng, sites[i], masters)
        pdf_path = args.out / f"{name}.pdf"
        raw = _render_pdf(record, pdf_path)

        acl = _make_acl(record, plan[i])
        (args.out / f"{name}.pdf.acl.json").write_text(json.dumps(acl, indent=2))

        n_pages = len(PdfReader(str(pdf_path)).pages)
        gold = _gold(record, raw, f"{name}.pdf", n_pages)
        (args.gold / f"{name}.json").write_text(json.dumps(gold, indent=2))
        manifest.append({
            "name": name, "site": record["_site"]["site_id"],
            "sensitivity": acl["sensitivity"], "n_lines": len(record["lines"]),
            "n_pages": n_pages,
        })

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _print_summary(manifest, args.gold)


def _print_summary(manifest: list[dict[str, Any]], gold_dir: Path) -> None:
    from collections import Counter
    print(f"generated {len(manifest)} purchase orders")
    print("  sites       :", dict(Counter(m["site"] for m in manifest)))
    print("  sensitivity :", dict(Counter(m["sensitivity"] for m in manifest)))
    print("  multi-page  :", sum(m["n_pages"] > 1 for m in manifest),
          f"(max {max(m['n_pages'] for m in manifest)} pages)")
    golds = [json.loads(p.read_text()) for p in sorted(gold_dir.glob("*.json"))]
    print("  ambiguous dates :", sum(bool(g["meta"]["ambiguous_date_fields"]) for g in golds))
    print("  unmastered parts:", sum(bool(g["meta"]["unmastered_parts"]) for g in golds))
    print("  unpriced lines  :", sum(not g["meta"]["all_lines_priced"] for g in golds))
    absent = Counter(f for g in golds for f in g["meta"]["absent_fields"])
    print("  absent fields   :", dict(absent))


if __name__ == "__main__":
    main()

"""Small in-memory worlds for the aggregate/ tests: documents with ACLs,
extraction records with chosen confidences, and header/table/footer chunks
shaped like the real chunker's output (the footer is the only chunk that
prints the order total; table chunks carry line_range)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from isc.common.confidence import Confidence, Signal
from isc.common.config import Settings
from isc.llm.ports import LLMResult
from isc.models.acl import AclSet, Sensitivity
from isc.models.chunk import Chunk
from isc.models.document import DocType, Document
from isc.models.records.base import ExtractedField
from isc.models.records.purchase_order import POLine, PurchaseOrder
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore

REPO = Path(__file__).resolve().parents[1]
MASTERS = REPO / "data" / "masters"


def ef(value, score: float = 0.99) -> ExtractedField:
    return ExtractedField(value=value, confidence=Confidence.of(Signal.MODEL, score))


def absent() -> ExtractedField:
    return ExtractedField.missing()


def money(v: Decimal) -> str:
    return f"{v:,.2f}"


@dataclass
class Line:
    number: int
    part: str
    price: Decimal | None
    price_conf: float = 0.99


@dataclass
class Po:
    doc_id: str
    po_number: str
    supplier_name: str
    supplier_id: str | None
    currency: str | None
    total: Decimal | None
    allow: set[str] = field(default_factory=lambda: {"everyone:*"})
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    jurisdictions: frozenset[str] = frozenset()
    lines: list[Line] = field(default_factory=list)
    total_conf: float = 0.99
    currency_conf: float = 0.99
    supplier_conf: float = 0.99
    extracted: bool = True
    indexed: bool = True
    # What the footer chunk prints; defaults to the extracted total. Set it
    # differently to simulate a value that cannot be located in the text.
    printed_total: Decimal | None = None

    def acl(self) -> AclSet:
        return AclSet(allow_terms=frozenset(self.allow), sensitivity=self.sensitivity,
                      jurisdictions=self.jurisdictions)

    def record(self) -> PurchaseOrder:
        return PurchaseOrder(
            document_id=self.doc_id,
            po_number=ef(self.po_number),
            supplier_name=ef(self.supplier_name, self.supplier_conf),
            supplier_id=(ef(self.supplier_id, self.supplier_conf) if self.supplier_id
                         else absent()),
            currency=ef(self.currency, self.currency_conf) if self.currency else absent(),
            total_amount=ef(self.total, self.total_conf) if self.total is not None else absent(),
            lines=[POLine(line_number=ef(ln.number), part_number=ef(ln.part),
                          unit_price=(ef(ln.price, ln.price_conf) if ln.price is not None
                                      else absent()))
                   for ln in self.lines],
        )

    def chunks(self) -> list[Chunk]:
        filters = {"po_number": self.po_number}
        if self.supplier_id:
            filters["supplier_id"] = self.supplier_id
        common = dict(document_id=self.doc_id, acl=self.acl(), filters=filters,
                      doc_type=DocType.PURCHASE_ORDER)
        out = [Chunk(id=f"chk_{self.doc_id}_0", ordinal=0, **common, text=(
            f"PURCHASE ORDER\nPO Number {self.po_number}\n"
            f"Supplier {self.supplier_name}  Vendor Code {self.supplier_id or ''}\n"
            f"Currency {self.currency or ''}"))]
        if self.lines:
            rows = "\n".join(
                f"| {ln.number} | {ln.part} | item | 1 | EA | "
                f"{money(ln.price) if ln.price is not None else ''} |"
                for ln in self.lines)
            out.append(Chunk(id=f"chk_{self.doc_id}_1", ordinal=1, is_table=True,
                             line_range=(self.lines[0].number, self.lines[-1].number),
                             **common, text="| Item | Part | Desc | Qty | UoM | Unit |\n" + rows))
        printed = self.printed_total if self.printed_total is not None else self.total
        out.append(Chunk(id=f"chk_{self.doc_id}_2", ordinal=len(out), **common, text=(
            f"{self.currency or ''}\nOrder Total  {money(printed) if printed else ''}\n"
            "Subject to the Master Supply Agreement.")))
        return out


def build(tmp_path: Path, pos: list[Po]) -> tuple[SqliteDocStore, LocalVectorStore]:
    docs = SqliteDocStore(tmp_path / "docstore.sqlite")
    store = LocalVectorStore(tmp_path / "store.pkl")
    for po in pos:
        docs.upsert_document(Document(
            id=po.doc_id, source_uri=f"{po.doc_id}.pdf", content_sha256=po.doc_id,
            doc_type=DocType.PURCHASE_ORDER, acl=po.acl()))
        if po.extracted:
            docs.upsert_record(po.doc_id, DocType.PURCHASE_ORDER.value,
                               po.record().model_dump(mode="json"))
        if po.indexed:
            chunks = po.chunks()
            store.add(chunks, [[1.0, 0.0]] * len(chunks), settings_fingerprint="fp")
    return docs, store


class PlanChat:
    """Scripted planner: returns one fixed plan (dict) or raw text."""

    def __init__(self, plan: dict | str) -> None:
        self.plan = plan
        self.calls = 0

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        self.calls += 1
        text = self.plan if isinstance(self.plan, str) else json.dumps(self.plan)
        return LLMResult(text=text, model="scripted-planner")


def plan(operation: str, supplier: str | None = None, part: str | None = None,
         currency: str | None = None) -> dict:
    return {"operation": operation, "supplier": supplier, "part_number": part,
            "currency": currency}


def settings() -> Settings:
    return Settings()   # paths.data -> repo data/, so data/masters is the real master

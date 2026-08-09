"""Parsed document representation: the contract between parse/ and everything after."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from isc.common.confidence import Confidence
from isc.models.acl import AclSet


class DocType(StrEnum):
    """The corpus. Extend deliberately: each type needs a schema, prompt, and gold set."""

    MSA = "msa"
    SUPPLIER_CONTRACT = "supplier_contract"
    PURCHASE_ORDER = "purchase_order"
    INVOICE = "invoice"
    ASN = "asn"
    CERT_OF_CONFORMANCE = "cert_of_conformance"
    ROHS_DECLARATION = "rohs_declaration"
    REACH_DECLARATION = "reach_declaration"
    COMMERCIAL_INVOICE = "commercial_invoice"
    PACKING_LIST = "packing_list"
    CERT_OF_ORIGIN = "cert_of_origin"
    NCR = "ncr"
    EIGHT_D_REPORT = "eight_d_report"
    ENGINEERING_CHANGE_NOTICE = "engineering_change_notice"
    SOP = "sop"
    UNKNOWN = "unknown"


class BlockType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    KEY_VALUE = "key_value"
    HEADER = "page_header"
    FOOTER = "page_footer"
    FIGURE = "figure"


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class Span(BaseModel):
    """A pointer back into the source. Every extracted value must carry one:
    a field without provenance cannot be reviewed, so it cannot be trusted."""

    model_config = ConfigDict(frozen=True)
    document_id: str
    page: int
    bbox: BBox | None = None
    text: str = ""


class Table(BaseModel):
    """Reconstructed table. Rows are lists of cell strings; header is row 0 by
    convention unless header_rows says otherwise."""

    rows: list[list[str]] = Field(default_factory=list)
    header_rows: int = 1
    caption: str = ""
    ocr_confidence: float | None = None

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        head, *body = self.rows
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        out += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(out)


class Block(BaseModel):
    """Smallest addressable unit produced by parse/."""

    id: str
    type: BlockType
    text: str = ""
    page: int = 1
    bbox: BBox | None = None
    table: Table | None = None
    ocr_confidence: float | None = None
    layout_confidence: float | None = None
    reading_order: int = 0


class Page(BaseModel):
    number: int
    width: float = 0.0
    height: float = 0.0
    blocks: list[Block] = Field(default_factory=list)
    is_scanned: bool = False


class Document(BaseModel):
    """The parse/ output artifact, serialised to runs/<run_id>/parse/<doc_id>.json."""

    id: str
    source_uri: str
    content_sha256: str
    doc_type: DocType = DocType.UNKNOWN
    doc_type_confidence: float | None = None
    acl: AclSet
    pages: list[Page] = Field(default_factory=list)
    parser: str = ""            # which parser in the fallback chain succeeded
    parser_degraded: bool = False
    # Set by parse_with_fallback. Carried in the artifact so extract/ folds it
    # into record confidence instead of recomputing the parser penalty.
    parse_confidence: Confidence | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def blocks(self) -> list[Block]:
        return [b for p in self.pages for b in p.blocks]

    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks() if b.text)

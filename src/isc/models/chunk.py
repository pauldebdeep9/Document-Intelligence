"""Index records. A Chunk cannot exist without ACL terms — that is the invariant."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from isc.models.acl import AclSet
from isc.models.document import BBox, DocType


class Chunk(BaseModel):
    """One retrievable unit.

    `acl` is required and non-defaulted: there is no constructor path that yields
    a chunk without permissions. Combined with AclSet's non-empty validator, the
    type system refuses to represent an unrestricted chunk.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    ordinal: int
    text: str
    acl: AclSet

    doc_type: DocType = DocType.UNKNOWN
    page_start: int = 1
    page_end: int = 1
    bboxes: tuple[BBox, ...] = ()
    section_path: tuple[str, ...] = ()   # breadcrumb from headings, for citations
    is_table: bool = False
    token_count: int = 0
    # Filterable metadata projected from extraction (supplier_id, po_number, ...).
    # Enables metadata-filter inference in retrieve/ without a second index.
    filters: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _guard(self) -> "Chunk":
        if not self.text.strip():
            raise ValueError(f"chunk {self.id} has no text")
        if not self.acl.allow_terms:
            raise ValueError(f"chunk {self.id} has no allow_terms")
        return self

    def citation_label(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        section = " > ".join(self.section_path)
        return f"{self.document_id} {pages}" + (f" [{section}]" if section else "")


class ScoredChunk(BaseModel):
    """A chunk with its retrieval scores. Kept separate so Chunk stays storage-shaped."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0

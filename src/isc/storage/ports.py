"""Storage boundary. Local implementations now; Azure equivalents later.

  BlobStore   -> local dir        -> Azure Blob / SharePoint document library
  DocStore    -> SQLite           -> Azure SQL / Cosmos
  VectorStore -> numpy brute force-> Azure AI Search (hybrid + filters)

VectorStore.search takes a Principal, not an optional filter string. A signature
that permits an unfiltered search is a signature that will eventually be called
that way.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from isc.models.acl import Principal
from isc.models.chunk import Chunk, ScoredChunk
from isc.models.document import Document


@runtime_checkable
class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...


@runtime_checkable
class DocStore(Protocol):
    def upsert_document(self, doc: Document) -> None: ...
    def get_document(self, doc_id: str) -> Document | None: ...
    def list_documents(self, doc_type: str | None = None) -> list[str]: ...
    def upsert_record(self, doc_id: str, doc_type: str, payload: dict[str, Any]) -> None: ...
    def get_record(self, doc_id: str) -> dict[str, Any] | None: ...


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        principal: Principal,
        *,
        k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """MUST apply the principal's ACL before ranking, not after."""
        ...

    def search_lexical(
        self,
        query: str,
        principal: Principal,
        *,
        k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]: ...

    def count(self) -> int: ...

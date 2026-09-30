"""The only way the aggregate path reads records: already permission-filtered.

The `records` table has no ACL column -- permissions live on Document. A
query built straight on that table would be the first code path in this
repo that can see every document regardless of who is asking, and an
aggregate leaks in ways a chunk list does not: a total that includes a
restricted order discloses its amount, and a count discloses its existence,
even when no restricted text is ever shown. So the principal is a required
positional argument, the Document's own AclSet is checked BEFORE its record
is loaded, and there is no overload that skips it -- the same shape as
LocalVectorStore._permitted() (ADR 0004).

Chunks come from LocalVectorStore.document_chunks(), which applies the same
per-chunk may_read() the search paths do: a chunk's ACL is its document's
ACL projected at index time, so this is a second check, not a different one.

P1 scale: this walks every document per question (20 here). The Azure
landing is the same contract as a query -- records in Azure SQL, permitted
set by a join on the principal's expanded acl terms (optionally enforced
again with row-level security), never "load everything then filter".
"""

from __future__ import annotations

from dataclasses import dataclass

from isc.common.tracing import span
from isc.models.acl import Principal
from isc.models.chunk import Chunk
from isc.models.document import DocType, Document
from isc.models.records.purchase_order import PurchaseOrder
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore


@dataclass(frozen=True)
class VisibleRecord:
    document: Document
    record: PurchaseOrder
    # This principal's permitted chunks of the document, in ordinal order --
    # what a value's citation must point at. Empty if the document is not
    # indexed; execute.py then cannot cite it and excludes its values.
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class RecordView:
    records: tuple[VisibleRecord, ...]
    # Permitted documents of this type with no extraction record yet. Only
    # ever counts documents this principal may read -- reporting it cannot
    # disclose anything they could not already open.
    unextracted: tuple[str, ...] = ()


def visible_records(
    principal: Principal, docs: SqliteDocStore, store: LocalVectorStore,
    doc_type: DocType = DocType.PURCHASE_ORDER,
) -> RecordView:
    if doc_type is not DocType.PURCHASE_ORDER:
        raise NotImplementedError(f"aggregate path has no record mapping for {doc_type}")
    records: list[VisibleRecord] = []
    unextracted: list[str] = []
    with span("aggregate.source", principal=principal.id):
        for doc_id in docs.list_documents(doc_type.value):
            doc = docs.get_document(doc_id)
            if doc is None or not principal.may_read(doc.acl):
                continue  # permission check first; the record is never loaded
            payload = docs.get_record(doc_id)
            if payload is None:
                unextracted.append(doc_id)
                continue
            records.append(VisibleRecord(
                document=doc,
                record=PurchaseOrder.model_validate(payload),
                chunks=tuple(store.document_chunks(doc_id, principal)),
            ))
    return RecordView(records=tuple(records), unextracted=tuple(unextracted))

# ADR 0004: ACL correctness is an index invariant, not a retrieval feature

**Status:** accepted · **Date:** 2026-08-09

## Context
A knowledge assistant over SharePoint content will surface documents the asker
may not read unless permissions are enforced at every layer. This is the failure
that ends a pilot permanently: it is not a quality regression, it is an incident.

Post-filtering — retrieve, then drop results the user cannot see — is the common
shortcut and it is wrong. It leaks through result counts, score distributions and
reranker behaviour, and it silently degrades recall for restricted users in a way
that looks like a relevance bug.

## Decision
1. `AclSet.allow_terms` is validated non-empty at construction. An unlabelled
   document is nobody's, not everybody's. Public access requires an explicit
   `everyone:*` grant.
2. Group membership is expanded to a transitive closure at index time. Retrieval
   never walks the group graph — a filter that needs a lookup is a filter that
   gets skipped under latency pressure.
3. `Chunk.acl` is required and non-defaulted. There is no constructor path
   producing a chunk without permissions.
4. `VectorStore.search` takes a `Principal` positionally. There is no signature
   that permits an unfiltered search.
5. Filtering happens before ranking, on both the dense and lexical paths.
6. `NO_PERMITTED_RESULTS` and `NO_RESULTS` are indistinguishable to the user.
   Confirming that a document exists is itself a disclosure.
7. `tests/adversarial/acl/` exists from the first commit and may never be
   skipped, xfailed, or weakened to land a feature. One leak fails the eval run
   regardless of every other metric.

## Consequences
- Pre-filtering costs a full scan in the local store. Azure AI Search does this
  natively with an OData filter over a `Collection(Edm.String)` field.
- Sensitivity labels and export-control jurisdictions are modelled now even
  though SharePoint ACLs alone would not require them. Retrofitting export
  control into a permission model is considerably worse than carrying it early.

## Revisit if
Never for the invariant. The mechanism changes when the backing store changes.

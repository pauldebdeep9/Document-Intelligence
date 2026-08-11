"""Permission model. ACL correctness is a hard invariant, not a feature.

Design commitments, all of which are load-bearing:

  1. Chunks carry *expanded* principal ids (`acl_terms`), computed at index time.
     Retrieval must never need to walk the group graph — a filter that requires a
     lookup is a filter that gets skipped under latency pressure.

  2. Filtering is pre-search, not post-search. Post-filtering leaks through
     result counts, scores, and reranker behaviour, and it silently degrades
     recall for restricted users in a way that looks like a relevance bug.

  3. Deny wins. SharePoint has no explicit deny, but Purview labels and export
     control do, and modelling it now costs nothing.

  4. An empty acl_terms set is an error, never "public". Fail closed.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from isc.common.errors import AclViolation


class PrincipalType(StrEnum):
    USER = "user"
    GROUP = "group"
    SITE = "site"          # SharePoint site / library scope
    EVERYONE = "everyone"  # tenant-wide; still an explicit grant


class Sensitivity(StrEnum):
    """Mirrors a Purview label. Orthogonal to ACL: both must pass."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    EXPORT_CONTROLLED = "export_controlled"


_ORDER = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.CONFIDENTIAL: 2,
    Sensitivity.EXPORT_CONTROLLED: 3,
}


class Principal(BaseModel):
    """A user, resolved with their full group closure. Built once per request."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: PrincipalType = PrincipalType.USER
    display_name: str = ""
    group_ids: frozenset[str] = Field(default_factory=frozenset)
    site_ids: frozenset[str] = Field(default_factory=frozenset)
    clearance: Sensitivity = Sensitivity.INTERNAL
    # Export control: a user may hold no jurisdiction, which is not the same as
    # holding all of them. Empty means "no export-controlled access".
    jurisdictions: frozenset[str] = Field(default_factory=frozenset)

    def terms(self) -> frozenset[str]:
        """Everything this principal can match against a chunk's acl_terms."""
        return frozenset(
            {f"user:{self.id}", "everyone:*"}
            | {f"group:{g}" for g in self.group_ids}
            | {f"site:{s}" for s in self.site_ids}
        )

    def may_read(self, acl: AclSet) -> bool:
        if acl.deny_terms & self.terms():
            return False
        if not acl.allow_terms:
            raise AclViolation("chunk reached retrieval with empty allow_terms")
        if not (acl.allow_terms & self.terms()):
            return False
        if _ORDER[acl.sensitivity] > _ORDER[self.clearance]:
            return False
        if acl.sensitivity is Sensitivity.EXPORT_CONTROLLED:
            if not acl.jurisdictions or not (acl.jurisdictions & self.jurisdictions):
                return False
        return True


class Grant(BaseModel):
    """One ACE from the source system, before expansion."""

    model_config = ConfigDict(frozen=True)

    principal_id: str
    principal_type: PrincipalType
    effect: str = "allow"  # allow | deny


class AclSet(BaseModel):
    """The projected, expanded permission state attached to a chunk or document."""

    model_config = ConfigDict(frozen=True)

    allow_terms: frozenset[str]
    deny_terms: frozenset[str] = Field(default_factory=frozenset)
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    jurisdictions: frozenset[str] = Field(default_factory=frozenset)
    source_uri: str = ""

    @field_validator("allow_terms")
    @classmethod
    def _non_empty(cls, v: frozenset[str]) -> frozenset[str]:
        # Fail closed at construction. There is no path that builds an
        # unrestricted chunk by omission.
        if not v:
            raise ValueError("allow_terms must not be empty; use 'everyone:*' explicitly")
        return v

    @classmethod
    def from_grants(
        cls,
        grants: Iterable[Grant],
        group_expansion: dict[str, set[str]],
        *,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
        jurisdictions: Iterable[str] = (),
        source_uri: str = "",
    ) -> AclSet:
        """Expand groups to their transitive member closure at index time."""
        allow: set[str] = set()
        deny: set[str] = set()
        for g in grants:
            bucket = deny if g.effect == "deny" else allow
            bucket.add(f"{g.principal_type}:{g.principal_id}")
            if g.principal_type is PrincipalType.GROUP:
                for member in group_expansion.get(g.principal_id, set()):
                    bucket.add(f"user:{member}")
        return cls(
            allow_terms=frozenset(allow),
            deny_terms=frozenset(deny),
            sensitivity=sensitivity,
            jurisdictions=frozenset(jurisdictions),
            source_uri=source_uri,
        )


def load_principals(acl_dir: Path) -> dict[str, Principal]:
    """Every named principal in the identity graph (data/acl/users.json),
    keyed by id. One reader for that file, not three: gen_gold.py's own
    _load_users(), cli.py's _resolve_principal() (a single-id lookup, still
    reads the same file shape) and eval/retrieval.py's runner -- which
    needs every principal at once (no_reader questions check up to 7 of
    them) rather than one at a time -- all draw from the same JSON shape."""
    raw = json.loads((acl_dir / "users.json").read_text())
    return {
        uid: Principal(
            id=uid, group_ids=frozenset(u["groups"]), site_ids=frozenset(u["sites"]),
            clearance=Sensitivity(u["clearance"]), jurisdictions=frozenset(u["jurisdictions"]),
        )
        for uid, u in raw.items()
    }

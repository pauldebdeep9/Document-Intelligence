"""Local directory source. The Graph connector replaces this, not the interface.

Delta sync contract (holds for both): a source yields SourceItems; the pipeline
decides what is new by content hash. Graph adds deltaLink tokens for change
detection, but hash-based dedupe stays as the correctness backstop, because
SharePoint reports metadata-only edits as content changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from isc.models.acl import AclSet, Sensitivity


@dataclass(frozen=True, slots=True)
class SourceItem:
    uri: str
    data: bytes
    acl: AclSet
    metadata: dict[str, str]


class LocalDirectorySource:
    """Reads files plus a sidecar <name>.acl.json carrying the permission state."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def items(self) -> Iterator[SourceItem]:
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix in {".json"} and p.name.endswith(".acl.json"):
                continue
            if p.suffix == ".json" or p.name == ".gitkeep":
                continue
            yield SourceItem(
                uri=str(p.relative_to(self.root)),
                data=p.read_bytes(),
                acl=self._acl_for(p),
                metadata={"filename": p.name, "suffix": p.suffix},
            )

    def _acl_for(self, p: Path) -> AclSet:
        sidecar = p.with_suffix(p.suffix + ".acl.json")
        if not sidecar.exists():
            # Fail closed: an unlabelled file is not public, it is nobody's.
            raise FileNotFoundError(
                f"no ACL sidecar for {p}; refusing to ingest without permissions"
            )
        raw = json.loads(sidecar.read_text())
        return AclSet(
            allow_terms=frozenset(raw["allow_terms"]),
            deny_terms=frozenset(raw.get("deny_terms", [])),
            sensitivity=Sensitivity(raw.get("sensitivity", "internal")),
            jurisdictions=frozenset(raw.get("jurisdictions", [])),
            source_uri=str(p),
        )


class GraphSource:
    """Microsoft Graph / SharePoint connector.

    Notes for when this gets wired (deliberately recorded now):
      * /sites/{id}/drive/root/delta for change tokens
      * permissions come from driveItem /permissions, NOT from the site default
      * group membership expansion via /groups/{id}/transitiveMembers
      * sensitivity labels via the extractSensitivityLabels endpoint
      * throttling is the dominant failure mode: honour Retry-After
    """

    def items(self) -> Iterator[SourceItem]:
        raise NotImplementedError("Graph connector: week 3 of the lab plan")

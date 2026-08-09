from __future__ import annotations

import numpy as np
import pytest

from isc.models.acl import AclSet, Principal, Sensitivity
from isc.models.chunk import Chunk


@pytest.fixture
def alice() -> Principal:
    return Principal(
        id="u_alice",
        group_ids=frozenset({"buyers-apac", "quality-sg"}),
        site_ids=frozenset({"site_sg01"}),
        clearance=Sensitivity.CONFIDENTIAL,
        jurisdictions=frozenset({"SG"}),
    )


@pytest.fixture
def frank() -> Principal:
    """Contractor. Lowest privilege principal in the identity graph."""
    return Principal(id="u_frank", group_ids=frozenset({"contractors"}),
                     clearance=Sensitivity.PUBLIC)


@pytest.fixture
def gita() -> Principal:
    return Principal(id="u_gita", group_ids=frozenset({"trade-compliance"}),
                     clearance=Sensitivity.EXPORT_CONTROLLED,
                     jurisdictions=frozenset({"US", "SG"}))


def make_chunk(cid: str, text: str, allow: set[str], **kw) -> Chunk:
    return Chunk(id=cid, document_id=f"d_{cid}", ordinal=0, text=text,
                 acl=AclSet(allow_terms=frozenset(allow), **kw))


@pytest.fixture
def vec():
    def _v(n: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        return rng.normal(size=n).tolist()
    return _v

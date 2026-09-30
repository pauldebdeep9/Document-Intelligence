from __future__ import annotations

import os

import numpy as np
import pytest

from isc.common.config import get_settings
from isc.models.acl import AclSet, Principal, Sensitivity
from isc.models.chunk import Chunk


@pytest.fixture(autouse=True)
def _no_isc_env(monkeypatch):
    """Every test starts with no ISC_* environment variable set.

    Since 334ce1c the environment outranks config/*.yaml, as intended -- which
    also means an ISC_* exported in a developer's shell (e.g.
    ISC_THRESHOLDS__REVIEW) silently changes test outcomes. A test that needs
    one sets it itself with monkeypatch.setenv. The settings cache is cleared
    on both sides so no test sees a Settings built under another's env.

    Not isolated here: .env and config/local.yaml. Those are files, read by
    the settings sources themselves, not environment variables.
    """
    for name in list(os.environ):
        if name.startswith("ISC_"):
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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

"""Thin FastAPI surface. Deliberately thin: the CLI is the primary interface.

The identity story here is a placeholder for Entra ID. What must not change is
the shape — a Principal is resolved from the request and passed down; no endpoint
accepts an optional principal, and none exposes an admin bypass.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from isc.models.acl import Principal

app = FastAPI(title="ISC Document Intelligence", version="0.1.0")


class AskRequest(BaseModel):
    question: str


def current_principal(
    x_user_id: str | None = Header(default=None),
    x_user_groups: str = Header(default=""),
) -> Principal:
    """Placeholder for Entra ID token validation.

    Real implementation: validate the bearer token, read oid/groups claims, and
    expand transitive membership via Graph with a short-lived cache. Header-based
    identity is for local development only and must never reach a shared host.
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return Principal(
        id=x_user_id,
        group_ids=frozenset(g for g in x_user_groups.split(",") if g),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest, principal: Principal = Depends(current_principal)) -> dict:
    raise HTTPException(status_code=501, detail="not implemented")


@app.get("/review")
def review_queue(principal: Principal = Depends(current_principal)) -> dict:
    raise HTTPException(status_code=501, detail="not implemented")

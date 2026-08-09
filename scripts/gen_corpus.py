"""Synthetic corpus generator.

Design notes for the real implementation:
  * Generate the gold record FIRST, then render the document from it. Rendering
    from known values is the only way to get field-level gold that is actually
    correct — annotating generated documents afterwards reintroduces the labelling
    error you were trying to avoid.
  * Vary hostile things on purpose: date formats per region, line items that wrap
    across pages, missing optional fields, a scanned-looking subset, and two
    suppliers with confusingly similar names.
  * Every document gets a .acl.json sidecar; ingest refuses files without one.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--types", default="purchase_order")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()
    raise NotImplementedError(
        "first slice: render 20 POs from generated gold records via reportlab, "
        "writing <name>.pdf, <name>.acl.json, and gold/extraction/<name>.json"
    )


if __name__ == "__main__":
    main()

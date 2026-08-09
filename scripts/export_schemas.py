"""Generate schemas/*.json from the Pydantic Raw models.

One direction only: Pydantic is the source of truth, JSON Schema is derived.
Two hand-maintained definitions drift, and the drift shows up as extraction
failures that look like model problems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isc.models.records import registry
import isc.models.records.invoice  # noqa: F401  (registration side effect)
import isc.models.records.purchase_order  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("schemas"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for doc_type, cls in registry.all().items():
        path = args.out / f"{doc_type}.schema.json"
        path.write_text(json.dumps(cls.raw_model.model_json_schema(), indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

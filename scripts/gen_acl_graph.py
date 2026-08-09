"""Synthetic identity graph: users, groups, sites, and the adversarial pairs.

The important output is not the users — it is `adversarial_pairs.json`: pairs of
principals where one can see a document and the other cannot. Those pairs drive
tests/adversarial/acl and the `restricted` class in the retrieval gold set.
A permission model with no adversarial pairs is untested by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = {
    "buyers-apac": ["u_alice", "u_ben"],
    "buyers-emea": ["u_chen"],
    "quality-sg": ["u_dara", "u_alice"],
    "engineering": ["u_ewan"],
    "contractors": ["u_frank"],          # deliberately low-privilege
    "trade-compliance": ["u_gita"],
}

SITES = ["site_sg01", "site_us42", "site_de07"]

USERS = {
    "u_alice": {"groups": ["buyers-apac", "quality-sg"], "sites": ["site_sg01"],
                "clearance": "confidential", "jurisdictions": ["SG"]},
    "u_ben":   {"groups": ["buyers-apac"], "sites": ["site_sg01"],
                "clearance": "internal", "jurisdictions": []},
    "u_chen":  {"groups": ["buyers-emea"], "sites": ["site_de07"],
                "clearance": "confidential", "jurisdictions": ["DE"]},
    "u_dara":  {"groups": ["quality-sg"], "sites": ["site_sg01"],
                "clearance": "internal", "jurisdictions": []},
    "u_ewan":  {"groups": ["engineering"], "sites": ["site_us42"],
                "clearance": "confidential", "jurisdictions": ["US"]},
    "u_frank": {"groups": ["contractors"], "sites": [],
                "clearance": "public", "jurisdictions": []},
    "u_gita":  {"groups": ["trade-compliance"], "sites": ["site_us42"],
                "clearance": "export_controlled", "jurisdictions": ["US", "SG"]},
}

# Each pair declares the doc types it needs present in the corpus. A pair whose
# doc types are absent is reported as PENDING, never as passing: a pair that
# cannot fire is zero coverage, and letting it read as green is how a permission
# suite ends up testing less than it appears to.
ADVERSARIAL_PAIRS = [
    {"can_see": "u_alice", "cannot_see": "u_frank", "requires": ["purchase_order"],
     "why": "contractor has no group grant on buyer documents"},
    {"can_see": "u_alice", "cannot_see": "u_chen", "requires": ["purchase_order"],
     "why": "regional split: APAC buyer vs EMEA buyer"},
    {"can_see": "u_gita", "cannot_see": "u_ewan", "requires": ["purchase_order"],
     "why": "export-controlled document requires a matching jurisdiction"},
    {"can_see": "u_alice", "cannot_see": "u_ben", "requires": ["purchase_order"],
     "why": "confidential clearance vs internal clearance"},
    {"can_see": "u_dara", "cannot_see": "u_ben", "requires": ["ncr"],
     "why": "quality NCR restricted to the quality group"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/acl"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "groups.json").write_text(json.dumps(GROUPS, indent=2))
    (args.out / "users.json").write_text(json.dumps(USERS, indent=2))
    (args.out / "sites.json").write_text(json.dumps(SITES, indent=2))
    (args.out / "adversarial_pairs.json").write_text(json.dumps(ADVERSARIAL_PAIRS, indent=2))
    print(f"wrote identity graph to {args.out}")


if __name__ == "__main__":
    main()

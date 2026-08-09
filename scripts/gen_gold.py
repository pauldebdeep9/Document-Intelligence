"""Build the retrieval gold set from the generated corpus.

Question classes and their proportions:
  answerable    ~70%  single-hop and multi-hop, with gold chunk ids
  unanswerable  ~15%  plausible questions the corpus cannot answer; abstention
                      is the only correct behaviour
  restricted    ~15%  answerable for one principal, invisible to another; the
                      pair comes from data/acl/adversarial_pairs.json

The unanswerable and restricted slices are the ones that catch real failures.
An eval set of only answerable questions rewards a system that never abstains
and never filters.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("data/gold"))
    args = ap.parse_args()
    raise NotImplementedError(
        "first slice: 20 questions derived from the generated gold records, "
        "3 unanswerable, 3 restricted"
    )


if __name__ == "__main__":
    main()

"""Guards against the editable install silently resolving `isc` to a different checkout.

This bit us for real: the editable install for isc-docint in the active environment was
pointing at an unrelated checkout (~/Documents/GenerativeAI/isc-docint), not this repository.
It went unnoticed for a while because that checkout's core modules happened to be
byte-identical to this repo's at the time — until one of them was edited here and the
change silently had no effect, because `import isc` was never resolving here at all.
"""

from pathlib import Path

import isc


def test_isc_resolves_inside_this_repositorys_src_directory() -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    resolved = Path(isc.__file__).resolve()

    assert resolved.is_relative_to(repo_src), (
        f"isc resolved to {resolved}, which is not under {repo_src} — the editable install "
        'for isc-docint is pointing at a different checkout. Fix: run pip install -e ".[dev]" '
        "from this repository's root."
    )

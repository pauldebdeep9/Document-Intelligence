"""Guards gen_corpus.py's reproducibility claim at the layer that actually
matters for content-addressing: PDF bytes, not extracted text.

reportlab stamps /CreationDate and /ModDate from the wall clock by default,
and folds that timestamp into the PDF's /ID -- so two --seed 2608 runs used to
produce semantically-identical but byte-different PDFs, which is a different
content hash and a different doc_id every time. A test that only compared
gold JSON or extracted text (as test_corpus_fidelity.py does) cannot catch
this: the gold and the extracted text WERE deterministic. Only comparing the
raw PDF bytes catches it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "gen_corpus.py"
MASTERS = ROOT / "data" / "masters"


def _generate(out: Path, gold: Path, seed: int, n: int = 3) -> None:
    subprocess.run(
        [sys.executable, str(GEN), "--n", str(n), "--types", "purchase_order",
         "--out", str(out), "--gold", str(gold), "--masters", str(MASTERS),
         "--seed", str(seed)],
        check=True, capture_output=True, text=True,
    )


def test_same_seed_produces_byte_identical_pdfs(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    _generate(out_a, tmp_path / "gold_a", seed=2608)
    _generate(out_b, tmp_path / "gold_b", seed=2608)

    pdfs = sorted(p.name for p in out_a.glob("*.pdf"))
    assert pdfs, "no PDFs generated"
    for name in pdfs:
        a = (out_a / name).read_bytes()
        b = (out_b / name).read_bytes()
        assert a == b, f"{name}: byte-identical PDF expected for the same seed"


def test_different_seeds_produce_different_pdfs(tmp_path):
    """The fix must not collapse to one universal constant -- that would pass
    the test above for the wrong reason and silently make every seed the
    same corpus."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    _generate(out_a, tmp_path / "gold_a", seed=2608, n=1)
    _generate(out_b, tmp_path / "gold_b", seed=9999, n=1)

    a = (out_a / "po_000.pdf").read_bytes()
    b = (out_b / "po_000.pdf").read_bytes()
    assert a != b

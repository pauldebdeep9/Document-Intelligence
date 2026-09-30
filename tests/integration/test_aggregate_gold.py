"""The records path against the real corpus and the P1-08 gold, offline.

Builds a slice into a temp dir with no network: real ingest -> parse ->
extract-wrap -> chunk -> ACL, with two stand-ins --

  * the extraction model is scripted to return each document's GOLD raw
    extraction, so every confidence, span, master-data check and routing
    decision is still computed by the real extractor;
  * the planner is scripted to return the plan each question should get.

So this measures execute + render + ACL + citations against gold, as each
question's own gold principal. It does NOT measure the planner: whether a
live model produces these plans (and says "none" to every other subtype) is
the live retrieval eval's job, with aggregate.enabled=true.

The harness proves offline-ness by sabotaging the provider registry, not by
convention.

Skipped when the corpus has not been generated (`make corpus`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import isc.llm.registry as registry
from isc.aggregate.answerer import RecordAnswerer
from isc.common.confidence import Thresholds
from isc.common.config import Settings
from isc.common.tracing import start_run
from isc.eval.retrieval import answer_contains_gold
from isc.extract.pipeline import run as run_extract
from isc.index.pipeline import run as run_index
from isc.ingest.pipeline import run as run_ingest
from isc.llm.ports import LLMResult
from isc.models.acl import AclSet, load_principals
from isc.models.document import DocType
from isc.parse.pipeline import run as run_parse
from isc.storage.local_blob import LocalBlobStore
from isc.storage.local_vector import LocalVectorStore
from isc.storage.sqlite_docstore import SqliteDocStore
from tests.aggregate_world import PlanChat, plan

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
GOLD_X = ROOT / "data" / "gold" / "extraction"
GOLD_Q = ROOT / "data" / "gold" / "retrieval" / "questions.json"

pytestmark = pytest.mark.skipif(
    not list(SYN.glob("*.pdf")), reason="corpus not generated; run `make corpus`")

# The plan each gold aggregate question should produce. Supplier strings are
# copied from the question text, exactly as the planner prompt requires.
PLANS = {
    "q_cd_01": plan("total_spend", "Omron Electronics Asia", currency="SGD"),
    "q_cd_02": plan("total_spend", "Keyence Singapore Pte Ltd", currency="SGD"),
    "q_cd_03": plan("total_spend", "Keyence Singapore Pte Ltd", currency="USD"),
    "q_cd_04": plan("total_spend", "Kestrel Industrial AG", currency="USD"),
    "q_cd_05": plan("part_prices", part="PLC-1756-L83"),
    "q_cd_06": plan("part_prices", part="TRM-BLK-2P5"),
    "q_cd_07": plan("part_prices", part="ENC-INC-1024"),
    "q_cd_08": plan("part_prices", part="TRM-BLK-2P5"),
    "q_am_04": plan("total_spend", "Kestrel Industrial"),
}


def _gold_raw() -> dict[str, dict]:
    return {json.loads(p.read_text())["document"]: json.loads(p.read_text())["raw"]
            for p in GOLD_X.glob("po_*.json")}


class _GoldExtractor:
    """Returns the gold raw extraction for whichever PO number is in the
    prompt, with a confident logprob -- 'perfect model output'."""

    def __init__(self) -> None:
        self._raw = list(_gold_raw().values())

    def complete(self, messages, *, schema=None, temperature=None, max_tokens=None):
        text = messages[-1].content
        for raw in self._raw:
            if raw["po_number"] and raw["po_number"] in text:
                return LLMResult(text=json.dumps(_as_model_output(raw)), model="gold",
                                 mean_logprob=-0.005)
        raise RuntimeError("no gold PO number in document text")


def _as_model_output(raw: dict) -> dict:
    def num(v):
        return None if v is None else float(str(v).replace(",", ""))
    out = dict(raw, total_amount=num(raw.get("total_amount")))
    out["lines"] = [dict(ln, **{k: num(ln.get(k))
                                for k in ("quantity", "unit_price", "extended_price")})
                    for ln in raw["lines"]]
    return out


class _HashEmbedder:
    dimensions = 64

    def embed(self, texts):
        import hashlib
        out = []
        for t in texts:
            v = [0.0] * self.dimensions
            for tok in t.lower().split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dimensions] += 1.0
            out.append(v)
        return out


def _no_network(*_args, **_kwargs):
    raise RuntimeError("network model used in the offline gold harness")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    # Module scope cannot use the function-scoped monkeypatch, so this one is
    # created here and undone at teardown; it stays in force for every test.
    # It also clears ISC_* itself: conftest's per-test clearing runs AFTER a
    # module fixture is built, and the Settings made here is used throughout.
    mp = pytest.MonkeyPatch()
    mp.delenv("OPENAI_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("ISC_"):
            mp.delenv(name)
    mp.setattr(registry, "get_chat_model", _no_network)
    mp.setattr(registry, "get_embedding_model", _no_network)
    try:
        out = tmp_path_factory.mktemp("aggregate_slice")
        s = Settings()
        run = start_run(out / "runs")
        blobs, docs = LocalBlobStore(out / "blobs"), SqliteDocStore(out / "docstore.sqlite")
        run_ingest(SYN, blobs, docs)
        assert not run_parse(run, blobs, docs).failed
        th = Thresholds(auto_accept=s.thresholds.auto_accept, review=s.thresholds.review,
                        reject=s.thresholds.reject)
        assert not run_extract(run, docs, _GoldExtractor(), DocType.PURCHASE_ORDER, th,
                               s.paths.data / "masters").failed
        store = LocalVectorStore(out / "store.pkl")
        assert not run_index(run, docs, _HashEmbedder(), store, s).failed
        gold = {q["id"]: q for q in json.loads(GOLD_Q.read_text())["questions"]}
        yield docs, store, s, gold, load_principals(s.paths.data / "acl")
    finally:
        mp.undo()


@pytest.mark.parametrize("qid", sorted(PLANS))
def test_gold_aggregate_question(world, qid):
    docs, store, s, gold, users = world
    q = gold[qid]
    principal = users[q["principal"]]
    ans = RecordAnswerer(PlanChat(PLANS[qid]), docs, store, s).try_answer(q["text"], principal)

    assert ans is not None and not ans.abstained, ans.text
    assert ans.route == "records"
    assert answer_contains_gold(q["gold_answer"], ans.text), ans.text
    assert all(principal.may_read(sc.chunk.acl) for sc in ans.supporting)
    assert ans.citations


@pytest.mark.parametrize("qid, part", [("q_cd_05", "PLC-1756-L83"), ("q_cd_06", "TRM-BLK-2P5"),
                                       ("q_cd_07", "ENC-INC-1024"), ("q_cd_08", "TRM-BLK-2P5")])
def test_part_prices_are_complete_not_just_the_gold_pair(world, qid, part):
    """Gold lists one pair of lines per part-price question, and
    answer_contains_gold() only checks that pair is present -- it cannot tell
    a complete answer from a two-line one. The complete answer is every
    priced line of that part the principal can read, recomputed here from
    the gold extraction files and the ACL sidecars independently of the
    code under test."""
    docs, store, s, gold, users = world
    q = gold[qid]
    principal = users[q["principal"]]
    expected = set()
    for name, raw in _gold_raw().items():
        acl_raw = json.loads((SYN / f"{name}.acl.json").read_text())
        acl = AclSet(allow_terms=frozenset(acl_raw["allow_terms"]),
                     deny_terms=frozenset(acl_raw["deny_terms"]),
                     sensitivity=acl_raw["sensitivity"],
                     jurisdictions=frozenset(acl_raw["jurisdictions"]))
        if principal.may_read(acl):
            expected |= {(raw["po_number"], ln["line_number"]) for ln in raw["lines"]
                         if ln["part_number"] == part and ln["unit_price"]}
    ans = RecordAnswerer(PlanChat(PLANS[qid]), docs, store, s).try_answer(q["text"], principal)
    listed = {(po, int(ln)) for po, ln in
              re.findall(r"PO (\d+) line (\d+): unit price", ans.text)}
    assert listed == expected
    assert len(expected) >= len(q["gold_answer"])

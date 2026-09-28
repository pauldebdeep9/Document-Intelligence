"""Run the planner routing eval live: one planner call per expected-plan case.

  python scripts/eval_planner.py            # make planner-eval

Writes runs/<run_id>/eval/planner.json (meta + every outcome) and planner.md,
and prints the markdown. Refuses to run with the response cache on unless
--allow-cache is given: a cached planner replays an earlier run's answers,
which is not a measurement of the prompt and model under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import isc.llm.registry as registry
from isc.aggregate.planner import PROMPT
from isc.common.config import get_settings
from isc.common.tracing import start_run
from isc.eval.planner import evaluate, load_cases, outcomes_as_json, render_markdown, summarise

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "planner" / "cases.json"
HANDWRITTEN = ROOT / "data" / "gold" / "planner" / "handwritten.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--handwritten", type=Path, default=HANDWRITTEN)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-cache", action="store_true",
                    help="run even though the response cache is enabled (not a measurement)")
    args = ap.parse_args(argv)

    s = get_settings()
    if s.cache.enabled and not args.allow_cache:
        print("eval_planner: the response cache is enabled, so planner calls could be "
              "replayed rather than measured. Set ISC_CACHE__ENABLED=false (or pass "
              "--allow-cache to run anyway).", file=sys.stderr)
        return 2

    cases = load_cases(args.gold) + load_cases(args.handwritten)
    chat = registry.get_chat_model()
    run = start_run(s.paths.runs, args.run_id)
    outcomes = evaluate(chat, cases, s.paths.data / "masters")
    summary = summarise(outcomes)

    prompt_path = s.paths.prompts / PROMPT
    meta = {
        "run_id": run.run_id,
        "model": s.llm.chat_deployment or s.llm.chat_model,
        "provider": s.llm.provider,
        "prompt": {"path": str(prompt_path.relative_to(ROOT)), "sha256": _sha256(prompt_path)},
        "cases": {str(p.relative_to(ROOT)): _sha256(p) for p in (args.gold, args.handwritten)},
        "cache_enabled": s.cache.enabled,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    out = run.artifact_dir("eval")
    (out / "planner.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "outcomes": outcomes_as_json(outcomes)},
        indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(summary, meta)
    (out / "planner.md").write_text(markdown)
    run.summarise()
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

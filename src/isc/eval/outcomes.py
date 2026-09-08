"""Persist eval/retrieval.py's per-outcome data so a corrected metric
definition can be re-scored without a live re-run (EV-01).

report.write() (report.py) only ever emits aggregates -- report.json/md have
no per-outcome data, so RetrievalReport.outcomes dies with the process that
produced it. This module is the missing round trip: write() dumps every
QuestionOutcome (and RetrievalEvalResult.failed) to
runs/<run_id>/eval/{outcomes.jsonl,failed.jsonl}; load() reconstructs a full
RetrievalEvalResult from those two files with no live model, index, or
docstore access -- see cli.py's `isc eval --rescore-from`.

Two files, not one, and both written together every time: outcomes.jsonl is
one JSON object per line, one line per QuestionOutcome, in the exact order
they appear in RetrievalReport.outcomes (never reordered or sorted -- see
_outcome_to_record()'s docstring on why order is load-bearing). failed.jsonl
is a one-line header (carrying schema_version and a count) followed by one
line per RetrievalEvalResult.failed entry -- the header exists specifically
so a run with zero failures still produces a version-checkable file; without
it, an empty failed.jsonl would be indistinguishable from a missing one
except by stat(), and a failed list that silently reads back as empty is the
exact hazard this module exists to avoid. load() requires both files to be
present and raises rather than treating a missing failed.jsonl as "nothing
failed".

`failed` is persisted for replay fidelity, not because it currently affects
any score: every RetrievalReport metric denominator is len(outcomes) or a
filtered subset of it, never a count of input questions, and today `failed`
is a console-only warning (cli.py) that never reaches report.json/md. Do not
assume a future reader that it already feeds scoring -- it doesn't, yet.

citations_valid round-trips too, for shape fidelity, though eval/retrieval.py's
run() never assigns it (always its dataclass default True) and no
RetrievalReport method reads it -- it is currently an inert field, not
something this module gives meaning to.

EVALUATOR-ONLY ARTIFACT -- DO NOT SHARE. outcomes.jsonl is a cross-principal
superset by construction: one file holds every principal's retrieved chunk
ids, citations, and answer text for a gold question, including the outcomes
that exist BECAUSE a principal must be denied (restricted's principal_b,
every no_reader principal). No single principal in the identity graph could
legitimately read this file's own contents. runs/ is gitignored, so this is
not a live leak today -- but this is not a redaction boundary, and none is
added here. Never attach this file to a review, share it, or serve it as-is.

Bump OUTCOMES_SCHEMA_VERSION whenever a field is added, removed, renamed, or
reinterpreted on QuestionOutcome or the failed-record shape -- load() checks
it on every record and refuses to silently misread an older or newer shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from isc.common.errors import IscError
from isc.eval.retrieval import QuestionOutcome, RetrievalEvalResult, RetrievalReport

OUTCOMES_SCHEMA_VERSION = 1

OUTCOMES_FILENAME = "outcomes.jsonl"
FAILED_FILENAME = "failed.jsonl"


class OutcomesSchemaMismatch(IscError):
    """A persisted outcomes/failed record's schema_version does not match
    this reader's OUTCOMES_SCHEMA_VERSION (or, for failed.jsonl, is missing
    entirely -- an empty file has no header to read a version from at all).

    Never caught and downgraded: silently reading a mismatched record shape
    as if it matched the current one would misassign or drop fields, and
    that corruption would only surface later as a metric that quietly
    disagrees with the original run for no visible reason -- exactly the
    kind of silent-and-wrong result this whole module exists to prevent.
    """

    def __init__(self, path: Path, line_no: int, expected: int, found: object) -> None:
        super().__init__(
            f"{path}:{line_no}: schema_version {found!r} does not match this reader's "
            f"OUTCOMES_SCHEMA_VERSION={expected} -- regenerate this file with the current "
            "isc.eval.outcomes.write(), or read it with a version of this module built for "
            "that schema_version."
        )
        self.path = path
        self.line_no = line_no
        self.expected = expected
        self.found = found


def _outcome_to_record(o: QuestionOutcome) -> dict:
    """gold_ids is the only non-JSON-native field on QuestionOutcome (a
    set[str]) -- sorted here so the file is stable/diffable across runs
    regardless of set iteration order; set membership doesn't care about
    element order, so this is lossless, not lossy. retrieved_ids and
    citations are NOT sorted -- their order is semantically load-bearing
    (mrr()/ndcg_at_k() in eval/metrics.py score position, not just
    membership), so they round-trip exactly as retrieved.
    """
    return {
        "schema_version": OUTCOMES_SCHEMA_VERSION,
        "question_id": o.question_id,
        "question_class": o.question_class,
        "subtype": o.subtype,
        "principal_id": o.principal_id,
        "is_gold_principal": o.is_gold_principal,
        "retrieved_ids": list(o.retrieved_ids),
        "gold_ids": sorted(o.gold_ids),
        "abstained": o.abstained,
        "abstention_reason": o.abstention_reason,
        "answer_text": o.answer_text,
        "citations": list(o.citations),
        "answer_correct": o.answer_correct,
        "abstention_correct": o.abstention_correct,
        "citations_valid": o.citations_valid,
        "leaked_chunk_ids": list(o.leaked_chunk_ids),
    }


def _record_to_outcome(path: Path, line_no: int, record: dict) -> QuestionOutcome:
    found = record.get("schema_version")
    if found != OUTCOMES_SCHEMA_VERSION:
        raise OutcomesSchemaMismatch(path, line_no, OUTCOMES_SCHEMA_VERSION, found)
    return QuestionOutcome(
        question_id=record["question_id"],
        question_class=record["question_class"],
        subtype=record["subtype"],
        principal_id=record["principal_id"],
        is_gold_principal=record["is_gold_principal"],
        retrieved_ids=list(record["retrieved_ids"]),
        gold_ids=set(record["gold_ids"]),
        abstained=record["abstained"],
        abstention_reason=record["abstention_reason"],
        answer_text=record["answer_text"],
        citations=list(record["citations"]),
        answer_correct=record["answer_correct"],
        abstention_correct=record["abstention_correct"],
        citations_valid=record["citations_valid"],
        leaked_chunk_ids=list(record["leaked_chunk_ids"]),
    )


def write(out_dir: Path, result: RetrievalEvalResult) -> tuple[Path, Path]:
    """Emit outcomes.jsonl and failed.jsonl into out_dir (created if
    missing), overwriting any prior contents. Always writes both files, even
    when result.failed is empty, so failed.jsonl's header line is always
    present and load() never has to guess whether an empty file means
    "zero failures" or "never written"."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = out_dir / OUTCOMES_FILENAME
    failed_path = out_dir / FAILED_FILENAME

    with outcomes_path.open("w") as fh:
        for o in result.report.outcomes:
            fh.write(json.dumps(_outcome_to_record(o)) + "\n")

    with failed_path.open("w") as fh:
        fh.write(json.dumps({
            "schema_version": OUTCOMES_SCHEMA_VERSION,
            "count": len(result.failed),
        }) + "\n")
        for question_id, principal_id, reason in result.failed:
            fh.write(json.dumps({
                "question_id": question_id,
                "principal_id": principal_id,
                "reason": reason,
            }) + "\n")

    return outcomes_path, failed_path


def load(outcomes_path: Path) -> RetrievalEvalResult:
    """Reconstruct a full RetrievalEvalResult (report.outcomes AND failed)
    from outcomes_path and its sibling failed.jsonl -- no live model, index,
    or docstore access. Both files must exist; a missing failed.jsonl is
    rejected loudly rather than read as "nothing failed", since that is
    exactly the silent-shrinkage failure mode persistence exists to avoid.
    """
    if not outcomes_path.exists():
        raise FileNotFoundError(f"no outcomes file at {outcomes_path}")
    failed_path = outcomes_path.parent / FAILED_FILENAME
    if not failed_path.exists():
        raise FileNotFoundError(
            f"no {FAILED_FILENAME} alongside {outcomes_path} -- write() always writes both "
            "files together, even when there are zero failures. A missing failed.jsonl means "
            "this directory was not produced by isc.eval.outcomes.write(), or was tampered "
            "with, either of which must be surfaced rather than silently read as zero "
            "failures."
        )

    outcomes: list[QuestionOutcome] = []
    with outcomes_path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            outcomes.append(_record_to_outcome(outcomes_path, line_no, json.loads(line)))

    with failed_path.open() as fh:
        lines = fh.readlines()
    if not lines:
        # write() always emits a header line even for zero failures -- a
        # truly empty file was truncated or hand-edited, not produced by
        # write(). found=None: there is no schema_version to read at all.
        raise OutcomesSchemaMismatch(failed_path, 1, OUTCOMES_SCHEMA_VERSION, None)
    header = json.loads(lines[0])
    found = header.get("schema_version")
    if found != OUTCOMES_SCHEMA_VERSION:
        raise OutcomesSchemaMismatch(failed_path, 1, OUTCOMES_SCHEMA_VERSION, found)
    body_lines = lines[1:]
    expected_count = header.get("count")
    if expected_count != len(body_lines):
        raise ValueError(
            f"{failed_path}: header declares count={expected_count} but {len(body_lines)} "
            "failed-record line(s) follow -- the file is truncated or was hand-edited"
        )

    failed: list[tuple[str, str, str]] = []
    for line in body_lines:
        record = json.loads(line)
        failed.append((record["question_id"], record["principal_id"], record["reason"]))

    return RetrievalEvalResult(report=RetrievalReport(outcomes=outcomes), failed=failed)

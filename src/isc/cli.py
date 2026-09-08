"""Stage-per-verb CLI. Every stage reads and writes files under runs/<run_id>/.

Slower than an in-memory pipeline, and deliberately so: each stage is
independently re-runnable and inspectable, which is what keeps the eval loop
cheap and makes a failure attributable to one stage rather than to "the pipeline".
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from isc.common.config import get_settings
from isc.common.logging import setup
from isc.common.tracing import start_run
from isc.models.acl import Principal, Sensitivity

app = typer.Typer(add_completion=False, help="ISC Document Intelligence")
console = Console()


@app.callback()
def _init(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    setup("DEBUG" if verbose else "INFO")


@app.command()
def ingest(source: Path = typer.Option(..., "--source"),
           run_id: str | None = typer.Option(None)) -> None:
    """Source -> blob store + document stubs, deduped by content hash."""
    s = get_settings()
    run = start_run(s.paths.runs, run_id)
    from isc.ingest.pipeline import run as do_ingest
    from isc.storage.local_blob import LocalBlobStore
    from isc.storage.sqlite_docstore import SqliteDocStore

    ids = do_ingest(source, LocalBlobStore(s.paths.data / "blobs"),
                    SqliteDocStore(s.paths.data / "docstore.sqlite"))
    console.print(f"[green]ingested[/] {len(ids)} documents  run={run.run_id}")
    run.summarise()


@app.command()
def parse(run_id: str | None = typer.Option(None)) -> None:
    """Blobs -> Documents with blocks, tables and parser provenance."""
    s = get_settings()
    run = start_run(s.paths.runs, run_id)
    from isc.parse.pipeline import run as do_parse
    from isc.storage.local_blob import LocalBlobStore
    from isc.storage.sqlite_docstore import SqliteDocStore

    result = do_parse(run, LocalBlobStore(s.paths.data / "blobs"),
                      SqliteDocStore(s.paths.data / "docstore.sqlite"))
    console.print(f"[green]parsed[/] {len(result.parsed)} documents  run={run.run_id}")
    if result.failed:
        console.print(f"[red]failed[/] {len(result.failed)} documents:")
        for doc_id, reason in result.failed:
            console.print(f"  {doc_id}: {reason}")
    run.summarise()
    if result.failed:
        raise typer.Exit(code=1)


@app.command()
def extract(doc_type: str = typer.Option(..., "--doc-type"),
            run_id: str | None = typer.Option(None)) -> None:
    """Documents -> ExtractionRecords, with low-confidence fields queued for review."""
    from isc.common.confidence import Thresholds
    from isc.extract.pipeline import run as do_extract
    from isc.llm.registry import get_chat_model
    from isc.models.document import DocType
    from isc.storage.sqlite_docstore import SqliteDocStore

    try:
        dt = DocType(doc_type)
    except ValueError:
        console.print(f"[red]unknown doc_type[/] {doc_type!r}")
        raise typer.Exit(code=1) from None

    s = get_settings()
    run = start_run(s.paths.runs, run_id)
    thresholds = Thresholds(auto_accept=s.thresholds.auto_accept,
                             review=s.thresholds.review, reject=s.thresholds.reject)
    docs = SqliteDocStore(s.paths.data / "docstore.sqlite")

    result = do_extract(run, docs, get_chat_model(), dt, thresholds, s.paths.data / "masters")
    console.print(f"[green]extracted[/] {len(result.extracted)} documents  run={run.run_id}")
    if result.failed:
        console.print(f"[red]failed[/] {len(result.failed)} documents:")
        for doc_id, reason in result.failed:
            console.print(f"  {doc_id}: {reason}")
    run.summarise()
    if result.failed:
        raise typer.Exit(code=1)


@app.command()
def index(run_id: str | None = typer.Option(None)) -> None:
    """Documents -> chunks -> embeddings, with ACL projected onto every chunk."""
    from isc.index.pipeline import run as do_index
    from isc.llm.registry import get_embedding_model
    from isc.storage.local_vector import LocalVectorStore
    from isc.storage.sqlite_docstore import SqliteDocStore

    s = get_settings()
    run = start_run(s.paths.runs, run_id)
    docs = SqliteDocStore(s.paths.data / "docstore.sqlite")
    store = LocalVectorStore(s.paths.data / "vector_store.pkl")

    result = do_index(run, docs, get_embedding_model(), store, s)
    console.print(f"[green]indexed[/] {len(result.indexed)} documents "
                  f"({result.chunks} chunks)  run={run.run_id}")
    if result.failed:
        console.print(f"[red]failed[/] {len(result.failed)} documents:")
        for doc_id, reason in result.failed:
            console.print(f"  {doc_id}: {reason}")
    run.summarise()
    if result.failed:
        raise typer.Exit(code=1)


def _resolve_principal(as_user: str, groups: str, acl_dir: Path) -> Principal:
    """Prefer the real ACL graph (data/acl/users.json) over the bare
    --as/--groups flags: a named user like u_alice carries site, clearance
    and jurisdiction the flags alone cannot express, and without them she is
    indistinguishable from u_ben against every restricted document in this
    corpus -- clearance is the ONLY thing that differs between them. Falls
    back to flags-only construction for an id not in the graph, so asking as
    an ad hoc/synthetic principal still works for testing."""
    users_path = acl_dir / "users.json"
    if users_path.exists():
        users = json.loads(users_path.read_text())
        if as_user in users:
            u = users[as_user]
            return Principal(
                id=as_user, group_ids=frozenset(u["groups"]), site_ids=frozenset(u["sites"]),
                clearance=Sensitivity(u["clearance"]), jurisdictions=frozenset(u["jurisdictions"]),
            )
    return Principal(id=as_user, group_ids=frozenset(g for g in groups.split(",") if g))


@app.command()
def ask(question: str, as_user: str = typer.Option(..., "--as"),
        groups: str = typer.Option("", "--groups", help="comma separated")) -> None:
    """Ask a question AS a specific principal. There is no unauthenticated mode."""
    from isc.answer.orchestrator import AnswerOrchestrator
    from isc.llm.registry import get_chat_model, get_embedding_model
    from isc.retrieve.retriever import Retriever
    from isc.storage.local_vector import LocalVectorStore

    s = get_settings()
    run = start_run(s.paths.runs)
    principal = _resolve_principal(as_user, groups, s.paths.data / "acl")

    store = LocalVectorStore(s.paths.data / "vector_store.pkl")
    chat = get_chat_model()
    retriever = Retriever(store, get_embedding_model(), chat, s)
    orchestrator = AnswerOrchestrator(retriever, chat, s)

    answer = orchestrator.ask(question, principal)

    if answer.abstained:
        console.print(f"[yellow]abstained[/] ({answer.abstention_reason.value}): {answer.text}")
    else:
        console.print(answer.text)
        console.print()
        t = Table("citation", "label")
        for c in answer.citations:
            t.add_row(c.chunk_id, c.label)
        console.print(t)
    run.summarise()


@app.command()
def review(limit: int = 20) -> None:
    """Show the HITL queue, weakest confidence first."""
    from isc.storage.sqlite_docstore import SqliteDocStore
    s = get_settings()
    rows = SqliteDocStore(s.paths.data / "docstore.sqlite").open_reviews(limit)
    t = Table("document", "field", "confidence", "weakest signal")
    for r in rows:
        t.add_row(r["document_id"], r["field_name"],
                  f"{r['confidence']:.3f}", r["weakest_signal"] or "")
    console.print(t)


@app.command()
def eval(harness: str = typer.Option("both", help="extraction|retrieval|both"),
         run_id: str = typer.Option(..., help="existing run with an extract/ stage"),
         rescore_from: Path | None = typer.Option(
             None, "--rescore-from",
             help="Path to a prior run's eval/outcomes.jsonl. Re-scores retrieval from that "
                  "file instead of making live embedding/chat calls -- no index, docstore, or "
                  "provider access. Requires --harness retrieval (extraction, if you also want "
                  "it, is already offline -- run it separately into its own report rather than "
                  "mixing it into a rescore, so the rescore's contract stays 'one outcomes file "
                  "in, one retrieval-only report out'). --run-id must be a run whose "
                  "eval/report.json does not already exist -- a rescore must never overwrite "
                  "the very report it will be compared against.")) -> None:
    """Score a completed extract run against gold, and/or run the P1-08
    retrieval gold set live against the current index, and write
    report.json + report.md.

    Extraction: no LLM calls -- reads runs/<run_id>/extract/*.json (raw +
    record, written by `isc extract`) and data/gold/extraction/*.json side
    by side.

    Retrieval: real embedding + chat calls, one per gold question (each
    asked as its own gold principal -- see eval/retrieval.py's run()).
    Refuses to run if either fingerprint in the gold's provenance disagrees
    with the live index or corpus (check_provenance()) -- see docs/adr/0007
    and docs/adr/0008. A single ACL leak fails the run: non-zero exit,
    regardless of every other metric. Every retrieval outcome is written to
    runs/<run_id>/eval/{outcomes.jsonl,failed.jsonl} (see eval/outcomes.py)
    so a later --rescore-from can re-derive report.json/md without a live
    re-run.

    --rescore-from: re-scores from a previously written outcomes.jsonl
    instead of running retrieval live. See eval/outcomes.py's module
    docstring -- that file is evaluator-only, never share or serve it as-is.
    """
    from isc.eval.report import write
    from isc.storage.sqlite_docstore import SqliteDocStore

    if rescore_from is not None and harness != "retrieval":
        console.print("[red]error[/] --rescore-from requires --harness retrieval "
                       f"(got --harness {harness!r})")
        raise typer.Exit(code=1)

    s = get_settings()
    run = start_run(s.paths.runs, run_id)
    docs = SqliteDocStore(s.paths.data / "docstore.sqlite")

    if rescore_from is not None:
        report_path = run.artifact_dir("eval") / "report.json"
        if report_path.exists():
            console.print(
                f"[red]error[/] refusing to rescore into run {run.run_id!r}: {report_path} "
                "already exists. Pass a different --run-id for the rescore output so the "
                "report it will be compared against is not overwritten."
            )
            raise typer.Exit(code=1)

    extraction_report = None
    if harness in {"extraction", "both"}:
        from isc.eval.pipeline import run as run_extraction_eval

        result = run_extraction_eval(run, docs, s.paths.data / "gold" / "extraction")
        extraction_report = result.report
        console.print(f"[green]scored[/] {len(result.scored)} documents  run={run.run_id}")
        if result.skipped:
            console.print(f"[yellow]skipped[/] {len(result.skipped)} artifacts:")
            for name, reason in result.skipped:
                console.print(f"  {name}: {reason}")

    retrieval_report = None
    if harness in {"retrieval", "both"}:
        from isc.eval import outcomes as eval_outcomes
        from isc.eval.retrieval import evaluate_acl_gate

        if rescore_from is not None:
            retrieval_result = eval_outcomes.load(rescore_from)
            retrieval_report = retrieval_result.report
            console.print(f"[green]loaded[/] {len(retrieval_report.outcomes)} retrieval "
                          f"outcomes from {rescore_from}  run={run.run_id}")
        else:
            from isc.answer.orchestrator import AnswerOrchestrator
            from isc.eval.retrieval import check_provenance
            from isc.eval.retrieval import run as run_retrieval_eval
            from isc.llm.registry import get_chat_model, get_embedding_model
            from isc.models.acl import load_principals
            from isc.retrieve.retriever import Retriever
            from isc.storage.local_vector import LocalVectorStore

            gold = json.loads(
                (s.paths.data / "gold" / "retrieval" / "questions.json").read_text())
            store = LocalVectorStore(s.paths.data / "vector_store.pkl")
            check_provenance(gold["provenance"], s, store, docs)  # raises loudly on mismatch

            chat = get_chat_model()
            orchestrator = AnswerOrchestrator(
                Retriever(store, get_embedding_model(), chat, s), chat, s)
            users = load_principals(s.paths.data / "acl")

            retrieval_result = run_retrieval_eval(gold["questions"], users, orchestrator)
            retrieval_report = retrieval_result.report
            console.print(f"[green]scored[/] {len(retrieval_report.outcomes)} retrieval "
                          f"outcomes  run={run.run_id}")
            outcomes_path, failed_path = eval_outcomes.write(
                run.artifact_dir("eval"), retrieval_result)
            console.print(f"[green]outcomes written[/] {outcomes_path}  {failed_path}")

        if retrieval_result.failed:
            console.print(f"[yellow]failed[/] {len(retrieval_result.failed)} (question, principal):")
            for qid, principal_id, reason in retrieval_result.failed:
                console.print(f"  {qid} as {principal_id}: {reason}")
        gate = evaluate_acl_gate(retrieval_report)
        if not gate.passed:
            console.print(f"[red]ACL LEAK[/] in {len(retrieval_report.leaks())} outcome(s): "
                           f"{[o.question_id for o in retrieval_report.leaks()]}")

    out = write(run.artifact_dir("eval"), extraction_report, retrieval_report,
                threshold=s.thresholds.auto_accept, review_threshold=s.thresholds.review)
    console.print(f"[green]report written[/] {out}")
    summary_path = run.summarise()

    # Slice summary: only when both harnesses just ran (`make slice`'s own
    # `isc eval --harness both`) is there enough context for one -- an
    # extraction-only or retrieval-only invocation has half the picture.
    # totals come from summary.json, not run.totals directly, so this
    # reflects EVERY stage that shared this run id (ingest/parse/extract/
    # index too -- see common/tracing.py's Run), not just this process's
    # own embedding+chat calls.
    if harness == "both":
        totals = json.loads(summary_path.read_text())["totals"]
        cost = totals.get("usd", 0.0)
        tokens = sum(v for k, v in totals.items() if k.startswith("tokens."))
        n_chunks = store.count() if retrieval_report is not None else 0
        n_questions = len(gold["questions"]) if retrieval_report is not None else 0
        console.print()
        console.print(f"[bold]Slice summary[/]  run={run.run_id}")
        console.print(f"  documents:  {len(docs.list_documents())}")
        console.print(f"  chunks:     {n_chunks}")
        console.print(f"  questions:  {n_questions}")
        if extraction_report is not None:
            rate = extraction_report.auto_accept_error_rate(s.thresholds.auto_accept)
            console.print(f"  extraction: auto-accept error rate {rate:.3%}")
        if retrieval_report is not None:
            acc = retrieval_report.answer_accuracy()
            console.print(
                f"  retrieval:  recall@8={retrieval_report.recall_at(8):.3f}  "
                f"mrr={retrieval_report.mean_mrr():.3f}  "
                f"answer_accuracy={acc['correct']}/{acc['n']} ({acc['accuracy']:.1%})  "
                f"passed={gate.passed}"
            )
        console.print(f"  cost:       ${cost:.4f}  ({int(tokens)} tokens)")

    if retrieval_report is not None and not gate.passed:
        raise typer.Exit(code=1)


@app.command(name="eval-diff")
def eval_diff(
    run_a: str = typer.Argument(..., help="run_id whose eval/outcomes.jsonl is the 'before' side"),
    run_b: str = typer.Argument(..., help="run_id whose eval/outcomes.jsonl is the 'after' side"),
) -> None:
    """Per-outcome diff between two runs (EV-04) -- see eval/diff.py's module
    docstring. Reads eval/outcomes.jsonl + eval/failed.jsonl for both runs
    (eval/outcomes.py's load()); never touches report.json, gold, the index,
    or any provider -- purely a comparison of already-persisted records.

    Exit code mirrors diff(1): 0 when identical, 1 when any difference is
    found (including gold_change -- it is a difference; the banner in the
    output carries what it means). citations_valid disagreeing between the
    two files, or a (question_id, principal_id) key repeated within one
    file, both raise uncaught -- the same refuse-rather-than-silently-misread
    discipline as check_provenance()'s GoldProvenanceMismatch above.
    """
    from isc.eval import diff as eval_diff_module
    from isc.eval import outcomes as eval_outcomes

    s = get_settings()
    result_a = eval_outcomes.load(s.paths.runs / run_a / "eval" / "outcomes.jsonl")
    result_b = eval_outcomes.load(s.paths.runs / run_b / "eval" / "outcomes.jsonl")

    diff_result = eval_diff_module.diff(result_a, result_b, label_a=run_a, label_b=run_b)
    console.print(eval_diff_module.render(diff_result))

    if not diff_result.is_identical:
        raise typer.Exit(code=1)


def _todo(stage: str) -> int:
    console.print(f"[yellow]{stage}[/] not implemented yet — see the first-slice list in README")
    return 1


if __name__ == "__main__":
    app()

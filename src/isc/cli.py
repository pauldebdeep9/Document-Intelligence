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
         run_id: str = typer.Option(..., help="existing run with an extract/ stage")) -> None:
    """Score a completed extract run against gold and write report.json + report.md.

    No LLM calls: reads runs/<run_id>/extract/*.json (raw + record, written by
    `isc extract`) and data/gold/extraction/*.json side by side.
    """
    from isc.eval.report import write
    from isc.storage.sqlite_docstore import SqliteDocStore

    s = get_settings()
    run = start_run(s.paths.runs, run_id)
    docs = SqliteDocStore(s.paths.data / "docstore.sqlite")

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

    if harness in {"retrieval", "both"}:
        console.print("[yellow]retrieval harness[/] not wired yet")

    out = write(run.artifact_dir("eval"), extraction_report, None,
                threshold=s.thresholds.auto_accept, review_threshold=s.thresholds.review)
    console.print(f"[green]report written[/] {out}")
    run.summarise()


def _todo(stage: str) -> int:
    console.print(f"[yellow]{stage}[/] not implemented yet — see the first-slice list in README")
    return 1


if __name__ == "__main__":
    app()

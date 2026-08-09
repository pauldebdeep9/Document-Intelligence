"""Stage-per-verb CLI. Every stage reads and writes files under runs/<run_id>/.

Slower than an in-memory pipeline, and deliberately so: each stage is
independently re-runnable and inspectable, which is what keeps the eval loop
cheap and makes a failure attributable to one stage rather than to "the pipeline".
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from isc.common.config import get_settings
from isc.common.logging import setup
from isc.common.tracing import start_run
from isc.models.acl import Principal

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
    raise typer.Exit(code=_todo("parse"))


@app.command()
def extract(doc_type: str = typer.Option(..., "--doc-type"),
            run_id: str | None = typer.Option(None)) -> None:
    """Documents -> ExtractionRecords, with low-confidence fields queued for review."""
    raise typer.Exit(code=_todo("extract"))


@app.command()
def index(run_id: str | None = typer.Option(None)) -> None:
    """Documents -> chunks -> embeddings, with ACL projected onto every chunk."""
    raise typer.Exit(code=_todo("index"))


@app.command()
def ask(question: str, as_user: str = typer.Option(..., "--as"),
        groups: str = typer.Option("", "--groups", help="comma separated")) -> None:
    """Ask a question AS a specific principal. There is no unauthenticated mode."""
    s = get_settings()
    run = start_run(s.paths.runs)
    principal = Principal(
        id=as_user,
        group_ids=frozenset(g for g in groups.split(",") if g),
    )
    from isc.answer.orchestrator import AnswerOrchestrator  # noqa: F401
    console.print(f"[yellow]not wired yet[/] — would ask as {principal.id}")
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
         run_id: str | None = typer.Option(None)) -> None:
    """Run the dual harness and write report.json + report.md."""
    raise typer.Exit(code=_todo("eval"))


def _todo(stage: str) -> int:
    console.print(f"[yellow]{stage}[/] not implemented yet — see the first-slice list in README")
    return 1


if __name__ == "__main__":
    app()

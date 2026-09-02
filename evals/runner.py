"""Executes the pipeline against a gold split and records raw per-item results.

This module computes NO metrics — it only runs the pipeline and writes down what happened.
Re-scoring a saved run (evals/report.py, using evals/metrics.py) is free; re-running this
module costs money and is non-deterministic. Do not merge the two, and this module does not
import evals.metrics.

Ingest is always corpus-wide: extracting pages, extracting the PurchaseOrder, chunking, and
embedding chunks happens exactly once per document per run, and every document's chunks are
pooled into `pooled_chunks`/`pooled_embeddings` regardless of retrieval mode.
isc.pipeline.process_document re-extracts and re-embeds per question, so it is not used here;
the isc.* primitives are composed directly instead.

What varies is which of that pool a given question is allowed to rank over, controlled by
`scoping`. `"scoped"` (the default) restricts each question's candidate set to its own gold
document's chunks, filtered out of the pool before ranking — never as a post-filter over
top_k_chunks's output, which would silently return fewer than k results for a reason invisible
in the record. `"pooled"` ranks every question against the full pool, undifferentiated by
document; this is the only configuration in which the near_duplicate questions (po-004 vs
po-005, near-identical text) test cross-document discrimination at all — scoping them would
make each trivially satisfiable by construction, not by retrieval quality. Both modes stay
runnable; comparing them is the point.
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from evals.corpus.generate import PDF_DIR
from evals.gold.schema import GoldSet, QuestionClass, validate_anchors
from evals.gold.split import Split
from isc.chunking import chunk_pages
from isc.llm import answer_question, embed_texts, extract_purchase_order
from isc.models import Chunk, PurchaseOrder
from isc.pdf import extract_pdf_pages
from isc.retrieval import top_k_chunks

RUNS_DIR = Path("runs")

_REPO_ROOT = Path(__file__).resolve().parents[1]

RetrievalScoping = Literal["scoped", "pooled"]


class RunConfig(BaseModel):
    """The frozen configuration a run was executed under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_model: str
    embedding_model: str
    chunk_size: int
    overlap: int
    k: int
    goldset_version: str
    split: Split
    document_count: int
    pooled_chunk_count: int
    scoping: RetrievalScoping


class DocumentRecord(BaseModel):
    """One document's ingest result: what was extracted, and how many chunks it produced."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    purchase_order: PurchaseOrder
    chunk_count: int


class ItemRecord(BaseModel):
    """One question's raw retrieval and answer result, or its error.

    The five retrieved_* lists are parallel and rank-ordered (index 0 is the top-ranked
    retrieved chunk). retrieved_page_numbers is stored alongside chunk_id/doc_id/text/score
    so a saved record can reconstruct full SourceEvidence objects later without guessing —
    SourceEvidence has five fields, not four, and re-scoring needs all of them.

    scope records what was actually searched for this question — its own doc_id in scoped
    mode, None in pooled mode — independent of retrieved_doc_ids, which records what came
    back. The two can differ (e.g. zero results); scope is never inferred from the other.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str
    doc_id: str
    question_class: QuestionClass
    scope: str | None
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_page_numbers: list[int]
    retrieved_texts: list[str]
    retrieved_scores: list[float]
    answer: str | None
    source_chunk_ids: list[str]
    error: str | None


class RunRecord(BaseModel):
    """A complete, re-scorable observation of one evaluation run.

    Unlike goldset.json (which is meant to be byte-reproducible and carries no timestamp), a
    RunRecord IS a dated observation of one specific execution — created_utc belongs here.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_utc: str
    git_commit_sha: str
    config: RunConfig
    documents: list[DocumentRecord]
    items: list[ItemRecord]


def _current_git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def preflight(
    goldset: GoldSet,
    chat_model: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    k: int,
    pdf_dir: Path = PDF_DIR,
    expected_goldset_version: str | None = None,
) -> None:
    """Validate the run configuration; raise ValueError before any client call is made.

    Checks, in order: if expected_goldset_version is given, the loaded goldset's version must
    match it exactly - a run scored against an unexpected gold set version is a config error,
    not a result, and this check exists so that mismatch is caught here rather than discovered
    later by a human reading RunConfig.goldset_version in a saved record. Passing None (the
    default) skips this check entirely - it is opt-in verification, not enforced by default,
    since preflight has no way to know what version a caller intended without being told.
    Then: model names are non-blank; k is at least 1; the goldset has at least one item;
    chunk_size/overlap are valid (reuses chunk_pages's own validation, called with an empty
    page list so it validates without touching real data); every document's PDF exists and
    parses, and every anchor resolves at this chunk_size/overlap (reuses
    evals.gold.schema.validate_anchors, which performs both checks together).

    This does not re-verify the TEST/HELD access guard — that already ran (or didn't) at the
    caller's layer via evals.gold.split.load_gold before goldset ever reached this function;
    preflight has no allow_test/allow_held/run_label params to re-enforce it with. It also
    does not check that every doc_id "belongs" to the claimed split: evals.gold.split.
    assign_split only recognizes the 12 real corpus documents and raises for anything else,
    so that check would make preflight unable to validate a hand-built goldset using
    synthetic doc_ids — exactly the kind of goldset every offline test in this module uses.
    """
    if expected_goldset_version is not None and goldset.version != expected_goldset_version:
        raise ValueError(
            f"goldset version mismatch: loaded {goldset.version!r}, expected "
            f"{expected_goldset_version!r}"
        )
    if not chat_model.strip():
        raise ValueError("chat_model must not be blank")
    if not embedding_model.strip():
        raise ValueError("embedding_model must not be blank")
    if k < 1:
        raise ValueError("k must be at least 1")
    if not goldset.items:
        raise ValueError("goldset has no items")

    chunk_pages([], doc_id="__preflight__", chunk_size=chunk_size, overlap=overlap)

    validate_anchors(goldset, chunk_size=chunk_size, overlap=overlap, pdf_dir=pdf_dir)


def _scope_candidates(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    scope: str | None,
) -> tuple[list[Chunk], list[list[float]]]:
    """Filter the candidate set to one document's chunks before ranking, or return every
    pooled chunk unfiltered when scope is None (pooled/corpus-wide retrieval).

    Filtering happens here, before top_k_chunks ever sees the candidates - a post-filter over
    its output would silently return fewer than k results for a scoped question, for a reason
    invisible in the saved record. A scope matching zero chunks returns ([], []) rather than
    raising, leaving top_k_chunks's own empty-candidate check as the single place that error
    is raised.
    """
    if scope is None:
        return chunks, embeddings
    scoped_chunks: list[Chunk] = []
    scoped_embeddings: list[list[float]] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if chunk.doc_id == scope:
            scoped_chunks.append(chunk)
            scoped_embeddings.append(embedding)
    return scoped_chunks, scoped_embeddings


def run_eval(
    goldset: GoldSet,
    split: Split,
    client: OpenAI,
    chat_model: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    k: int,
    pdf_dir: Path = PDF_DIR,
    scoping: RetrievalScoping = "scoped",
    expected_goldset_version: str | None = None,
) -> RunRecord:
    """Run the pipeline over every question in goldset (trusted to already be split-filtered)
    and return a raw RunRecord. Computes no metrics.

    scoping controls each question's retrieval candidate set: "scoped" (the default) restricts
    every question to its own gold document's chunks; "pooled" ranks against every document's
    chunks in the split, undifferentiated. Both remain reachable - comparing them is the
    measurement this parameter exists to make possible.
    """
    preflight(
        goldset,
        chat_model,
        embedding_model,
        chunk_size,
        overlap,
        k,
        pdf_dir,
        expected_goldset_version,
    )

    documents: list[DocumentRecord] = []
    pooled_chunks: list[Chunk] = []
    pooled_embeddings: list[list[float]] = []

    for item in goldset.items:
        pages = extract_pdf_pages(pdf_dir / f"{item.doc_id}.pdf")
        purchase_order = extract_purchase_order(client, pages, chat_model)
        chunks = chunk_pages(pages, doc_id=item.doc_id, chunk_size=chunk_size, overlap=overlap)
        chunk_embeddings = (
            embed_texts(client, [chunk.text for chunk in chunks], embedding_model)
            if chunks
            else []
        )
        pooled_chunks.extend(chunks)
        pooled_embeddings.extend(chunk_embeddings)
        documents.append(
            DocumentRecord(
                doc_id=item.doc_id,
                purchase_order=purchase_order,
                chunk_count=len(chunks),
            )
        )

    items: list[ItemRecord] = []
    for item in goldset.items:
        for retrieval in item.retrieval:
            retrieved_chunk_ids: list[str] = []
            retrieved_doc_ids: list[str] = []
            retrieved_page_numbers: list[int] = []
            retrieved_texts: list[str] = []
            retrieved_scores: list[float] = []
            answer: str | None = None
            source_chunk_ids: list[str] = []
            error: str | None = None
            scope = retrieval.doc_id if scoping == "scoped" else None
            try:
                query_embedding = embed_texts(client, [retrieval.question], embedding_model)[0]
                candidate_chunks, candidate_embeddings = _scope_candidates(
                    pooled_chunks, pooled_embeddings, scope
                )
                sources = top_k_chunks(
                    candidate_chunks, candidate_embeddings, query_embedding, k=k
                )
                retrieved_chunk_ids = [source.chunk_id for source in sources]
                retrieved_doc_ids = [source.doc_id for source in sources]
                retrieved_page_numbers = [source.page_number for source in sources]
                retrieved_texts = [source.text for source in sources]
                retrieved_scores = [source.score for source in sources]
                grounded_answer = answer_question(client, retrieval.question, sources, chat_model)
                answer = grounded_answer.answer
                source_chunk_ids = grounded_answer.source_chunk_ids
            except Exception as exc:  # noqa: BLE001 - recorded per-item, run continues
                error = f"{type(exc).__name__}: {exc}"

            items.append(
                ItemRecord(
                    question_id=retrieval.question_id,
                    doc_id=item.doc_id,
                    question_class=retrieval.question_class,
                    scope=scope,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieved_page_numbers=retrieved_page_numbers,
                    retrieved_texts=retrieved_texts,
                    retrieved_scores=retrieved_scores,
                    answer=answer,
                    source_chunk_ids=source_chunk_ids,
                    error=error,
                )
            )

    config = RunConfig(
        chat_model=chat_model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        k=k,
        goldset_version=goldset.version,
        split=split,
        document_count=len(documents),
        pooled_chunk_count=len(pooled_chunks),
        scoping=scoping,
    )

    return RunRecord(
        run_id=str(uuid4()),
        created_utc=datetime.now(UTC).isoformat(),
        git_commit_sha=_current_git_commit_sha(),
        config=config,
        documents=documents,
        items=items,
    )


def save_run_record(record: RunRecord, runs_dir: Path = RUNS_DIR) -> Path:
    """Write record to runs_dir/{run_id}.json, creating runs_dir if needed. Returns the path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_path = runs_dir / f"{record.run_id}.json"
    output_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path

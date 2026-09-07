import logging
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import APITimeoutError, OpenAI, OpenAIError
from pypdf.errors import PdfReadError

from isc.models import PipelineResult
from isc.pipeline import process_document

app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(file: UploadFile, question: Annotated[str, Form()] = "") -> PipelineResult:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be blank")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Filename must end in .pdf")

    try:
        load_dotenv()
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / Path(file.filename).name
            with pdf_path.open("wb") as temp_file:
                copyfileobj(file.file, temp_file)
            with OpenAI(api_key=os.environ["OPENAI_API_KEY"]) as client:
                return process_document(
                    pdf_path=pdf_path,
                    question=question,
                    client=client,
                    chat_model=os.environ["OPENAI_CHAT_MODEL"],
                    embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
                )
    except Exception as exc:
        logger.exception("Document processing request failed")
        status_code = 500
        detail = "Unable to process the document."
        if isinstance(exc, APITimeoutError):
            status_code = 504
            detail = "The AI service timed out. Please try again later."
        elif isinstance(exc, OpenAIError):
            status_code = 502
            detail = "The AI service could not complete the request."
        elif isinstance(exc, PdfReadError) or (
            isinstance(exc, ValueError)
            and (
                str(exc).startswith("PDF could not be read: ")
                or str(exc) == "Encrypted PDFs are not supported"
            )
        ):
            status_code = 400
            detail = "The PDF could not be read. Upload a valid, unencrypted PDF."
        elif isinstance(exc, ValueError) and str(exc) in (
            "No native PDF text found; OCR is not supported by this PoC",
            "No chunks available for retrieval",
        ):
            status_code = 422
            detail = "No extractable text found. Upload a PDF containing selectable text."
        raise HTTPException(status_code=status_code, detail=detail) from exc

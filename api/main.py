# ─────────────────────────────────────────────────────────
# api/main.py — FastAPI REST API
#
# Run with: uvicorn api.main:app --reload
# ─────────────────────────────────────────────────────────

import sys
import os
import tempfile
from pathlib import Path
from typing import Optional

# Pre-initialize PyTorch/SentenceTransformers on Windows to avoid OpenMP DLL conflicts
try:
    from sentence_transformers import SentenceTransformer
    _ = SentenceTransformer
except ImportError:
    pass

# Add project root so imports work (same approach as ui/app.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Config

# Try importing custom exceptions — they may not exist yet in all environments
try:
    from src.exceptions import (
        DocumentParseError,
        EmbeddingError,
        LLMGenerationError,
    )
except ImportError:
    # Fall back to generic exceptions if the module isn't there
    DocumentParseError = Exception
    EmbeddingError = Exception
    LLMGenerationError = Exception


# ── App setup ─────────────────────────────────────────────

app = FastAPI(
    title="Document Intelligence API",
    version="1.0.0",
    description="RAG pipeline for document Q&A",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy pipeline singleton ──────────────────────────────

_pipeline = None


def get_pipeline():
    """Get or create the pipeline instance. Only initialized once."""
    global _pipeline
    if _pipeline is None:
        from pipeline import DocumentPipeline
        _pipeline = DocumentPipeline()
    return _pipeline


# ── Request / Response Models ─────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    document_name: str = Field(..., min_length=1, description="Which document to query")
    mode: str = Field(default="qa", description="Query mode: qa, extract, summarize, anomaly")


class HealthResponse(BaseModel):
    status: str
    version: str


class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    visual_pages: int
    table_pages: int
    total_chunks: int
    indexed_chunks: int
    processing_time_sec: float


class QueryResponse(BaseModel):
    answer: str
    sources: list
    entities: dict
    anomalies: list
    faithfulness: Optional[dict] = None
    context_chunks_used: Optional[int] = None
    retrieval: Optional[dict] = None
    retrieved_texts: Optional[list] = None


class DocumentListResponse(BaseModel):
    documents: list[str]


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    # Validate file format
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = Path(file.filename).suffix.lower().lstrip(".")
    if extension not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{extension}. "
                   f"Supported: {', '.join(Config.SUPPORTED_FORMATS)}"
        )

    # Validate file size — read into memory first
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > Config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {Config.MAX_FILE_SIZE_MB} MB."
        )

    # Write to temp file for the pipeline
    suffix = Path(file.filename).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        pipeline = get_pipeline()
        summary = pipeline.index(tmp_path, original_filename=file.filename)
        return summary

    except (DocumentParseError, EmbeddingError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/v1/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    """Query an indexed document."""
    pipeline = get_pipeline()

    # Check that the document exists
    if req.document_name not in pipeline.get_document_list():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{req.document_name}' not found. "
                   f"Available: {pipeline.get_document_list()}"
        )

    try:
        result = pipeline.query(
            question=req.question,
            collection_name=req.document_name,
            mode=req.mode,
        )
        return result

    except (LLMGenerationError,) as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/v1/documents", response_model=DocumentListResponse)
def list_documents():
    """List all indexed documents."""
    pipeline = get_pipeline()
    return {"documents": pipeline.get_document_list()}


@app.delete("/api/v1/documents/{document_name}")
def delete_document(document_name: str):
    """Remove a document from the index."""
    pipeline = get_pipeline()

    if document_name not in pipeline.get_document_list():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found"
        )

    pipeline.remove_document(document_name)
    return {"detail": f"Document '{document_name}' removed"}

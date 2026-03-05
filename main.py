"""
main.py
───────
FastAPI application entry point.

• Mounts the Jinja2 template engine and static files.
• Registers the upload API route.
• Serves the frontend at the root path.
• Configures structured logging.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from doc_handler import extract_content, rebuild_document
from ai_service import process_document
from diff_service import generate_diff_html

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocCorrect",
    description="Upload a .docx → AI‑powered grammar & clarity correction → download corrected .docx",
    version="1.0.0",
)

# ── Static files & templates ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ── In-memory store for comparison data (keyed by UUID) ──────────────────────
_comparison_store: dict[str, dict] = {}



# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the upload form."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "max_size_mb": config.MAX_FILE_SIZE_MB,
        },
    )


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    prompt: str = Form(default="Please fix any grammatical or conceptual errors.")
):
    """
    Accept a .docx upload, segment it, correct its content via OpenAI,
    and return the corrected document as a downloadable file.
    """

    # ── 1. Validate file extension ────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .docx files are accepted.",
        )

    # ── 2. Validate MIME type ─────────────────────────────────────────────────
    if file.content_type and file.content_type != config.ALLOWED_CONTENT_TYPE:
        # Some clients send generic types; we only hard‑reject known mismatches
        if file.content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {file.content_type}. Expected a .docx file.",
            )

    # ── 3. Read & validate file size ──────────────────────────────────────────
    contents = await file.read()
    if len(contents) > config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {config.MAX_FILE_SIZE_MB} MB.",
        )

    # ── 4a. Validate verification type ────────────────────────────────────────

    logger.info(
        "Received file: %s  (%d bytes)",
        file.filename,
        len(contents),
    )

    # ── 4. Extract content blocks from the document ───────────────────────────
    try:
        import io
        blocks = extract_content(io.BytesIO(contents))
    except Exception as exc:
        logger.exception("Failed to parse .docx")
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the document. It may be corrupted: {exc}",
        )

    if not blocks:
        raise HTTPException(
            status_code=400,
            detail="The document appears to be empty.",
        )

    logger.info("Extracted %d content blocks", len(blocks))

    # ── 4b. Capture original text for comparison ──────────────────────────────
    original_text = "\n".join(
        b.get("text", "") for b in blocks if b.get("text", "").strip()
    )


    # ── 5. Correct content via OpenAI ─────────────────────────────────────────
    try:
        verified_text = process_document(original_text, prompt)
    except RuntimeError as exc:
        # Raised when API key is missing
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("AI processing failed")
        raise HTTPException(
            status_code=502,
            detail="The AI service encountered an error. Please try again later.",
        )

    # ── 5c. Store comparison data ─────────────────────────────────────────────
    comparison_id = str(uuid.uuid4())
    _comparison_store[comparison_id] = {
        "original": original_text,
        "verified": verified_text,
    }
    logger.info("Stored comparison data with ID: %s", comparison_id)

    # ── 6. Rebuild the corrected document ─────────────────────────────────────
    try:
        doc_buffer = rebuild_document(verified_text)
    except Exception as exc:
        logger.exception("Failed to rebuild document")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate the corrected document.",
        )

    # ── 7. Return the file for download ───────────────────────────────────────
    original_stem = Path(file.filename).stem
    download_name = f"corrected_{original_stem}.docx"

    logger.info("Returning corrected document: %s", download_name)

    return StreamingResponse(
        doc_buffer,
        media_type=config.ALLOWED_CONTENT_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Comparison-Id": comparison_id,
        },
    )


# ── Comparison page ───────────────────────────────────────────────────────────

@app.get("/comparison/{comparison_id}", response_class=HTMLResponse)
async def comparison_page(request: Request, comparison_id: str):
    """
    Render the side-by-side diff comparison page.
    """
    data = _comparison_store.get(comparison_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Comparison not found. It may have expired or already been viewed.",
        )

    diff_result = generate_diff_html(data["original"], data["verified"])

    return templates.TemplateResponse(
        "comparison.html",
        {
            "request": request,
            "original_html": diff_result["original_html"],
            "verified_html": diff_result["verified_html"],
            "combined_html": diff_result["combined_html"],
            "stats": diff_result["stats"],
        },
    )



# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple health‑check endpoint for load balancers / monitoring."""
    return {"status": "ok"}

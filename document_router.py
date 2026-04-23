"""
Document Router for Mortgage AI

FastAPI router for document upload and OCR extraction endpoints.
Handles PDF and image uploads for pay stubs, bank statements, and tax returns.

Usage:
    from document_router import router
    app.include_router(router, prefix="/api")
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/documents", tags=["documents"])

# =============================================================================
# Configuration
# =============================================================================

# Upload directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Pydantic Models
# =============================================================================


class ExtractedFields(BaseModel):
    """Extracted fields from document."""
    annual_income: Optional[float] = None
    monthly_income: Optional[float] = None
    employer_name: Optional[str] = None
    pay_period: Optional[str] = None
    bank_balance: Optional[float] = None
    gross_pay_ytd: Optional[float] = None
    net_pay: Optional[float] = None
    pay_date: Optional[str] = None
    employee_name: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response from document upload endpoint."""
    success: bool
    document_id: str
    extracted_fields: Dict[str, Any]
    confidence: float
    document_type: str
    pages_processed: int
    extraction_method: str
    warnings: list[str]
    raw_text_preview: str


class DocumentUploadRequest(BaseModel):
    """Request schema for document upload (for documentation)."""
    file: str = Field(..., description="File uploaded as multipart/form-data")


# =============================================================================
# Utility Functions
# =============================================================================


def validate_file(file: UploadFile) -> tuple[bool, str]:
    """
    Validate uploaded file.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"

    # Check MIME type
    allowed_mime_types = ["application/pdf", "image/png", "image/jpeg"]
    if file.content_type not in allowed_mime_types:
        return False, f"Invalid MIME type: {file.content_type}"

    return True, ""


async def get_file_size(file: UploadFile) -> int:
    """Get file size by reading content."""
    content = await file.read()
    return len(content)


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a financial document.

    Extracts income, employment, and bank data from:
    - Pay stubs
    - Bank statements
    - Tax returns (W-2, 1099)

    **File requirements:**
    - Format: PDF, PNG, or JPG
    - Maximum size: 10MB
    - Must be readable (not heavily compressed or corrupted)

    **Extraction confidence:**
    - High (≥80%): Most fields extracted, values validated
    - Medium (60-79%): Some fields extracted, manual review recommended
    - Low (<60%): Poor extraction, manual entry required
    """
    # Validate file
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Read file content
    try:
        content = await file.read()
        file_size = len(content)
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read file content")

    # Check file size
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum: 10MB"
        )

    # Generate unique filename
    file_ext = Path(file.filename).suffix.lower()
    document_id = str(uuid.uuid4())
    temp_path = Path(UPLOAD_DIR) / f"{document_id}{file_ext}"

    try:
        # Save temporary file
        with open(temp_path, "wb") as f:
            f.write(content)

        # Import OCR extractor
        try:
            from ocr_extractor import extract_document
        except ImportError as e:
            logger.error(f"OCR extractor not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="OCR service unavailable. Please install dependencies: pip install pdfplumber pytesseract pillow pdf2image"
            )

        # Extract data from document
        logger.info(f"Extracting from {temp_path}...")
        result = extract_document(str(temp_path), force_ocr=False)

        # Log extraction result
        logger.info(
            f"Extraction complete: type={result.document_type}, "
            f"confidence={result.confidence:.0%}, method={result.extraction_method}"
        )

        # Build response
        response = DocumentUploadResponse(
            success=True,
            document_id=document_id,
            extracted_fields=result.extracted_fields,
            confidence=result.confidence,
            document_type=result.document_type,
            pages_processed=result.pages_processed,
            extraction_method=result.extraction_method,
            warnings=result.warnings,
            raw_text_preview=result.raw_text_preview,
        )

        return response

    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Cleanup temporary file
        try:
            temp_path.unlink()
        except OSError:
            pass


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats and their limitations."""
    return {
        "supported_formats": [
            {
                "extension": ".pdf",
                "mime_type": "application/pdf",
                "description": "Portable Document Format",
                "extraction_methods": ["pdfplumber (digital)", "pytesseract (scanned)"],
            },
            {
                "extension": ".png",
                "mime_type": "image/png",
                "description": "Portable Network Graphics",
                "extraction_methods": ["pytesseract (OCR)"],
            },
            {
                "extension": ".jpg",
                "extension_alt": ".jpeg",
                "mime_type": "image/jpeg",
                "description": "JPEG Image",
                "extraction_methods": ["pytesseract (OCR)"],
            },
        ],
        "limitations": {
            "max_file_size_mb": 10,
            "max_pages": 10,
            "min_confidence_threshold": 0.3,
        },
        "best_practices": [
            "Use digital PDFs when available (faster, more accurate)",
            "Ensure scanned documents are at least 300 DPI",
            "Avoid heavily compressed images",
            "Make sure text is clearly visible and not obscured",
        ],
    }


@router.post("/validate")
async def validate_document(file: UploadFile = File(...)):
    """
    Quick validation of document without full extraction.

    Returns file metadata and whether it's suitable for processing.
    """
    # Validate file
    is_valid, error_msg = validate_file(file)

    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={
                "valid": False,
                "error": error_msg,
            }
        )

    # Get file size
    content = await file.read()
    file_size = len(content)

    return {
        "valid": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": file_size,
        "size_mb": round(file_size / 1024 / 1024, 2),
        "within_size_limit": file_size <= MAX_FILE_SIZE,
    }

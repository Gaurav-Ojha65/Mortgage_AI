"""
OCR Extractor for Mortgage AI

Extracts financial data from uploaded documents (pay stubs, bank statements,
tax returns) using pdfplumber for digital PDFs and pytesseract for scanned
documents.

Usage:
    from ocr_extractor import extract_document

    result = extract_document("pay_stub.pdf")
    print(result["extracted_fields"])
    print(f"Confidence: {result['confidence']:.0%}")

Dependencies:
    pip install pdfplumber pytesseract pillow pdf2image

System requirements:
    - Tesseract OCR installed: https://github.com/tesseract-ocr/tesseract
    - poppler-utils for pdf2image (Linux: apt install poppler-utils)
"""

import os
import re
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# PDF processing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# PDF to image conversion
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_bytes = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExtractedFields:
    """Standardized extracted fields from document."""
    annual_income: Optional[float] = None
    monthly_income: Optional[float] = None
    employer_name: Optional[str] = None
    pay_period: Optional[str] = None
    bank_balance: Optional[float] = None
    gross_pay_ytd: Optional[float] = None
    net_pay: Optional[float] = None
    pay_date: Optional[str] = None
    employee_name: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of document extraction."""
    extracted_fields: Dict[str, Any]
    confidence: float
    raw_text_preview: str
    warnings: List[str]
    document_type: str = "unknown"
    pages_processed: int = 0
    extraction_method: str = "unknown"


# =============================================================================
# Field Extraction Patterns
# =============================================================================

# Regex patterns for field extraction
PATTERNS = {
    # Income patterns
    "annual_income": [
        r"(?:annual\s*salary|annual\s*income|yearly\s*income)\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
        r"\$\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s*year|annually|\/year)",
    ],
    "monthly_income": [
        r"(?:monthly\s*income|monthly\s*pay|monthly\s*salary)\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
        r"\$\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s*month|monthly|\/month)",
    ],
    "gross_pay_ytd": [
        r"(?:ytd\s*gross|gross\s*ytd|year\s*to\s*date\s*gross)\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
        r"gross\s*pay\s*ytd\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
    ],
    "bank_balance": [
        r"(?:balance|available\s*balance|account\s*balance|current\s*balance)\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
        r"ending\s*balance\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
    ],
    "net_pay": [
        r"(?:net\s*pay|take\s*home\s*pay|net\s*amount)\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
        r"total\s*net\s*pay\s*[:\$]?\s*\$?([\d,]+(?:\.\d{2})?)",
    ],

    # Employer patterns
    "employer_name": [
        r"(?:employer|company|organization)\s*[:]\s*([A-Za-z\s&,.]+?)(?:\n|$)",
        r"^([A-Z][A-Za-z\s&,.]+(?:Inc|LLC|Ltd|Corp)\.?)$",
    ],

    # Pay period patterns
    "pay_period": [
        r"(?:pay\s*period|period\s*ending|pay\s*date)\s*[:]\s*(\d{1,2}\/\d{1,2}\/\d{2,4})",
        r"(?:for\s*the\s*period|period)\s+(\d{1,2}\/\d{1,2}\/\d{2,4})\s*(?:to|-)\s*(\d{1,2}\/\d{1,2}\/\d{2,4})",
    ],

    # Employee name patterns
    "employee_name": [
        r"(?:employee|name|employee\s*name)\s*[:]\s*([A-Za-z\s]+?)(?:\n|$)",
        r"pay\s*to\s*the\s*order\s*of\s+([A-Za-z\s]+?)(?:\n|$)",
    ],
}

# Keywords for gross pay detection
GROSS_PAY_KEYWORDS = [
    "gross pay", "gross", "base pay", "regular pay",
    "total earnings", "current gross", "period gross"
]

# Keywords for bank balance
BALANCE_KEYWORDS = [
    "balance", "available balance", "account balance",
    "current balance", "ending balance", "available"
]


# =============================================================================
# Utility Functions
# =============================================================================


def parse_currency(value: str) -> Optional[float]:
    """Parse currency string to float."""
    if not value:
        return None

    # Remove currency symbols and commas
    cleaned = re.sub(r'[,$\s]', '', value)

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(value: str) -> Optional[str]:
    """Parse date string to ISO format."""
    date_patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{2,4})',  # MM/DD/YYYY or MM/DD/YY
        r'(\d{4})-(\d{2})-(\d{2})',         # YYYY-MM-DD
        r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',   # Month DD, YYYY
    ]

    for pattern in date_patterns:
        match = re.search(pattern, value)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                # Try to construct ISO date
                try:
                    if pattern == date_patterns[0]:  # MM/DD/YY
                        month, day, year = groups
                        year = int(year) if int(year) > 2000 else int(year) + 2000
                        return f"{year}-{int(month):02d}-{int(day):02d}"
                    elif pattern == date_patterns[1]:  # YYYY-MM-DD
                        return value
                except (ValueError, IndexError):
                    continue

    return value  # Return original if parsing fails


def extract_with_patterns(text: str, field_name: str) -> Optional[str]:
    """Extract field value using regex patterns."""
    patterns = PATTERNS.get(field_name, [])

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Return first captured group or full match
            if match.groups():
                return match.group(1)
            return match.group(0)

    return None


def extract_with_keywords(text: str, keywords: List[str], context_lines: int = 2) -> Optional[float]:
    """Extract numeric value near keywords."""
    lines = text.split('\n')

    for i, line in enumerate(lines):
        for keyword in keywords:
            if keyword.lower() in line.lower():
                # Look at current line and surrounding context
                context_start = max(0, i - context_lines)
                context_end = min(len(lines), i + context_lines + 1)
                context = '\n'.join(lines[context_start:context_end])

                # Find dollar amounts in context
                dollar_pattern = r'\$?\s*([\d,]+(?:\.\d{2})?)'
                matches = re.findall(dollar_pattern, context)

                if matches:
                    # Return the largest amount (likely the total)
                    amounts = [parse_currency(m) for m in matches]
                    amounts = [a for a in amounts if a is not None and a > 0]
                    if amounts:
                        return max(amounts)

    return None


def calculate_confidence(extracted: Dict[str, Any], raw_text: str) -> float:
    """
    Calculate confidence score based on:
    - Number of fields extracted
    - Text quality (length, readability)
    - Field value validity
    """
    if not raw_text or len(raw_text.strip()) < 50:
        return 0.1

    # Count successfully extracted fields
    target_fields = [
        'annual_income', 'monthly_income', 'employer_name',
        'pay_period', 'bank_balance', 'gross_pay_ytd'
    ]

    extracted_count = sum(
        1 for field in target_fields
        if extracted.get(field) is not None
    )

    # Base confidence from field extraction rate
    field_confidence = extracted_count / len(target_fields)

    # Text quality bonus
    text_quality = min(1.0, len(raw_text) / 1000)

    # Combine factors
    confidence = (field_confidence * 0.7) + (text_quality * 0.3)

    return min(1.0, max(0.0, confidence))


# =============================================================================
# PDF Extraction (Digital PDFs)
# =============================================================================


def extract_from_pdf_plumber(file_path: str) -> Tuple[str, int]:
    """
    Extract text from PDF using pdfplumber (for digital PDFs).

    Returns:
        Tuple of (extracted_text, page_count)
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

    raw_text_parts = []
    page_count = 0

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)

        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                raw_text_parts.append(text)

            # Also try table extraction for structured data
            tables = page.extract_tables()
            for table in tables:
                if table:
                    for row in table:
                        if row:
                            row_text = ' | '.join(str(cell) for cell in row if cell)
                            raw_text_parts.append(row_text)

    return '\n'.join(raw_text_parts), page_count


# =============================================================================
# OCR Extraction (Scanned Documents)
# =============================================================================


def extract_from_ocr(file_path: str) -> Tuple[str, int]:
    """
    Extract text from scanned PDF using pytesseract.

    Returns:
        Tuple of (extracted_text, page_count)
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract not installed. Run: pip install pytesseract")

    if not PIL_AVAILABLE:
        raise ImportError("Pillow not installed. Run: pip install pillow")

    raw_text_parts = []
    page_count = 0

    # Check file type
    file_ext = Path(file_path).suffix.lower()

    if file_ext in ['.pdf', '.PDF']:
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image not installed. Run: pip install pdf2image")

        # Convert PDF pages to images
        try:
            images = convert_from_bytes(open(file_path, 'rb').read(), dpi=300)
            page_count = len(images)

            for i, image in enumerate(images):
                # OCR each page
                text = pytesseract.image_to_string(image)
                raw_text_parts.append(f"--- Page {i+1} ---\n{text}")

        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            raise

    elif file_ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        # Direct image OCR
        image = Image.open(file_path)
        page_count = 1
        text = pytesseract.image_to_string(image)
        raw_text_parts.append(text)

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    return '\n'.join(raw_text_parts), page_count


# =============================================================================
# Field Extraction Logic
# =============================================================================


def extract_fields(text: str) -> Dict[str, Any]:
    """
    Extract all target fields from raw text.

    Returns:
        Dictionary of extracted fields
    """
    fields = {}

    # Extract annual income
    annual_income_str = extract_with_patterns(text, "annual_income")
    if annual_income_str:
        fields["annual_income"] = parse_currency(annual_income_str)

    # Extract monthly income
    monthly_income_str = extract_with_patterns(text, "monthly_income")
    if monthly_income_str:
        fields["monthly_income"] = parse_currency(monthly_income_str)

    # If no monthly income but have annual, calculate
    if "monthly_income" not in fields and "annual_income" in fields:
        fields["monthly_income"] = fields["annual_income"] / 12

    # Extract employer name
    employer_str = extract_with_patterns(text, "employer_name")
    if employer_str:
        fields["employer_name"] = employer_str.strip()

    # Extract pay period
    pay_period_str = extract_with_patterns(text, "pay_period")
    if pay_period_str:
        fields["pay_period"] = parse_date(pay_period_str)

    # Extract bank balance using keywords
    bank_balance = extract_with_keywords(text, BALANCE_KEYWORDS)
    if bank_balance:
        fields["bank_balance"] = bank_balance

    # Extract gross pay YTD
    gross_pay_ytd_str = extract_with_patterns(text, "gross_pay_ytd")
    if gross_pay_ytd_str:
        fields["gross_pay_ytd"] = parse_currency(gross_pay_ytd_str)

    # Try keyword-based gross pay extraction
    if "gross_pay_ytd" not in fields:
        gross_pay = extract_with_keywords(text, GROSS_PAY_KEYWORDS)
        if gross_pay:
            fields["gross_pay_ytd"] = gross_pay

    # Extract net pay
    net_pay_str = extract_with_patterns(text, "net_pay")
    if net_pay_str:
        fields["net_pay"] = parse_currency(net_pay_str)

    # Extract employee name
    employee_str = extract_with_patterns(text, "employee_name")
    if employee_str:
        fields["employee_name"] = employee_str.strip()

    return fields


def generate_warnings(extracted: Dict[str, Any]) -> List[str]:
    """Generate warnings for missing or suspicious fields."""
    warnings = []

    # Check for missing critical fields
    if not extracted.get("annual_income") and not extracted.get("monthly_income"):
        warnings.append("Could not extract income information")

    if not extracted.get("employer_name"):
        warnings.append("Could not extract employer name")

    if not extracted.get("pay_period"):
        warnings.append("Could not extract pay period")

    # Check for suspicious values
    annual = extracted.get("annual_income")
    if annual:
        if annual < 10000:
            warnings.append(f"Annual income seems unusually low: ${annual:,.0f}")
        elif annual > 10000000:
            warnings.append(f"Annual income seems unusually high: ${annual:,.0f}")

    monthly = extracted.get("monthly_income")
    if monthly and annual:
        expected_monthly = annual / 12
        if abs(monthly - expected_monthly) > expected_monthly * 0.3:
            warnings.append(f"Monthly income doesn't match annual (expected ~${expected_monthly:,.0f})")

    return warnings


# =============================================================================
# Main Extraction Function
# =============================================================================


def extract_document(
    file_path: str,
    force_ocr: bool = False,
    min_confidence: float = 0.3
) -> ExtractionResult:
    """
    Extract financial data from a document.

    Args:
        file_path: Path to PDF or image file
        force_ocr: Force OCR even for digital PDFs (useful for scanned PDFs)
        min_confidence: Minimum confidence threshold

    Returns:
        ExtractionResult with extracted fields and metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type not supported
        ImportError: If required dependencies missing
    """
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Validate file type
    file_ext = Path(file_path).suffix.lower()
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg']

    if file_ext not in supported_extensions:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported: {supported_extensions}")

    # Validate file size (max 10MB)
    max_size = 10 * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB (max 10MB)")

    logger.info(f"Extracting from {file_path} (size: {file_size / 1024:.0f}KB)")

    # Extract text
    raw_text = ""
    page_count = 0
    extraction_method = "unknown"
    warnings = []

    try:
        if file_ext == '.pdf' and not force_ocr:
            # Try digital extraction first
            try:
                raw_text, page_count = extract_from_pdf_plumber(file_path)
                extraction_method = "pdfplumber"
                logger.info(f"Extracted {len(raw_text)} chars via pdfplumber ({page_count} pages)")
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}. Falling back to OCR...")

        # Fall back to OCR if needed
        if not raw_text or force_ocr:
            raw_text, page_count = extract_from_ocr(file_path)
            extraction_method = "pytesseract"
            logger.info(f"Extracted {len(raw_text)} chars via OCR ({page_count} pages)")

    except ImportError as e:
        warnings.append(f"Extraction limited: {str(e)}")
        raw_text = ""
        extraction_method = "failed"

    # Extract fields from text
    if raw_text:
        extracted_fields = extract_fields(raw_text)
        confidence = calculate_confidence(extracted_fields, raw_text)
        field_warnings = generate_warnings(extracted_fields)
        warnings.extend(field_warnings)
    else:
        extracted_fields = {}
        confidence = 0.0
        warnings.append("No text could be extracted from document")

    # Build result
    result = ExtractionResult(
        extracted_fields=asdict(ExtractedFields(**extracted_fields)),
        confidence=confidence,
        raw_text_preview=raw_text[:500] if raw_text else "",
        warnings=warnings,
        document_type=_classify_document(raw_text),
        pages_processed=page_count,
        extraction_method=extraction_method
    )

    logger.info(f"Extraction complete: confidence={confidence:.0%}, warnings={len(warnings)}")

    return result


def _classify_document(text: str) -> str:
    """Classify document type based on content."""
    if not text:
        return "unknown"

    text_lower = text.lower()

    # Pay stub indicators
    pay_stub_keywords = ["pay stub", "pay statement", "earnings statement", "pay period"]
    if any(kw in text_lower for kw in pay_stub_keywords):
        return "pay_stub"

    # Bank statement indicators
    bank_keywords = ["bank statement", "account statement", "transaction history", "balance"]
    if any(kw in text_lower for kw in bank_keywords):
        return "bank_statement"

    # Tax return indicators
    tax_keywords = ["tax return", "1040", "w-2", "w2", "1099"]
    if any(kw in text_lower for kw in tax_keywords):
        return "tax_return"

    return "financial_document"


# =============================================================================
# Async version for FastAPI
# =============================================================================


async def extract_document_async(
    file_content: bytes,
    filename: str,
    force_ocr: bool = False
) -> ExtractionResult:
    """
    Async version that accepts file content directly.

    Args:
        file_content: Raw file bytes
        filename: Original filename (for extension detection)
        force_ocr: Force OCR extraction

    Returns:
        ExtractionResult
    """
    import tempfile
    import asyncio

    # Save to temp file
    file_ext = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        # Run extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: extract_document(tmp_path, force_ocr=force_ocr)
        )
        return result
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract financial data from documents")
    parser.add_argument("file_path", help="Path to PDF or image file")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR extraction")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    try:
        result = extract_document(args.file_path, force_ocr=args.force_ocr)

        if args.json:
            output = json.dumps(asdict(result), indent=2, default=str)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"Result saved to {args.output}")
            else:
                print(output)
        else:
            print("\n" + "=" * 60)
            print("EXTRACTION RESULT")
            print("=" * 60)
            print(f"Document Type: {result.document_type}")
            print(f"Extraction Method: {result.extraction_method}")
            print(f"Pages Processed: {result.pages_processed}")
            print(f"Confidence: {result.confidence:.0%}")
            print("\nExtracted Fields:")
            for field, value in asdict(result.extracted_fields).items():
                if value is not None:
                    print(f"  {field}: {value}")

            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  ⚠ {warning}")

            print("\nRaw Text Preview:")
            print(result.raw_text_preview[:300] + "..." if len(result.raw_text_preview) > 300 else result.raw_text_preview)

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"Error: {e}")

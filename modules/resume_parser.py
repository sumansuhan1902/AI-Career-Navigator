"""In-memory text extraction for PDF resumes."""

from __future__ import annotations

import re
from typing import BinaryIO

import pymupdf


MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 20
PDF_HEADER_PATTERN = re.compile(rb"(?m)^%PDF-[0-9]\.[0-9](?:\r\n|\r|\n|$)")


class ResumeParsingError(Exception):
    """Base exception for user-correctable PDF resume issues."""


class PDFTooLargeError(ResumeParsingError):
    """Raised when an uploaded PDF exceeds the configured byte limit."""


class PDFPageLimitError(ResumeParsingError):
    """Raised when an uploaded PDF exceeds the configured page limit."""


class EmptyPDFError(ResumeParsingError):
    """Raised when a valid PDF contains no extractable text or images."""


class ScannedPDFError(ResumeParsingError):
    """Raised when a PDF appears to contain scanned images instead of text."""


class InvalidPDFError(ResumeParsingError):
    """Raised when a PDF cannot be opened or read."""


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract text from a bounded PDF stream without writing it to disk."""
    try:
        file_size = getattr(pdf_file, "size", None)
        if isinstance(file_size, int) and file_size > MAX_UPLOAD_SIZE_BYTES:
            raise PDFTooLargeError("The uploaded PDF exceeds the 10 MB size limit.")

        getvalue = getattr(pdf_file, "getvalue", None)
        if callable(getvalue):
            pdf_bytes = getvalue()
        else:
            original_position = pdf_file.tell()
            try:
                pdf_file.seek(0)
                chunks = bytearray()
                while len(chunks) <= MAX_UPLOAD_SIZE_BYTES:
                    chunk_size = min(
                        64 * 1024,
                        MAX_UPLOAD_SIZE_BYTES + 1 - len(chunks),
                    )
                    chunk = pdf_file.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise InvalidPDFError(
                            "The uploaded file did not contain readable PDF data."
                        )
                    chunks.extend(chunk)
                pdf_bytes = chunks
            finally:
                pdf_file.seek(original_position)
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            raise InvalidPDFError(
                "The uploaded file did not contain readable PDF data."
            )
        if len(pdf_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise PDFTooLargeError("The uploaded PDF exceeds the 10 MB size limit.")
    except (AttributeError, OSError, ValueError) as error:
        raise InvalidPDFError("The uploaded file could not be read.") from error

    if not pdf_bytes:
        raise EmptyPDFError("The uploaded PDF is empty.")
    if not PDF_HEADER_PATTERN.search(bytes(pdf_bytes[:1024])):
        raise InvalidPDFError(
            "The uploaded file is not a valid PDF (missing PDF signature)."
        )

    try:
        with pymupdf.open(stream=bytes(pdf_bytes), filetype="pdf") as document:
            if document.needs_pass:
                raise InvalidPDFError("Password-protected PDFs are not supported.")
            if document.page_count == 0:
                raise EmptyPDFError("The PDF contains no pages.")
            if document.page_count > MAX_PDF_PAGES:
                raise PDFPageLimitError(
                    "The uploaded PDF exceeds the 20-page limit."
                )

            page_text = [page.get_text("text") for page in document]
            extracted_text = "\n".join(text.strip() for text in page_text if text.strip())
            if extracted_text:
                return extracted_text

            has_images = any(page.get_images(full=True) for page in document)
            if has_images:
                raise ScannedPDFError(
                    "This PDF appears to be scanned. OCR is not supported yet; "
                    "please use a text-based PDF."
                )
            raise EmptyPDFError("The PDF contains no extractable text.")
    except ResumeParsingError:
        raise
    except Exception as error:
        raise InvalidPDFError(
            "The PDF is invalid or corrupted and could not be read."
        ) from error

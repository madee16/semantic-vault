"""
Document parser module extracting and normalizing text from TXT, Markdown, and PDF documents.
Calculates SHA-256 content hash for deduplication.
"""
import hashlib
import io
import re
from typing import Tuple
import pymupdf  # Modern PyMuPDF import


class EmptyDocumentError(Exception):
    """Raised when an uploaded document contains no readable text."""
    pass


class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file extension is uploaded."""
    pass


class DocumentParser:
    """Parser and text extractor for TXT, MD, and PDF documents."""

    SUPPORTED_EXTENSIONS = {"txt", "md", "markdown", "pdf"}

    @classmethod
    def parse(cls, file_bytes: bytes, filename: str) -> Tuple[str, str, str]:
        """
        Extract and normalize text from raw file bytes.
        Returns: (normalized_text, file_type, content_hash)
        """
        extension = filename.lower().split(".")[-1] if "." in filename else ""
        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension}'. Supported types: TXT, MD, PDF."
            )

        normalized_type = "md" if extension in {"md", "markdown"} else extension

        if normalized_type == "pdf":
            raw_text = cls._extract_pdf(file_bytes)
        elif normalized_type == "md":
            raw_text = cls._extract_markdown(file_bytes)
        elif normalized_type == "txt":
            raw_text = cls._extract_text(file_bytes)
        else:
            raise UnsupportedFileTypeError(f"File type '{normalized_type}' cannot be parsed.")

        normalized_text = cls._normalize_text(raw_text)
        if not normalized_text:
            raise EmptyDocumentError("Document contains no readable text or only whitespace.")

        content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        return normalized_text, normalized_type, content_hash

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF pages using PyMuPDF."""
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                text = page.get_text()
                if text:
                    pages_text.append(text)
            doc.close()
            return "\n\n".join(pages_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

    @classmethod
    def _extract_markdown(cls, file_bytes: bytes) -> str:
        """Extract and clean Markdown by stripping inline formatting markers while preserving structure."""
        text = cls._extract_text(file_bytes)
        # Strip bold/italic markers (**text**, *text*)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        cleaned = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", cleaned)
        # Strip markdown links [label](url) -> label
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        # Strip inline code backticks `code` -> code
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        return cleaned

    @staticmethod
    def _extract_text(file_bytes: bytes) -> str:
        """Decode raw text trying UTF-8 first, falling back to Latin-1."""
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize line endings, tabs, and duplicate whitespace while preserving paragraph structure.
        """
        # Convert CRLF to LF
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace non-breaking spaces
        normalized = normalized.replace("\u00a0", " ")
        # Collapse multiple horizontal whitespaces (spaces/tabs) into a single space
        normalized = re.sub(r"[ \t]+", " ", normalized)
        # Collapse 3+ newlines into 2 (preserve paragraph separation)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

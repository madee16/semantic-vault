"""
Unit tests for DocumentParser: PDF, Markdown, and TXT parsing, normalization, and deduplication hashing.
"""
import pytest
import fitz  # PyMuPDF

from src.services.document_parser import (
    DocumentParser,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)


def test_parse_plain_text():
    """Verify parsing and whitespace normalization of plain text."""
    raw_content = b"  Line 1.\r\n\r\n\r\nLine 2 with   extra   spaces.\t\n"
    text, file_type, content_hash = DocumentParser.parse(raw_content, "sample.txt")

    assert file_type == "txt"
    assert "Line 1." in text
    assert "Line 2 with extra spaces." in text
    assert len(content_hash) == 64  # SHA-256


def test_parse_markdown():
    """Verify markdown text extraction preserves core headings and paragraphs."""
    raw_md = b"# Main Title\n\nHere is a **bold** paragraph with [a link](https://example.com).\n\n## Section 2"
    text, file_type, content_hash = DocumentParser.parse(raw_md, "notes.md")

    assert file_type == "md"
    assert "Main Title" in text
    assert "bold paragraph" in text
    assert len(content_hash) == 64


def test_parse_pdf():
    """Verify PDF extraction via PyMuPDF (fitz) on a dynamically generated PDF buffer."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "SemanticVault PDF Extraction Test Page 1.")
    pdf_bytes = doc.tobytes()
    doc.close()

    text, file_type, content_hash = DocumentParser.parse(pdf_bytes, "manual.pdf")
    assert file_type == "pdf"
    assert "SemanticVault PDF Extraction Test Page 1." in text
    assert len(content_hash) == 64


def test_empty_document_raises_error():
    """Verify that uploading empty bytes or whitespace-only files raises EmptyDocumentError."""
    with pytest.raises(EmptyDocumentError, match="Document contains no readable text"):
        DocumentParser.parse(b"   \r\n\t  ", "empty.txt")


def test_unsupported_file_type_raises_error():
    """Verify that files with unsupported extensions raise UnsupportedFileTypeError."""
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        DocumentParser.parse(b"some content", "data.exe")

from __future__ import annotations

import io

import pytest

from src.document_text import DocumentExtractionError, extract_text

LONG_TEXT = "Project manager with ten years of experience in Berlin. " * 5


def test_plain_text_file() -> None:
    text = extract_text(LONG_TEXT.encode(), "cv.txt", "text/plain")
    assert "Project manager" in text


def test_unknown_extension_decodes_as_text() -> None:
    text = extract_text(LONG_TEXT.encode(), "cv", "")
    assert "Berlin" in text


def test_docx_extraction() -> None:
    import docx

    buffer = io.BytesIO()
    document = docx.Document()
    document.add_paragraph(LONG_TEXT)
    document.save(buffer)
    text = extract_text(buffer.getvalue(), "cv.docx")
    assert "Project manager" in text


def test_too_short_content_rejected() -> None:
    with pytest.raises(DocumentExtractionError):
        extract_text(b"hi", "cv.txt")


def test_legacy_doc_rejected() -> None:
    with pytest.raises(DocumentExtractionError):
        extract_text(LONG_TEXT.encode(), "cv.doc")


def test_broken_pdf_raises_helpful_error() -> None:
    with pytest.raises(DocumentExtractionError):
        extract_text(b"not a pdf at all, just some bytes", "cv.pdf")

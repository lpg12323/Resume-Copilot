"""Unit tests for the PDF resume parser."""

import pytest
from fpdf import FPDF
from pypdf import PdfWriter

from resume_copilot.parsers import PDFParseError, ResumePDFParser


@pytest.fixture
def parser() -> ResumePDFParser:
    """Provide a fresh parser instance for each test."""
    return ResumePDFParser()


def _make_pdf_with_text(tmp_path: pytest.TempPathFactory, filename: str, text: str) -> str:
    """Create a minimal PDF file containing the given text.

    Returns the absolute path to the created PDF.
    """
    pdf_path = tmp_path / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(str(pdf_path))

    return str(pdf_path)


def test_parse_pdf_success(parser: ResumePDFParser, tmp_path: pytest.TempPathFactory) -> None:
    """Parser should successfully extract text from a valid PDF."""
    expected_text = "Senior Python Engineer with LangGraph experience"
    pdf_path = _make_pdf_with_text(tmp_path, "valid_resume.pdf", expected_text)

    result = parser.parse_pdf(pdf_path)

    assert expected_text in result


def test_parse_pdf_file_not_found(parser: ResumePDFParser) -> None:
    """Parser should raise PDFParseError when the file does not exist."""
    with pytest.raises(PDFParseError, match="not found"):
        parser.parse_pdf("/non/existent/resume.pdf")


def test_parse_pdf_unsupported_extension(
    parser: ResumePDFParser, tmp_path: pytest.TempPathFactory
) -> None:
    """Parser should raise PDFParseError for non-PDF file extensions."""
    txt_file = tmp_path / "resume.txt"
    txt_file.write_text("This is not a PDF")

    with pytest.raises(PDFParseError, match="Unsupported file extension"):
        parser.parse_pdf(str(txt_file))


def test_parse_pdf_empty_content(
    parser: ResumePDFParser, tmp_path: pytest.TempPathFactory
) -> None:
    """Parser should raise PDFParseError when the PDF has no extractable text."""
    pdf_path = tmp_path / "empty_resume.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    with open(pdf_path, "wb") as f:
        writer.write(f)

    with pytest.raises(PDFParseError, match="no extractable text"):
        parser.parse_pdf(str(pdf_path))

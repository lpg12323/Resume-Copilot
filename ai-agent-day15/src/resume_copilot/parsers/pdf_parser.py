"""PDF resume parser based on pypdf.

This module provides ``ResumePDFParser`` for extracting plain text from
PDF resume files, together with ``PDFParseError`` for defensive error
handling.
"""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from resume_copilot.logger import logger


class PDFParseError(Exception):
    """Raised when a PDF file cannot be parsed or contains invalid content."""


class ResumePDFParser:
    """Parser for extracting text from PDF resume files.

    The parser performs defensive checks before attempting to read the file:

    - The file must exist on disk.
    - The file must have a ``.pdf`` extension (case-insensitive).
    - The extracted text must not be empty after whitespace stripping.

    Example:
        >>> parser = ResumePDFParser()
        >>> text = parser.parse_pdf("/path/to/resume.pdf")
    """

    SUPPORTED_EXTENSION = ".pdf"

    def __init__(self) -> None:
        """Initialize the parser with a reusable logger context."""
        self._logger = logger.bind(component="ResumePDFParser")

    def parse_pdf(self, file_path: str) -> str:
        """Extract plain text from a PDF resume file.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            Extracted text with whitespace normalized.

        Raises:
            PDFParseError: If the file does not exist, is not a PDF, is
                corrupted, or yields no extractable text.
        """
        path = Path(file_path)
        self._logger.info("Starting PDF parsing: {}", path)

        self._validate_file_path(path)

        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            self._logger.error("Failed to read PDF '{}': {}", path, exc)
            raise PDFParseError(f"File is not a valid PDF: '{path}'") from exc
        except Exception as exc:
            self._logger.error("Unexpected error reading PDF '{}': {}", path, exc)
            raise PDFParseError(f"Unexpected error reading PDF: '{path}'") from exc

        text_parts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
            except Exception as exc:
                self._logger.warning(
                    "Failed to extract text from page {} of '{}': {}",
                    page_number,
                    path,
                    exc,
                )
                continue

            if page_text:
                text_parts.append(page_text)
                self._logger.debug(
                    "Extracted {} characters from page {}",
                    len(page_text),
                    page_number,
                )

        raw_text = "\n".join(text_parts)
        normalized_text = self._normalize_text(raw_text)

        if not normalized_text:
            self._logger.error("PDF '{}' contains no extractable text", path)
            raise PDFParseError(f"PDF contains no extractable text: '{path}'")

        self._logger.info(
            "Successfully parsed PDF '{}': {} characters extracted",
            path,
            len(normalized_text),
        )
        return normalized_text

    def _validate_file_path(self, path: Path) -> None:
        """Validate that *path* points to an existing PDF file.

        Raises:
            PDFParseError: If the path does not exist or does not have a
                supported extension.
        """
        if not path.exists():
            self._logger.error("PDF file not found: {}", path)
            raise PDFParseError(f"PDF file not found: '{path}'")

        if not path.is_file():
            self._logger.error("Path is not a file: {}", path)
            raise PDFParseError(f"Path is not a file: '{path}'")

        if path.suffix.lower() != self.SUPPORTED_EXTENSION:
            self._logger.error(
                "Unsupported file extension '{}', expected '{}'",
                path.suffix,
                self.SUPPORTED_EXTENSION,
            )
            raise PDFParseError(
                f"Unsupported file extension '{path.suffix}' for file '{path}'. "
                f"Only '{self.SUPPORTED_EXTENSION}' files are supported."
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize extracted PDF text.

        Removes leading/trailing whitespace per line, collapses multiple
        consecutive blank lines, and strips surrounding whitespace.
        """
        lines = [line.strip() for line in text.splitlines()]
        # Collapse consecutive empty lines to a single empty line.
        normalized_lines: list[str] = []
        previous_empty = False
        for line in lines:
            is_empty = line == ""
            if is_empty and previous_empty:
                continue
            normalized_lines.append(line)
            previous_empty = is_empty

        return "\n".join(normalized_lines).strip()

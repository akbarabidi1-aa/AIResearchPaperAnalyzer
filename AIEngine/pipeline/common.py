"""Shared pieces of the page pipeline: error types and safe PDF opening."""
import os

try:
    import pymupdf                      # PyMuPDF >= 1.24.3
except ImportError:                     # older PyMuPDF only ships the legacy name
    import fitz as pymupdf

# MuPDF prints repair warnings for damaged files straight to stderr; errors are raised as exceptions instead.
pymupdf.TOOLS.mupdf_display_errors(False)

# PDF user space: 72 points per inch.
POINTS_PER_INCH = 72.0


class PdfPipelineError(Exception):
    """Base class for every expected failure of the page pipeline."""


class PdfNotFoundError(PdfPipelineError):
    """The PDF path does not exist or is not a file."""


class InvalidPdfError(PdfPipelineError):
    """The file is empty, corrupt, password-protected or not a PDF."""


class EmptyPdfError(PdfPipelineError):
    """The PDF opened but has no pages."""


class OutputDirectoryError(PdfPipelineError):
    """The output directory could not be created or written to."""


class PageRenderError(PdfPipelineError):
    """A page could not be rendered or saved."""

    def __init__(self, page_number, message):
        super().__init__(f"Page {page_number}: {message}")
        self.page_number = page_number


class PageExtractionError(PdfPipelineError):
    """Native text extraction failed for a page."""

    def __init__(self, page_number, message):
        super().__init__(f"Page {page_number}: {message}")
        self.page_number = page_number


def open_pdf(pdf_path):
    """Open a PDF and return the PyMuPDF document. The caller closes it (use `with`).

    Raises PdfNotFoundError, InvalidPdfError or EmptyPdfError; never returns an unusable document.
    """
    pdf_path = os.fspath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise PdfNotFoundError(f"PDF not found: {pdf_path}")
    if os.path.getsize(pdf_path) == 0:
        raise InvalidPdfError(f"PDF is empty (0 bytes): {pdf_path}")

    try:
        # filetype is forced so a non-PDF with a .pdf name is never opened as another format.
        doc = pymupdf.open(pdf_path, filetype="pdf")
    except Exception as exc:
        raise InvalidPdfError(f"Not a readable PDF: {pdf_path} ({exc})") from exc

    try:
        if doc.needs_pass:
            raise InvalidPdfError(f"PDF is password-protected: {pdf_path}")
        if doc.page_count == 0:
            raise EmptyPdfError(f"PDF has no pages: {pdf_path}")
    except Exception:
        doc.close()
        raise
    return doc


def ensure_directory(path):
    """Create `path` if needed and return its absolute form. Raises OutputDirectoryError."""
    path = os.path.abspath(os.fspath(path))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot create output directory: {path} ({exc})") from exc
    return path

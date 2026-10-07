"""V2 document-processing pipeline. Phase 2: page rendering and native page extraction.

Runs alongside the V1 engine (ai_engine.py); V1 does not import this package.
"""
from .common import (
    EmptyPdfError,
    InvalidPdfError,
    OutputDirectoryError,
    PageExtractionError,
    PageRenderError,
    PdfNotFoundError,
    PdfPipelineError,
)
from .native_extract import extract_native_pages, page_has_text_layer
from .page_pipeline import default_work_dir, process_pdf, validate_pages_document
from .render import DEFAULT_DPI, pdf_bbox_to_pixels, render_pdf_pages

__all__ = [
    "DEFAULT_DPI",
    "EmptyPdfError",
    "InvalidPdfError",
    "OutputDirectoryError",
    "PageExtractionError",
    "PageRenderError",
    "PdfNotFoundError",
    "PdfPipelineError",
    "default_work_dir",
    "extract_native_pages",
    "page_has_text_layer",
    "pdf_bbox_to_pixels",
    "process_pdf",
    "render_pdf_pages",
    "validate_pages_document",
]

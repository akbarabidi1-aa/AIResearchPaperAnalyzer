"""V2 document-processing pipeline. Phase 2: page rendering and native page extraction.
Phase 3: layout detection (typed page regions).

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
from .layout import (
    DEFAULT_LAYOUT_THRESHOLD,
    LayoutDetectionError,
    LayoutDetector,
    LayoutError,
    LayoutInputError,
    LayoutModelError,
    PPDocLayoutV3Detector,
    RawDetection,
    Region,
    RegionType,
    count_region_types,
    detect_document_layout,
    image_bbox_to_pdf,
    normalize_detections,
    normalize_label,
    validate_layout_document,
)
from .layout_eval import bbox_iou, class_accuracy, match_regions
from .native_extract import extract_native_pages, page_has_text_layer
from .page_pipeline import default_work_dir, process_pdf, validate_pages_document
from .render import DEFAULT_DPI, pdf_bbox_to_pixels, render_pdf_pages

__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_LAYOUT_THRESHOLD",
    "EmptyPdfError",
    "InvalidPdfError",
    "LayoutDetectionError",
    "LayoutDetector",
    "LayoutError",
    "LayoutInputError",
    "LayoutModelError",
    "OutputDirectoryError",
    "PPDocLayoutV3Detector",
    "PageExtractionError",
    "PageRenderError",
    "PdfNotFoundError",
    "PdfPipelineError",
    "RawDetection",
    "Region",
    "RegionType",
    "bbox_iou",
    "class_accuracy",
    "count_region_types",
    "default_work_dir",
    "detect_document_layout",
    "extract_native_pages",
    "image_bbox_to_pdf",
    "match_regions",
    "normalize_detections",
    "normalize_label",
    "page_has_text_layer",
    "pdf_bbox_to_pixels",
    "process_pdf",
    "render_pdf_pages",
    "validate_layout_document",
    "validate_pages_document",
]

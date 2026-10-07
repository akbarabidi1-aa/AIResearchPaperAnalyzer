"""Read the native (embedded) text of each PDF page, as blocks with bounding boxes (PyMuPDF).

Coordinates: PyMuPDF page coordinates. Origin = top-left corner of the page as displayed (after
/Rotate and CropBox), x to the right, y DOWNWARD, unit = PDF point (1/72 inch). Boxes are
[x0, y0, x1, y1] with x0 < x1 and y0 < y1. Page width/height use the same unit.

No OCR and no classification happens here: every block is a generic "text" block.
"""
import re

from .common import PageExtractionError, open_pdf, pymupdf

# has_text_layer thresholds (see page_has_text_layer).
MIN_TEXT_CHARS = 20     # letters/digits on the page
MIN_TEXT_WORDS = 3      # words of 2+ letters

_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

_TEXT_BLOCK = 0         # PyMuPDF block type: 0 = text, 1 = image


def page_has_text_layer(text):
    """True when `text` looks like a real text layer, not an image-only page.

    Rule: at least MIN_TEXT_CHARS letters/digits AND at least MIN_TEXT_WORDS words of two or more
    letters. Whitespace, punctuation and undecodable glyphs (U+FFFD) do not count. So a page that
    only carries a page number, a short stamp ("Page 3", "DRAFT") or stray fragments is treated as
    having no usable text layer, like a scanned page.
    """
    if not text:
        return False
    chars = sum(1 for ch in text if ch.isalnum())
    if chars < MIN_TEXT_CHARS:
        return False
    return len(_WORD_RE.findall(text)) >= MIN_TEXT_WORDS


def extract_native_pages(pdf_path):
    """Return one dict per page of `pdf_path`, in page order (page numbers are 1-based):

        {"page_number", "width", "height", "has_text_layer", "text",
         "blocks": [{"block_index", "bbox": [x0, y0, x1, y1], "text", "block_type": "text"}]}
    """
    with open_pdf(pdf_path) as doc:
        return extract_document(doc)


def extract_document(doc):
    """Same as extract_native_pages, for an already opened PyMuPDF document."""
    pages = []
    for index in range(doc.page_count):
        page_number = index + 1
        try:
            pages.append(_extract_page(doc.load_page(index), page_number))
        except Exception as exc:
            raise PageExtractionError(page_number, f"native extraction failed ({exc})") from exc
    return pages


def _extract_page(page, page_number):
    rect = page.rect
    width, height = rect.width, rect.height

    text = page.get_text("text")
    # PyMuPDF reports text positions in the UNROTATED page; this matrix moves them to the page as
    # displayed and rendered (identity when /Rotate is 0), so boxes always line up with the image.
    to_displayed = page.rotation_matrix
    blocks = []
    # Blocks come in the order of the PDF content stream; reading order is not reconstructed here.
    for x0, y0, x1, y1, block_text, _block_no, block_type in page.get_text("blocks"):
        if block_type != _TEXT_BLOCK or not block_text.strip():
            continue
        box = (pymupdf.Rect(x0, y0, x1, y1) * to_displayed).normalize()
        bbox = _clamp_to_page([box.x0, box.y0, box.x1, box.y1], width, height)
        if bbox is None:
            continue
        blocks.append({
            "block_index": len(blocks),
            "bbox": bbox,
            "text": block_text.strip("\n"),
            "block_type": "text",
        })

    return {
        "page_number": page_number,
        "width": width,
        "height": height,
        "has_text_layer": page_has_text_layer(text),
        "text": text,
        "blocks": blocks,
    }


def _clamp_to_page(bbox, width, height):
    """Clip a box to the page rectangle; None if nothing of it lies on the page."""
    x0, y0, x1, y1 = bbox
    x0, x1 = max(0.0, min(x0, x1)), min(float(width), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(height), max(y0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]

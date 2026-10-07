"""Render every page of a PDF to a PNG image (PyMuPDF)."""
import glob
import os

from .common import (
    POINTS_PER_INCH,
    OutputDirectoryError,
    PageRenderError,
    ensure_directory,
    open_pdf,
    pymupdf,
)

DEFAULT_DPI = 180
MIN_DPI = 36
MAX_DPI = 600

PAGE_IMAGE_PATTERN = "page_{:03d}.png"


def page_image_name(page_number):
    """Stable file name for a 1-based page number: page_001.png, page_002.png, ..."""
    return PAGE_IMAGE_PATTERN.format(page_number)


def render_pdf_pages(pdf_path, output_dir, dpi=DEFAULT_DPI):
    """Render all pages of `pdf_path` into `output_dir` as page_NNN.png.

    Returns one dict per page, in page order (page numbers are 1-based):

        {"page_number", "image_path", "width", "height", "dpi",
         "pdf_width", "pdf_height", "image_width", "image_height", "scale_x", "scale_y"}

    width/height (= image_width/image_height) are pixels; pdf_width/pdf_height are PDF points.
    A point (x, y) in page coordinates is at pixel (x * scale_x, y * scale_y) in the image.

    Fails as a whole (PdfPipelineError subclass) if any page cannot be rendered; pages are never skipped.
    """
    with open_pdf(pdf_path) as doc:
        return render_document(doc, output_dir, dpi)


def render_document(doc, output_dir, dpi=DEFAULT_DPI):
    """Same as render_pdf_pages, for an already opened PyMuPDF document."""
    if not isinstance(dpi, (int, float)) or isinstance(dpi, bool) or not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(f"dpi must be a number between {MIN_DPI} and {MAX_DPI}, got {dpi!r}")

    output_dir = ensure_directory(output_dir)
    _remove_previous_page_images(output_dir)

    zoom = dpi / POINTS_PER_INCH
    matrix = pymupdf.Matrix(zoom, zoom)
    pages = []

    for index in range(doc.page_count):
        page_number = index + 1
        image_path = os.path.join(output_dir, page_image_name(page_number))
        try:
            page = doc.load_page(index)
            rect = page.rect
            if rect.width <= 0 or rect.height <= 0:
                raise PageRenderError(page_number, "page has no area")
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(image_path)
        except PageRenderError:
            raise
        except OSError as exc:
            raise OutputDirectoryError(f"Cannot write page image: {image_path} ({exc})") from exc
        except Exception as exc:
            raise PageRenderError(page_number, f"rendering failed ({exc})") from exc

        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            raise PageRenderError(page_number, f"image was not written: {image_path}")

        pages.append({
            "page_number": page_number,
            "image_path": image_path,
            "width": pix.width,
            "height": pix.height,
            "dpi": dpi,
            "pdf_width": rect.width,
            "pdf_height": rect.height,
            "image_width": pix.width,
            "image_height": pix.height,
            # Measured from the real image size, which is rounded to whole pixels,
            # so these differ very slightly from dpi / 72.
            "scale_x": pix.width / rect.width,
            "scale_y": pix.height / rect.height,
        })

    return pages


def pdf_bbox_to_pixels(bbox, scale_x, scale_y):
    """Map a page-coordinate box [x0, y0, x1, y1] (points) to image pixels (floats)."""
    x0, y0, x1, y1 = bbox
    return [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]


def _remove_previous_page_images(output_dir):
    # A re-run on a shorter PDF must not leave page images of the earlier run behind.
    for old in glob.glob(os.path.join(glob.escape(output_dir), "page_*.png")):
        try:
            os.remove(old)
        except OSError as exc:
            raise OutputDirectoryError(f"Cannot replace existing page image: {old} ({exc})") from exc

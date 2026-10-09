"""Phase 2 page pipeline: PDF -> page images + native per-page text -> pages.json.

Working directory layout (one directory per document):

    <work_dir>/
        original.pdf            (only with copy_original=True)
        pages/page_001.png ...
        metadata/pages.json
"""
import hashlib
import json
import os
import re
import shutil
import time

from .common import OutputDirectoryError, PdfNotFoundError, ensure_directory, open_pdf
from .native_extract import extract_document
from .render import DEFAULT_DPI, render_document

SCHEMA_VERSION = 1
PAGES_DIR = "pages"
METADATA_DIR = "metadata"
PAGES_JSON = "pages.json"
ORIGINAL_PDF = "original.pdf"

COORDINATE_SYSTEM = {
    "origin": "top-left",
    "y_axis": "down",
    "unit": "pt",                   # PDF point, 1/72 inch
    "bbox_format": "x0,y0,x1,y1",
    "pixel_mapping": "pixel_x = x * scale_x; pixel_y = y * scale_y",
}


def process_pdf(pdf_path, work_dir, dpi=DEFAULT_DPI, copy_original=False):
    """Render pages, extract native text and write <work_dir>/metadata/pages.json.

    Returns {"metadata_path", "work_dir", "document", "timings"}; `document` is the content of
    pages.json and `timings` holds open/render/extract/total seconds (not written to the file).
    Raises a PdfPipelineError subclass on any failure; no pages.json is written in that case.
    """
    started = time.perf_counter()

    work_dir = ensure_directory(work_dir)
    pages_dir = ensure_directory(os.path.join(work_dir, PAGES_DIR))
    metadata_dir = ensure_directory(os.path.join(work_dir, METADATA_DIR))
    metadata_path = os.path.join(metadata_dir, PAGES_JSON)

    # A stale file from an earlier run must not survive a failed run.
    _remove_if_exists(metadata_path)

    t0 = time.perf_counter()
    doc = open_pdf(pdf_path)
    open_s = time.perf_counter() - t0

    with doc:
        t0 = time.perf_counter()
        rendered = render_document(doc, pages_dir, dpi)
        render_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        native = extract_document(doc)
        extract_s = time.perf_counter() - t0

    pages = [_merge_page(r, n) for r, n in zip(rendered, native)]
    with_text = sum(1 for p in pages if p["has_text_layer"])

    document = {
        "schema_version": SCHEMA_VERSION,
        "document": {
            "source_pdf": os.path.basename(os.fspath(pdf_path)),
            "source_sha256": _sha256(pdf_path),
            "page_count": len(pages),
            "dpi": dpi,
            "pages_with_text_layer": with_text,
            "pages_without_text_layer": len(pages) - with_text,
            "coordinate_system": dict(COORDINATE_SYSTEM),
        },
        "pages": pages,
    }

    problems = validate_pages_document(document)
    if problems:    # a bug in this module, not a property of the PDF
        raise AssertionError("pages.json would be invalid: " + "; ".join(problems))

    if copy_original:
        _copy(pdf_path, os.path.join(work_dir, ORIGINAL_PDF))
    _write_json(metadata_path, document)

    return {
        "metadata_path": metadata_path,
        "work_dir": work_dir,
        "document": document,
        "timings": {
            "open_s": open_s,
            "render_s": render_s,
            "extract_s": extract_s,
            "total_s": time.perf_counter() - started,
        },
    }


def default_work_dir(pdf_path, base_dir):
    """<base_dir>/<safe-stem>-<first 8 hex of the file's SHA-256>.

    Deterministic for a given file, different for different files with the same name, and never
    built from the raw file name (only [A-Za-z0-9_-], at most 40 characters, is kept).
    """
    if not os.path.isfile(pdf_path):
        raise PdfNotFoundError(f"PDF not found: {os.fspath(pdf_path)}")
    stem = os.path.splitext(os.path.basename(os.fspath(pdf_path)))[0]
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:40] or "document"
    return os.path.join(os.fspath(base_dir), f"{safe}-{_sha256(pdf_path)[:8]}")


def has_current_pages(work_dir, pdf_path, dpi=DEFAULT_DPI):
    """True when `work_dir` already holds what process_pdf would write for this PDF at this DPI:
    a valid pages.json with the file's SHA-256, native text lines, and every page image."""
    work_dir = os.fspath(work_dir)
    try:
        with open(os.path.join(work_dir, METADATA_DIR, PAGES_JSON), encoding="utf-8") as fh:
            document = json.load(fh)
        if validate_pages_document(document):
            return False
        return (document["document"].get("source_sha256") == _sha256(pdf_path)
                and document["document"].get("dpi") == dpi
                and all("lines" in block for page in document["pages"] for block in page["blocks"])
                and all(os.path.isfile(os.path.join(work_dir, *page["image_path"].split("/")))
                        for page in document["pages"]))
    except (OSError, ValueError):
        return False


def validate_pages_document(document):
    """Structural check of a pages.json object. Returns a list of problems (empty = valid)."""
    problems = []

    def need(obj, key, types, where):
        value = obj.get(key) if isinstance(obj, dict) else None
        ok = isinstance(value, types) and not (isinstance(value, bool) and bool not in _as_tuple(types))
        if not ok:
            problems.append(f"{where}.{key} missing or wrong type")
        return value if ok else None

    number = (int, float)

    def check_bbox(bbox, where, width, height):
        if bbox is None:
            return
        if len(bbox) != 4 or not all(isinstance(v, number) and not isinstance(v, bool) for v in bbox):
            problems.append(f"{where}.bbox is not four numbers")
            return
        x0, y0, x1, y1 = bbox
        if not (x0 < x1 and y0 < y1):
            problems.append(f"{where}.bbox is empty or inverted")
        elif isinstance(width, number) and isinstance(height, number) and \
                not (0 <= x0 and 0 <= y0 and x1 <= width and y1 <= height):
            problems.append(f"{where}.bbox lies outside the page")

    if not isinstance(document, dict):
        return ["root is not an object"]

    info = need(document, "document", dict, "root") or {}
    pages = need(document, "pages", list, "root") or []
    need(info, "source_pdf", str, "document")
    count = need(info, "page_count", int, "document")
    if count is not None and count != len(pages):
        problems.append(f"document.page_count is {count} but there are {len(pages)} pages")
    if not pages:
        problems.append("pages is empty")

    for i, page in enumerate(pages):
        where = f"pages[{i}]"
        if need(page, "page_number", int, where) not in (None, i + 1):
            problems.append(f"{where}.page_number is not {i + 1}")
        for key in ("pdf_width", "pdf_height", "scale_x", "scale_y"):
            value = need(page, key, number, where)
            if value is not None and value <= 0:
                problems.append(f"{where}.{key} is not positive")
        for key in ("image_width", "image_height"):
            value = need(page, key, int, where)
            if value is not None and value <= 0:
                problems.append(f"{where}.{key} is not positive")
        need(page, "has_text_layer", bool, where)
        need(page, "text", str, where)
        need(page, "image_path", str, where)

        width, height = page.get("pdf_width"), page.get("pdf_height")
        for j, block in enumerate(need(page, "blocks", list, where) or []):
            bwhere = f"{where}.blocks[{j}]"
            need(block, "text", str, bwhere)
            need(block, "block_type", str, bwhere)
            check_bbox(need(block, "bbox", list, bwhere), bwhere, width, height)
            # "lines" was added in Phase 4; a pages.json written before that has none.
            if isinstance(block, dict) and "lines" in block:
                for k, line in enumerate(need(block, "lines", list, bwhere) or []):
                    lwhere = f"{bwhere}.lines[{k}]"
                    need(line, "text", str, lwhere)
                    check_bbox(need(line, "bbox", list, lwhere), lwhere, width, height)

    return problems


def _merge_page(rendered, native):
    return {
        "page_number": rendered["page_number"],
        "pdf_width": rendered["pdf_width"],
        "pdf_height": rendered["pdf_height"],
        "image_width": rendered["image_width"],
        "image_height": rendered["image_height"],
        "scale_x": rendered["scale_x"],
        "scale_y": rendered["scale_y"],
        "dpi": rendered["dpi"],
        "has_text_layer": native["has_text_layer"],
        "text": native["text"],
        "blocks": native["blocks"],
        # Relative to the working directory, with forward slashes, so the directory can be moved.
        "image_path": f"{PAGES_DIR}/{os.path.basename(rendered['image_path'])}",
    }


def _as_tuple(types):
    return types if isinstance(types, tuple) else (types,)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot replace: {path} ({exc})") from exc


def _copy(source, target):
    try:
        if os.path.abspath(os.fspath(source)) != os.path.abspath(target):
            shutil.copyfile(source, target)
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot copy the PDF to: {target} ({exc})") from exc


def _write_json(path, data):
    # Written to a temporary file first, so a reader never sees a half-written pages.json.
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot write metadata: {path} ({exc})") from exc

"""Phase 3 layout detection: page image -> typed regions -> layout.json.

    Phase 2 page images  ->  LayoutDetector  ->  RawDetection (model labels, image pixels)
                         ->  normalize_detections  ->  Region (canonical type, image + PDF boxes)
                         ->  <work_dir>/metadata/layout.json

Only the detector class knows the model library. Everything else works on RawDetection and Region, so
another detector can replace PP-DocLayoutV3 by subclassing LayoutDetector.

Regions are located and typed only: no OCR, no table structure, no formula recognition, no figure
understanding and no reading order.
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum

from .common import PdfPipelineError, ensure_directory
from .layout_eval import bbox_area, bbox_iou, is_valid_bbox
from .page_pipeline import METADATA_DIR, PAGES_JSON, _write_json, validate_pages_document

LAYOUT_SCHEMA_VERSION = 1
LAYOUT_JSON = "layout.json"
LAYOUT_DEBUG_DIR = "layout_debug"

# The model's own default (draw_threshold in its inference.yml).
DEFAULT_LAYOUT_THRESHOLD = 0.5

# Thresholds of find_overlap_issues (reporting only, nothing is removed).
DUPLICATE_IOU = 0.8
PAGE_SIZED_FRACTION = 0.9

COORDINATE_SYSTEM = {
    "bbox_format": "x0,y0,x1,y1",
    "origin": "top-left",
    "y_axis": "down",
    "image_bbox_unit": "px",
    "pdf_bbox_unit": "pt",
    "pdf_mapping": "pdf_x = image_x / scale_x; pdf_y = image_y / scale_y (scale_x, scale_y from pages.json)",
}


class LayoutError(PdfPipelineError):
    """Base class for layout detection failures."""


class LayoutInputError(LayoutError):
    """The Phase 2 artifacts (pages.json, page images) are missing or invalid."""


class LayoutModelError(LayoutError):
    """The layout model could not be loaded."""


class LayoutDetectionError(LayoutError):
    """Detection failed on a page."""

    def __init__(self, page_number, message):
        super().__init__(f"Page {page_number}: {message}")
        self.page_number = page_number


class RegionType(str, Enum):
    """Canonical region classes of the pipeline."""
    TITLE = "TITLE"
    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    EQUATION = "EQUATION"
    CAPTION = "CAPTION"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    PAGE_NUMBER = "PAGE_NUMBER"
    LIST = "LIST"
    OTHER = "OTHER"


# PP-DocLayoutV3 label (label_list of its inference.yml, all 25) -> canonical class.
# The model label is always kept in Region.source_label; a label missing here becomes OTHER.
PP_DOCLAYOUT_V3_LABEL_MAP = {
    "doc_title": RegionType.TITLE,              # title of the document
    "paragraph_title": RegionType.TITLE,        # section / subsection heading
    "text": RegionType.TEXT,                    # body paragraph
    "abstract": RegionType.TEXT,
    "reference": RegionType.TEXT,               # reference section
    "reference_content": RegionType.TEXT,       # one bibliography entry
    "algorithm": RegionType.TEXT,               # pseudo-code block
    "vertical_text": RegionType.TEXT,
    "table": RegionType.TABLE,
    "image": RegionType.FIGURE,
    "chart": RegionType.FIGURE,
    "display_formula": RegionType.EQUATION,
    "inline_formula": RegionType.EQUATION,
    "figure_title": RegionType.CAPTION,         # caption of a figure, table or chart
    "vision_footnote": RegionType.CAPTION,      # note attached to a figure or table
    "header": RegionType.HEADER,
    "header_image": RegionType.HEADER,
    "footer": RegionType.FOOTER,
    "footer_image": RegionType.FOOTER,
    "number": RegionType.PAGE_NUMBER,
    "content": RegionType.LIST,                 # table of contents
    "formula_number": RegionType.OTHER,         # the "(3)" tag beside an equation, not a formula
    "footnote": RegionType.OTHER,
    "aside_text": RegionType.OTHER,             # margin text
    "seal": RegionType.OTHER,                   # stamp
}


@dataclass(frozen=True)
class RawDetection:
    """One box as the model returned it: its own label, image pixels [x0, y0, x1, y1]."""
    label: str
    score: float
    bbox: tuple


@dataclass(frozen=True)
class Region:
    """One typed region of a page, in both coordinate systems ([x0, y0, x1, y1], origin top-left)."""
    region_id: str
    page_number: int
    type: RegionType
    score: float
    image_bbox: tuple       # pixels of the Phase 2 page image
    pdf_bbox: tuple         # PDF points, same system as pages.json
    source_label: str
    source_model: str

    def to_dict(self):
        data = asdict(self)
        data["type"] = self.type.value
        data["image_bbox"] = list(self.image_bbox)
        data["pdf_bbox"] = list(self.pdf_bbox)
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            region_id=data["region_id"],
            page_number=data["page_number"],
            type=RegionType(data["type"]),
            score=data["score"],
            image_bbox=tuple(data["image_bbox"]),
            pdf_bbox=tuple(data["pdf_bbox"]),
            source_label=data["source_label"],
            source_model=data["source_model"],
        )


class LayoutDetector:
    """Interface of a layout model. Subclasses set `name` and `label_map` and implement
    `_load_model` and `_detect`; nothing outside the subclass touches the model library."""

    name = ""
    label_map = {}

    def __init__(self):
        self._model = None
        self.load_count = 0         # how many times the model was really loaded
        self.load_seconds = 0.0

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        """Load the model once; later calls do nothing. Raises LayoutModelError."""
        if self.loaded:
            return self
        started = time.perf_counter()
        try:
            self._model = self._load_model()
        except LayoutError:
            raise
        except Exception as exc:
            raise LayoutModelError(f"Cannot load layout model {self.name}: {exc}") from exc
        self.load_count += 1
        self.load_seconds = time.perf_counter() - started
        return self

    def detect(self, image_path, threshold=DEFAULT_LAYOUT_THRESHOLD):
        """Detections of one page image with score >= threshold, as RawDetection, in model order."""
        check_threshold(threshold)
        self.load()
        return self._detect(image_path, threshold)

    def describe(self):
        """The "model" object of layout.json (without the threshold)."""
        return {"name": self.name}

    def _load_model(self):
        raise NotImplementedError

    def _detect(self, image_path, threshold):
        raise NotImplementedError


class PPDocLayoutV3Detector(LayoutDetector):
    """PP-DocLayoutV3 through PaddleOCR's standalone LayoutDetection module (not PP-StructureV3)."""

    name = "PP-DocLayoutV3"
    label_map = PP_DOCLAYOUT_V3_LABEL_MAP

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def describe(self):
        return {
            "name": self.name,
            "provider": "PaddleOCR",
            "version": _package_version("paddleocr"),
            "framework": "paddlepaddle " + _package_version("paddlepaddle"),
            "device": self.device,
        }

    def _load_model(self):
        # The model is cached under ~/.paddlex after the first download; skip PaddleX's start-up
        # probe of every model host (it only chooses where a missing model is downloaded from).
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # Imported here: importing paddle takes seconds and nothing else in the pipeline needs it.
        from paddleocr import LayoutDetection
        return LayoutDetection(model_name=self.name, device=self.device)

    def _detect(self, image_path, threshold):
        results = list(self._model.predict(os.fspath(image_path), batch_size=1, threshold=threshold))
        return [RawDetection(label=str(box["label"]),
                             score=float(box["score"]),
                             bbox=tuple(float(v) for v in box["coordinate"]))
                for box in results[0]["boxes"]]


def check_threshold(threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"layout threshold must be a number between 0 and 1, got {threshold!r}")
    return float(threshold)


def normalize_label(source_label, label_map):
    """Canonical class of a model label; OTHER for a label the map does not know."""
    return label_map.get(source_label, RegionType.OTHER)


def clip_bbox(bbox, width, height):
    """Clip [x0, y0, x1, y1] to [0, width] x [0, height]. None if it is not a box or nothing is left."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    clipped = (max(x0, 0.0), max(y0, 0.0), min(x1, float(width)), min(y1, float(height)))
    return clipped if is_valid_bbox(clipped) else None


def image_bbox_to_pdf(bbox, scale_x, scale_y):
    """Map an image-pixel box to PDF points: the inverse of render.pdf_bbox_to_pixels."""
    x0, y0, x1, y1 = bbox
    return (x0 / scale_x, y0 / scale_y, x1 / scale_x, y1 / scale_y)


def normalize_detections(detections, page, threshold, label_map, source_model):
    """Turn the raw detections of one page into Regions.

    `page` is the page entry of pages.json (page_number, image and PDF size, scale_x, scale_y).
    Detections below `threshold` are left out. Boxes are clipped to the image; a box that is malformed
    or empty after clipping is discarded. Returns (regions, discarded_count); regions keep model order.
    """
    threshold = check_threshold(threshold)
    regions, discarded = [], 0
    for detection in detections:
        score = detection.score
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            discarded += 1
            continue
        if score < threshold:
            continue

        image_bbox = clip_bbox(detection.bbox, page["image_width"], page["image_height"])
        pdf_bbox = image_bbox and clip_bbox(image_bbox_to_pdf(image_bbox, page["scale_x"], page["scale_y"]),
                                            page["pdf_width"], page["pdf_height"])
        if not pdf_bbox:
            discarded += 1
            continue

        regions.append(Region(
            region_id=f"p{page['page_number']:03d}_r{len(regions) + 1:03d}",
            page_number=page["page_number"],
            type=normalize_label(detection.label, label_map),
            score=float(score),
            image_bbox=image_bbox,
            pdf_bbox=pdf_bbox,
            source_label=detection.label,
            source_model=source_model,
        ))
    return regions, discarded


def find_overlap_issues(regions, image_width, image_height):
    """Report suspicious boxes of one page without changing anything.

    Returns dicts {"kind", "region_ids", "value"}:
      "duplicate"   two regions of the same type with IoU >= DUPLICATE_IOU (value = IoU)
      "page_sized"  one region covering >= PAGE_SIZED_FRACTION of the page image (value = fraction)
    """
    issues = []
    page_area = float(image_width * image_height)
    for i, region in enumerate(regions):
        fraction = bbox_area(region.image_bbox) / page_area
        if fraction >= PAGE_SIZED_FRACTION:
            issues.append({"kind": "page_sized", "region_ids": [region.region_id], "value": fraction})
        for other in regions[i + 1:]:
            if region.type == other.type:
                iou = bbox_iou(region.image_bbox, other.image_bbox)
                if iou >= DUPLICATE_IOU:
                    issues.append({"kind": "duplicate", "region_ids": [region.region_id, other.region_id],
                                   "value": iou})
    return issues


def load_pages_document(work_dir):
    """Read and check <work_dir>/metadata/pages.json. Raises LayoutInputError."""
    path = os.path.join(os.fspath(work_dir), METADATA_DIR, PAGES_JSON)
    if not os.path.isfile(path):
        raise LayoutInputError(f"No Phase 2 metadata found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
    except (OSError, ValueError) as exc:
        raise LayoutInputError(f"Cannot read {path}: {exc}") from exc
    problems = validate_pages_document(document)
    if problems:
        raise LayoutInputError(f"Invalid {path}: " + "; ".join(problems[:5]))
    return document


def detect_document_layout(work_dir, detector, threshold=DEFAULT_LAYOUT_THRESHOLD, visualize=False):
    """Run `detector` on every page image of a Phase 2 working directory and write metadata/layout.json.

    The model is loaded once, then the pages are processed in order. With visualize=True a copy of each
    page with the boxes drawn is saved as <work_dir>/layout_debug/page_NNN_layout.png.

    Returns {"layout_path", "work_dir", "document", "timings", "issues", "visualizations"}. `timings` has
    model_init_s (0.0 if the detector was already loaded), inference_s, page_inference_s (list) and
    total_s; like `issues` (see find_overlap_issues) it is not written to the file.
    Raises a LayoutError; no layout.json is left behind in that case.
    """
    started = time.perf_counter()
    threshold = check_threshold(threshold)
    work_dir = os.path.abspath(os.fspath(work_dir))
    pages_document = load_pages_document(work_dir)
    layout_path = os.path.join(work_dir, METADATA_DIR, LAYOUT_JSON)
    _remove_stale(layout_path)

    was_loaded = detector.loaded
    detector.load()
    model_init_s = 0.0 if was_loaded else detector.load_seconds

    pages, issues, page_seconds, images = [], [], [], []
    for page in pages_document["pages"]:
        page_number = page["page_number"]
        image_path = os.path.join(work_dir, *page["image_path"].split("/"))
        if not os.path.isfile(image_path):
            raise LayoutInputError(f"Page {page_number}: page image not found: {image_path}")

        t0 = time.perf_counter()
        try:
            detections = detector.detect(image_path, threshold)
        except LayoutError:
            raise
        except Exception as exc:
            raise LayoutDetectionError(page_number, f"layout detection failed ({exc})") from exc
        page_seconds.append(time.perf_counter() - t0)

        regions, discarded = normalize_detections(detections, page, threshold, detector.label_map, detector.name)
        issues.extend(find_overlap_issues(regions, page["image_width"], page["image_height"]))
        images.append((image_path, regions))
        pages.append({
            "page_number": page_number,
            "image_path": page["image_path"],
            "image_width": page["image_width"],
            "image_height": page["image_height"],
            "pdf_width": page["pdf_width"],
            "pdf_height": page["pdf_height"],
            "discarded_detections": discarded,
            "regions": [region.to_dict() for region in regions],
        })

    document = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "model": {**detector.describe(), "threshold": threshold},
        "source": {
            "pages_json": f"{METADATA_DIR}/{PAGES_JSON}",
            "source_pdf": pages_document["document"].get("source_pdf"),
            "source_sha256": pages_document["document"].get("source_sha256"),
        },
        "coordinate_system": dict(COORDINATE_SYSTEM),
        "pages": pages,
    }
    problems = validate_layout_document(document)
    if problems:    # a bug in this module, not a property of the document
        raise AssertionError("layout.json would be invalid: " + "; ".join(problems))

    visualizations = []
    if visualize:
        from .layout_visualize import draw_layout, layout_image_name
        debug_dir = ensure_directory(os.path.join(work_dir, LAYOUT_DEBUG_DIR))
        for (image_path, regions), page in zip(images, pages):
            target = os.path.join(debug_dir, layout_image_name(page["page_number"]))
            draw_layout(image_path, regions, target)
            visualizations.append(target)

    _write_json(layout_path, document)
    return {
        "layout_path": layout_path,
        "work_dir": work_dir,
        "document": document,
        "timings": {
            "model_init_s": model_init_s,
            "inference_s": sum(page_seconds),
            "page_inference_s": page_seconds,
            "total_s": time.perf_counter() - started,
        },
        "issues": issues,
        "visualizations": visualizations,
    }


def count_region_types(document):
    """{canonical type: count} over all pages of a layout.json object, in RegionType order."""
    counts = {}
    for page in document["pages"]:
        for region in page["regions"]:
            counts[region["type"]] = counts.get(region["type"], 0) + 1
    return {t.value: counts[t.value] for t in RegionType if t.value in counts}


def validate_layout_document(document):
    """Structural check of a layout.json object. Returns a list of problems (empty = valid)."""
    if not isinstance(document, dict):
        return ["root is not an object"]
    problems = []
    number = (int, float)

    model = document.get("model")
    if not isinstance(model, dict):
        problems.append("model missing")
    else:
        if not isinstance(model.get("name"), str) or not model.get("name"):
            problems.append("model.name missing")
        if not isinstance(model.get("threshold"), number) or isinstance(model.get("threshold"), bool):
            problems.append("model.threshold missing")

    pages = document.get("pages")
    if not isinstance(pages, list) or not pages:
        return problems + ["pages missing or empty"]

    types = {t.value for t in RegionType}
    seen_ids = set()
    for i, page in enumerate(pages):
        where = f"pages[{i}]"
        if not isinstance(page, dict):
            problems.append(f"{where} is not an object")
            continue
        if page.get("page_number") != i + 1:
            problems.append(f"{where}.page_number is not {i + 1}")
        sizes = [page.get(k) for k in ("image_width", "image_height", "pdf_width", "pdf_height")]
        if not all(isinstance(v, number) and not isinstance(v, bool) and v > 0 for v in sizes):
            problems.append(f"{where} has no valid page size")
            continue
        if not isinstance(page.get("regions"), list):
            problems.append(f"{where}.regions missing")
            continue

        for j, region in enumerate(page["regions"]):
            rwhere = f"{where}.regions[{j}]"
            if not isinstance(region, dict):
                problems.append(f"{rwhere} is not an object")
                continue
            region_id = region.get("region_id")
            if not isinstance(region_id, str) or not region_id or region_id in seen_ids:
                problems.append(f"{rwhere}.region_id missing or repeated")
            seen_ids.add(region_id)
            if region.get("page_number") != page.get("page_number"):
                problems.append(f"{rwhere}.page_number differs from its page")
            if region.get("type") not in types:
                problems.append(f"{rwhere}.type is not a canonical class")
            score = region.get("score")
            if isinstance(score, bool) or not isinstance(score, number) or not 0.0 <= score <= 1.0:
                problems.append(f"{rwhere}.score is not between 0 and 1")
            for key in ("source_label", "source_model"):
                if not isinstance(region.get(key), str) or not region.get(key):
                    problems.append(f"{rwhere}.{key} missing")
            for key, width, height in (("image_bbox", sizes[0], sizes[1]), ("pdf_bbox", sizes[2], sizes[3])):
                bbox = region.get(key)
                if not is_valid_bbox(bbox):
                    problems.append(f"{rwhere}.{key} is not a valid box")
                elif not (0 <= bbox[0] and 0 <= bbox[1] and bbox[2] <= width and bbox[3] <= height):
                    problems.append(f"{rwhere}.{key} lies outside the page")
    return problems


def _package_version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _remove_stale(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise LayoutInputError(f"Cannot replace: {path} ({exc})") from exc

"""Phase 4 selective OCR: text of every layout region, from the PDF where possible, else from OCR.

    Phase 2 pages.json (native lines)  +  Phase 3 layout.json (typed regions)
        ->  route_region      native text inside the region?  yes: keep it   no: OCR the region
        ->  OCREngine         crop of the Phase 2 page image -> lines with confidence and geometry
        ->  <work_dir>/metadata/ocr.json      every text marked "native" or "ocr"

Only the engine class knows the OCR library. Everything else works on OcrLine and plain dicts, so
another engine can replace PaddleOCR by subclassing OCREngine.

Text only: no table rows/columns/cells, no formula recognition, no figure or chart understanding and
no reading order between regions.
"""
import glob
import json
import math
import os
import time
from dataclasses import dataclass

from .common import OutputDirectoryError, PdfPipelineError, ensure_directory
from .layout import (
    COORDINATE_SYSTEM,
    LAYOUT_JSON,
    RegionType,
    _package_version,
    clip_bbox,
    image_bbox_to_pdf,
    load_pages_document,
    validate_layout_document,
)
from .layout_eval import bbox_area, is_valid_bbox
from .page_pipeline import METADATA_DIR, PAGES_JSON, _write_json

OCR_SCHEMA_VERSION = 1
OCR_JSON = "ocr.json"
OCR_CROPS_DIR = "ocr_crops"
OCR_CROP_PATTERN = "page_{:03d}_region_{:03d}.png"

# Region classes that carry running text: native text if the PDF has it, otherwise OCR.
# OTHER is footnotes, margin text and equation numbers (see PP_DOCLAYOUT_V3_LABEL_MAP).
TEXT_REGION_TYPES = frozenset({
    RegionType.TEXT, RegionType.TITLE, RegionType.CAPTION, RegionType.HEADER, RegionType.FOOTER,
    RegionType.PAGE_NUMBER, RegionType.LIST, RegionType.OTHER,
})
# Native text is kept as raw text; OCR (raw lines, no cells) only on request.
RAW_TEXT_REGION_TYPES = frozenset({RegionType.TABLE})
# EQUATION and FIGURE get no text in this phase.

# A native line belongs to a region when at least this share of the LINE's area lies inside it.
LINE_INSIDE_MIN = 0.5
# Native text with more than this share of undecodable characters is not used.
MAX_UNDECODABLE_SHARE = 0.1

DEFAULT_OCR_PADDING = 8     # pixels of page added around a region before OCR
MIN_CROP_SIDE = 6           # pixels; a smaller region cannot hold readable text
BLANK_CONTRAST = 16         # grey levels (0-255); a crop with less contrast is empty paper

TEXT_SOURCE_NATIVE = "native"
TEXT_SOURCE_OCR = "ocr"
TEXT_SOURCE_NONE = "none"

ROUTE_NATIVE = "native"
ROUTE_OCR = "ocr"
ROUTE_SKIP = "skip"


class OcrError(PdfPipelineError):
    """Base class for OCR failures."""


class OcrInputError(OcrError):
    """The Phase 2 / Phase 3 artifacts are missing, invalid or do not belong together."""


class OcrModelError(OcrError):
    """The OCR models could not be loaded."""


class OcrCropError(OcrError):
    """A region cannot be cut out of the page image."""


@dataclass(frozen=True)
class OcrLine:
    """One recognized line of text. bbox [x0, y0, x1, y1] and polygon (four [x, y] corners) are
    pixels of the image the engine was given."""
    text: str
    confidence: float
    bbox: tuple
    polygon: tuple = ()


@dataclass(frozen=True)
class Routing:
    """Where the text of a region comes from: decision is ROUTE_NATIVE, ROUTE_OCR or ROUTE_SKIP."""
    decision: str
    reason: str
    native_text: str = None
    native_blocks: tuple = ()       # block_index of the pages.json blocks that gave the text


class OCREngine:
    """Interface of an OCR engine. Subclasses set `name` and implement `_load_model` and
    `_recognize`; nothing outside the subclass touches the OCR library."""

    name = ""

    def __init__(self):
        self._model = None
        self.load_count = 0         # how many times the models were really loaded
        self.load_seconds = 0.0

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        """Load the models once; later calls do nothing. Raises OcrModelError."""
        if self.loaded:
            return self
        started = time.perf_counter()
        try:
            self._model = self._load_model()
        except OcrError:
            raise
        except Exception as exc:
            raise OcrModelError(f"Cannot load OCR engine {self.name}: {exc}") from exc
        self.load_count += 1
        self.load_seconds = time.perf_counter() - started
        return self

    def recognize_page(self, page_image):
        """OCR a whole page image (PIL). Returns OcrLine objects in page pixels."""
        self.load()
        return _checked_lines(self._recognize(page_image), page_image.width, page_image.height)

    def recognize_region(self, page_image, image_bbox, padding=DEFAULT_OCR_PADDING):
        """OCR one region of a page image (PIL).

        Returns {"lines": [OcrLine in PAGE pixels], "crop_bbox": [x0, y0, x1, y1], "crop": PIL image,
        "blank": bool}. A crop without contrast is not sent to the engine ("blank": True, no lines).
        Lines centred in the padding, outside the region itself, are text of a neighbour and left out.
        Raises OcrCropError for a region that cannot be cut out.
        """
        box = crop_box(image_bbox, page_image.width, page_image.height, padding)
        crop = page_image.crop(box)
        result = {"lines": [], "crop_bbox": list(box), "crop": crop, "blank": is_blank(crop)}
        if result["blank"]:
            return result

        self.load()
        x0, y0, x1, y1 = clip_bbox(image_bbox, page_image.width, page_image.height)
        for line in _checked_lines(self._recognize(crop), crop.width, crop.height):
            moved = OcrLine(text=line.text, confidence=line.confidence,
                            bbox=(line.bbox[0] + box[0], line.bbox[1] + box[1],
                                  line.bbox[2] + box[0], line.bbox[3] + box[1]),
                            polygon=tuple((x + box[0], y + box[1]) for x, y in line.polygon))
            centre_x, centre_y = (moved.bbox[0] + moved.bbox[2]) / 2, (moved.bbox[1] + moved.bbox[3]) / 2
            if x0 <= centre_x <= x1 and y0 <= centre_y <= y1:
                result["lines"].append(moved)
        return result

    def describe(self):
        """The "engine" object of ocr.json."""
        return {"name": self.name}

    def _load_model(self):
        raise NotImplementedError

    def _recognize(self, image):
        """Lines of a PIL RGB image as OcrLine, in pixels of that image, in any order."""
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """PP-OCRv5 through PaddleOCR's standalone TextDetection and TextRecognition modules.

    The mobile models are used: they are small and fast enough on CPU. The recognition model is the
    English one; pass another `recognition_model` (for example "PP-OCRv5_mobile_rec") for other scripts.
    """

    name = "PaddleOCR"
    DETECTION_MODEL = "PP-OCRv5_mobile_det"
    RECOGNITION_MODEL = "en_PP-OCRv5_mobile_rec"

    def __init__(self, device="cpu", detection_model=DETECTION_MODEL, recognition_model=RECOGNITION_MODEL,
                 batch_size=8):
        super().__init__()
        self.device = device
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.batch_size = batch_size

    def describe(self):
        return {
            "name": self.name,
            "version": _package_version("paddleocr"),
            "framework": "paddlepaddle " + _package_version("paddlepaddle"),
            "device": self.device,
            "detection_model": self.detection_model,
            "recognition_model": self.recognition_model,
        }

    def _load_model(self):
        # Same reasons as in PPDocLayoutV3Detector._load_model.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextDetection, TextRecognition
        # oneDNN is off for detection: with paddlepaddle 3.3.1 on CPU the detection model fails in it
        # ("ConvertPirAttribute2RuntimeAttribute not support"). Recognition works and is 4x faster with it.
        return {
            "detection": TextDetection(model_name=self.detection_model, device=self.device, enable_mkldnn=False),
            "recognition": TextRecognition(model_name=self.recognition_model, device=self.device,
                                           enable_mkldnn=True),
        }

    def _recognize(self, image):
        import numpy as np
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])     # BGR, as OpenCV
        height, width = pixels.shape[:2]

        detected = list(self._model["detection"].predict(pixels, batch_size=1))[0]["dt_polys"]
        boxes, polygons, crops = [], [], []
        for polygon in detected:
            points = [(float(x), float(y)) for x, y in polygon]
            # Pages are rendered upright, so the bounding rectangle of the polygon is the line.
            x0 = max(math.floor(min(x for x, _ in points)), 0)
            y0 = max(math.floor(min(y for _, y in points)), 0)
            x1 = min(math.ceil(max(x for x, _ in points)), width)
            y1 = min(math.ceil(max(y for _, y in points)), height)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append((x0, y0, x1, y1))
            polygons.append(tuple(points))
            crops.append(pixels[y0:y1, x0:x1])
        if not crops:
            return []

        recognized = list(self._model["recognition"].predict(crops, batch_size=self.batch_size))
        return [OcrLine(text=str(result["rec_text"]), confidence=float(result["rec_score"]), bbox=box,
                        polygon=polygon)
                for result, box, polygon in zip(recognized, boxes, polygons)]


def check_padding(padding):
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError(f"OCR padding must be a whole number of pixels >= 0, got {padding!r}")
    return padding


def crop_box(image_bbox, image_width, image_height, padding=DEFAULT_OCR_PADDING):
    """Whole-pixel box (x0, y0, x1, y1) to cut out for a region: the part of `image_bbox` that lies on
    the image, grown by `padding` pixels but never beyond the image.

    Raises OcrCropError for a box that is malformed, lies outside the image, or is smaller than
    MIN_CROP_SIDE pixels on a side. Raises ValueError for a bad `padding`.
    """
    check_padding(padding)
    clipped = clip_bbox(image_bbox, image_width, image_height)
    if clipped is None:
        raise OcrCropError(f"region box is invalid or outside the page image: {image_bbox!r}")
    x0, y0 = math.floor(clipped[0]), math.floor(clipped[1])
    x1, y1 = math.ceil(clipped[2]), math.ceil(clipped[3])
    if x1 - x0 < MIN_CROP_SIDE or y1 - y0 < MIN_CROP_SIDE:
        raise OcrCropError(f"region is too small for OCR: {x1 - x0} x {y1 - y0} px")
    return (max(x0 - padding, 0), max(y0 - padding, 0),
            min(x1 + padding, int(image_width)), min(y1 + padding, int(image_height)))


def is_blank(image):
    """True for a PIL image without contrast (empty paper, a uniform fill)."""
    low, high = image.convert("L").getextrema()
    return high - low < BLANK_CONTRAST


def clean_line(text):
    """Conservative cleanup of one line: outer whitespace removed, inner runs of whitespace -> one space."""
    return " ".join(text.split())


def join_lines(lines):
    """Text of a region from its lines, given as (bbox, text) pairs in any order.

    Lines are put in rows from top to bottom; two lines are in the same row when each one's vertical
    centre lies inside the other. Fragments of a row are joined left to right with one space, rows
    with a line break. Every fragment is cleaned with clean_line; nothing is corrected, re-hyphenated
    or rewritten.
    """
    cleaned = [(bbox, clean_line(text)) for bbox, text in lines]
    rows = group_rows([item for item in cleaned if item[1]])
    return "\n".join(" ".join(text for _, text in row) for row in rows)


def group_rows(items):
    """Put items whose first element is a box [x0, y0, x1, y1] into rows: top to bottom, each row left
    to right. Two items are in the same row when each one's vertical centre lies inside the other."""
    rows = []
    for item in sorted(items, key=lambda item: ((item[0][1] + item[0][3]) / 2, item[0][0])):
        bbox = item[0]
        centre = (bbox[1] + bbox[3]) / 2
        first = rows[-1][0][0] if rows else None
        if first and first[1] <= centre <= first[3] and bbox[1] <= (first[1] + first[3]) / 2 <= bbox[3]:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [sorted(row, key=lambda item: item[0][0]) for row in rows]


def is_usable_text(text):
    """True when native text can be used as it is: at least one letter or digit, and at most
    MAX_UNDECODABLE_SHARE of its characters undecodable (U+FFFD, private use, control)."""
    visible = [ch for ch in text or "" if not ch.isspace()]
    if not any(ch.isalnum() for ch in visible):
        return False
    undecodable = sum(1 for ch in visible if ch == "�" or "" <= ch <= "" or ord(ch) < 32)
    return undecodable / len(visible) <= MAX_UNDECODABLE_SHARE


def native_lines_in_region(pdf_bbox, page):
    """Native lines of a pages.json page that belong to the box `pdf_bbox` (PDF points).

    A line belongs to the region when at least LINE_INSIDE_MIN of the line's own area is inside it.
    Returns [(line bbox, line text, block_index)] in pages.json order.
    """
    found = []
    for block in page["blocks"]:
        for line in block["lines"]:
            bbox = line["bbox"]
            width = min(bbox[2], pdf_bbox[2]) - max(bbox[0], pdf_bbox[0])
            height = min(bbox[3], pdf_bbox[3]) - max(bbox[1], pdf_bbox[1])
            if width > 0 and height > 0 and width * height / bbox_area(bbox) >= LINE_INSIDE_MIN:
                found.append((bbox, line["text"], block["block_index"]))
    return found


def route_region(region, page, ocr_tables=False):
    """Decide where the text of one layout region comes from. Returns a Routing.

    `region` is a region of layout.json, `page` the page of pages.json it lies on.

      EQUATION, FIGURE                 -> skip   (no text in this phase)
      usable native text in the region -> native (the text is in Routing.native_text)
      TABLE without native text        -> skip, or OCR with ocr_tables=True
      any other class without it       -> OCR

    The page-level has_text_layer flag only words the reason; the decision is made per region.
    """
    region_type = RegionType(region["type"])
    if region_type not in TEXT_REGION_TYPES and region_type not in RAW_TEXT_REGION_TYPES:
        return Routing(ROUTE_SKIP, "region_type_has_no_text")

    lines = native_lines_in_region(region["pdf_bbox"], page)
    text = join_lines([(bbox, text) for bbox, text, _ in lines])
    if is_usable_text(text):
        return Routing(ROUTE_NATIVE, "native_text_in_region", text, tuple(sorted({index for _, _, index in lines})))

    if region_type in RAW_TEXT_REGION_TYPES and not ocr_tables:
        return Routing(ROUTE_SKIP, "table_ocr_not_requested")
    if text:
        return Routing(ROUTE_OCR, "native_text_unusable")
    if not page["has_text_layer"]:
        return Routing(ROUTE_OCR, "page_has_no_text_layer")
    return Routing(ROUTE_OCR, "no_native_text_in_region")


def should_ocr(region, page, ocr_tables=False):
    """True only when the region needs OCR (see route_region)."""
    return route_region(region, page, ocr_tables).decision == ROUTE_OCR


def region_confidence(lines):
    """Confidence of a region: mean of its line confidences weighted by their number of characters.
    None without text."""
    weights = [len(clean_line(line.text).replace(" ", "")) for line in lines]
    if not sum(weights):
        return None
    if len(lines) == 1:     # kept exactly, without rounding noise from the weighting
        return lines[0].confidence
    return sum(line.confidence * weight for line, weight in zip(lines, weights)) / sum(weights)


def load_layout_document(work_dir, pages_document):
    """Read <work_dir>/metadata/layout.json and check it belongs to `pages_document`. Raises OcrInputError."""
    path = os.path.join(os.fspath(work_dir), METADATA_DIR, LAYOUT_JSON)
    if not os.path.isfile(path):
        raise OcrInputError(f"No Phase 3 layout found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
    except (OSError, ValueError) as exc:
        raise OcrInputError(f"Cannot read {path}: {exc}") from exc
    problems = validate_layout_document(document)
    if problems:
        raise OcrInputError(f"Invalid {path}: " + "; ".join(problems[:5]))

    pages = pages_document["pages"]
    same_pages = len(document["pages"]) == len(pages) and all(
        (a["image_width"], a["image_height"]) == (b["image_width"], b["image_height"])
        for a, b in zip(document["pages"], pages))
    same_source = document.get("source", {}).get("source_sha256") == pages_document["document"].get("source_sha256")
    if not (same_pages and same_source):
        raise OcrInputError(f"{path} was made from other page images than pages.json; run Phase 3 again")
    return document


def run_document_ocr(work_dir, engine, padding=DEFAULT_OCR_PADDING, ocr_tables=False, save_crops=False):
    """Give every layout region of a working directory its text and write metadata/ocr.json.

    Needs the Phase 2 and Phase 3 artifacts in `work_dir`; neither is modified. Regions are routed
    first; the engine is loaded once, and only if at least one region needs OCR. A region whose OCR
    fails is kept with text null and an "error"; the run goes on. With save_crops=True the image sent
    to the engine is saved as <work_dir>/ocr_crops/page_NNN_region_NNN.png for every OCR'd region.

    Returns {"ocr_path", "work_dir", "document", "timings", "crops"}. `timings` has engine_init_s
    (0.0 if the engine was already loaded or not needed), ocr_s, region_ocr_s ({region_id: seconds})
    and total_s; it is not written to the file.
    Raises a PdfPipelineError for unusable input or an engine that does not load; no ocr.json is left
    behind in that case.
    """
    started = time.perf_counter()
    check_padding(padding)
    work_dir = os.path.abspath(os.fspath(work_dir))
    pages_document = load_pages_document(work_dir)
    layout_document = load_layout_document(work_dir, pages_document)
    if any("lines" not in block for page in pages_document["pages"] for block in page["blocks"]):
        raise OcrInputError("pages.json has no native text lines (written before Phase 4); run Phase 2 again")
    ocr_path = os.path.join(work_dir, METADATA_DIR, OCR_JSON)
    _remove_stale(ocr_path)

    routed = [[(region, route_region(region, page, ocr_tables)) for region in layout_page["regions"]]
              for page, layout_page in zip(pages_document["pages"], layout_document["pages"])]
    needs_ocr = any(routing.decision == ROUTE_OCR for page_routes in routed for _, routing in page_routes)

    engine_init_s = 0.0
    if needs_ocr:
        was_loaded = engine.loaded
        engine.load()
        engine_init_s = 0.0 if was_loaded else engine.load_seconds
    crops_dir = _prepare_crops_dir(work_dir) if save_crops else None

    pages, region_seconds, crops = [], {}, []
    for page, page_routes in zip(pages_document["pages"], routed):
        page_image = None
        regions = []
        for index, (region, routing) in enumerate(page_routes, 1):
            entry = _region_entry(region, routing)
            if routing.decision == ROUTE_NATIVE:
                entry.update(text=routing.native_text, text_source=TEXT_SOURCE_NATIVE,
                             native_blocks=list(routing.native_blocks))
            elif routing.decision == ROUTE_OCR:
                if page_image is None:
                    page_image = _open_page_image(work_dir, page)
                t0 = time.perf_counter()
                result = None
                try:
                    result = engine.recognize_region(page_image, region["image_bbox"], padding)
                    entry.update(_ocr_fields(result, page, engine.name))
                except Exception as exc:    # one unreadable region must not lose the rest of the document
                    entry.update(_region_entry(region, routing), error=f"{type(exc).__name__}: {exc}")
                region_seconds[region["region_id"]] = time.perf_counter() - t0
                if crops_dir and result:
                    target = os.path.join(crops_dir, OCR_CROP_PATTERN.format(page["page_number"], index))
                    try:
                        result["crop"].save(target, format="PNG")
                    except OSError as exc:
                        raise OutputDirectoryError(f"Cannot write OCR crop: {target} ({exc})") from exc
                    crops.append(target)
            regions.append(entry)
        pages.append({"page_number": page["page_number"], "has_text_layer": page["has_text_layer"],
                      "regions": regions})

    document = {
        "schema_version": OCR_SCHEMA_VERSION,
        "engine": engine.describe(),
        "settings": {"padding_px": padding, "ocr_tables": bool(ocr_tables), "line_inside_min": LINE_INSIDE_MIN},
        "source": {
            "pages_json": f"{METADATA_DIR}/{PAGES_JSON}",
            "layout_json": f"{METADATA_DIR}/{LAYOUT_JSON}",
            "source_pdf": pages_document["document"].get("source_pdf"),
            "source_sha256": pages_document["document"].get("source_sha256"),
        },
        "coordinate_system": dict(COORDINATE_SYSTEM),
        "pages": pages,
    }
    document["summary"] = summarize_ocr_document(document)
    problems = validate_ocr_document(document)
    if problems:    # a bug in this module, not a property of the document
        raise AssertionError("ocr.json would be invalid: " + "; ".join(problems))

    _write_json(ocr_path, document)
    return {
        "ocr_path": ocr_path,
        "work_dir": work_dir,
        "document": document,
        "timings": {
            "engine_init_s": engine_init_s,
            "ocr_s": sum(region_seconds.values()),
            "region_ocr_s": region_seconds,
            "total_s": time.perf_counter() - started,
        },
        "crops": crops,
    }


def summarize_ocr_document(document):
    """Counts over all regions of an ocr.json object.

    text_regions = native_regions + ocr_regions + ocr_failures; ocr_regions includes ocr_empty (OCR
    ran and found no text). mean_ocr_confidence is the mean over OCR'd regions with text, or None.
    """
    regions = [region for page in document["pages"] for region in page["regions"]]
    ocr = [r for r in regions if r["text_source"] == TEXT_SOURCE_OCR]
    confidences = [r["confidence"] for r in ocr if r["confidence"] is not None]
    failures = sum(1 for r in regions if "error" in r)
    native = sum(1 for r in regions if r["text_source"] == TEXT_SOURCE_NATIVE)
    return {
        "pages": len(document["pages"]),
        "regions": len(regions),
        "text_regions": native + len(ocr) + failures,
        "native_regions": native,
        "ocr_regions": len(ocr),
        "ocr_empty": sum(1 for r in ocr if not r["text"]),
        "ocr_failures": failures,
        "skipped_regions": sum(1 for r in regions if r["routing"]["decision"] == ROUTE_SKIP),
        "mean_ocr_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


def validate_ocr_document(document):
    """Structural check of an ocr.json object. Returns a list of problems (empty = valid)."""
    if not isinstance(document, dict):
        return ["root is not an object"]
    problems = []
    engine = document.get("engine")
    if not isinstance(engine, dict) or not isinstance(engine.get("name"), str) or not engine.get("name"):
        problems.append("engine.name missing")
    pages = document.get("pages")
    if not isinstance(pages, list) or not pages:
        return problems + ["pages missing or empty"]

    types = {t.value for t in RegionType}
    for i, page in enumerate(pages):
        where = f"pages[{i}]"
        if not isinstance(page, dict) or page.get("page_number") != i + 1 or not isinstance(page.get("regions"), list):
            problems.append(f"{where} has no page_number {i + 1} or no regions")
            continue
        for j, region in enumerate(page["regions"]):
            rwhere = f"{where}.regions[{j}]"
            if not isinstance(region, dict):
                problems.append(f"{rwhere} is not an object")
                continue
            if not isinstance(region.get("region_id"), str) or region.get("page_number") != page["page_number"]:
                problems.append(f"{rwhere} has no region_id or a wrong page_number")
            if region.get("canonical_type") not in types:
                problems.append(f"{rwhere}.canonical_type is not a canonical class")
            for key in ("image_bbox", "pdf_bbox"):
                if not is_valid_bbox(region.get(key)):
                    problems.append(f"{rwhere}.{key} is not a valid box")

            source, text, confidence = region.get("text_source"), region.get("text"), region.get("confidence")
            routing = region.get("routing")
            if not isinstance(routing, dict) or routing.get("decision") not in (ROUTE_NATIVE, ROUTE_OCR, ROUTE_SKIP):
                problems.append(f"{rwhere}.routing missing")
            if source == TEXT_SOURCE_NATIVE:
                ok = isinstance(text, str) and text and confidence is None and region.get("engine") is None
            elif source == TEXT_SOURCE_OCR:
                ok = (isinstance(text, str) and isinstance(region.get("engine"), str)
                      and isinstance(region.get("lines"), list) and is_valid_bbox(region.get("crop_bbox"))
                      and (_is_confidence(confidence) if text else confidence is None)
                      and all(isinstance(line, dict) and isinstance(line.get("text"), str)
                              and _is_confidence(line.get("confidence")) and is_valid_bbox(line.get("image_bbox"))
                              and is_valid_bbox(line.get("pdf_bbox")) for line in region["lines"]))
            else:
                ok = source == TEXT_SOURCE_NONE and text is None and confidence is None
            if not ok:
                problems.append(f"{rwhere}: text, confidence and geometry do not fit text_source {source!r}")
    return problems


def _region_entry(region, routing):
    return {
        "region_id": region["region_id"],
        "page_number": region["page_number"],
        "canonical_type": region["type"],
        "source_label": region["source_label"],
        "image_bbox": list(region["image_bbox"]),
        "pdf_bbox": list(region["pdf_bbox"]),
        "text": None,
        "text_source": TEXT_SOURCE_NONE,
        "confidence": None,
        "engine": None,
        "routing": {"decision": routing.decision, "reason": routing.reason},
    }


def _ocr_fields(result, page, engine_name):
    lines = result["lines"]
    line_entries = []
    for line in lines:
        pdf_bbox = clip_bbox(image_bbox_to_pdf(line.bbox, page["scale_x"], page["scale_y"]),
                             page["pdf_width"], page["pdf_height"])
        line_entries.append({
            "text": clean_line(line.text),
            "confidence": line.confidence,
            "image_bbox": list(line.bbox),
            "pdf_bbox": list(pdf_bbox),
            "polygon": [list(point) for point in line.polygon],
        })
    return {
        "text": join_lines([(line.bbox, line.text) for line in lines]),
        "text_source": TEXT_SOURCE_OCR,
        "confidence": region_confidence(lines),
        "engine": engine_name,
        "raw_text": "\n".join(line.text for line in lines),     # as the engine returned it, in its order
        "crop_bbox": result["crop_bbox"],
        "lines": line_entries,
    }


def _checked_lines(lines, width, height):
    """Keep the engine's lines that have text, a confidence between 0 and 1 and a box on the image."""
    checked = []
    for line in lines:
        bbox = clip_bbox(line.bbox, width, height)
        if bbox is None or not clean_line(line.text) or not _is_confidence(line.confidence):
            continue
        checked.append(OcrLine(text=line.text, confidence=float(line.confidence), bbox=bbox,
                               polygon=tuple((float(x), float(y)) for x, y in line.polygon)))
    return checked


def _is_confidence(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            and 0.0 <= value <= 1.0)


def _open_page_image(work_dir, page):
    from PIL import Image       # imported here: a digital PDF needs neither Pillow nor the engine
    path = os.path.join(work_dir, *page["image_path"].split("/"))
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise OcrInputError(f"Page {page['page_number']}: cannot read page image: {path} ({exc})") from exc
    if (image.width, image.height) != (page["image_width"], page["image_height"]):
        raise OcrInputError(f"Page {page['page_number']}: page image does not have the size stored in pages.json")
    return image


def _prepare_crops_dir(work_dir):
    crops_dir = ensure_directory(os.path.join(work_dir, OCR_CROPS_DIR))
    # Crops of an earlier run must not be mistaken for crops of this one.
    for old in glob.glob(os.path.join(glob.escape(crops_dir), "page_*_region_*.png")):
        os.remove(old)
    return crops_dir


def _remove_stale(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise OcrInputError(f"Cannot replace: {path} ({exc})") from exc

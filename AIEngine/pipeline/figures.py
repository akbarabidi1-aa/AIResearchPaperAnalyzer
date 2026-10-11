"""Phase 7 figure extraction: FIGURE region -> crop -> caption -> type -> (charts only) chart data -> figures.json.

    Phase 3 FIGURE region ("image" or "chart" of the layout model)
        ->  crop of the Phase 2 page image       the region itself, and the region with padding
        ->  caption                              the CAPTION region above or below it, by geometry
        ->  figure number                        "Figure 3" split from the caption text
        ->  text in the figure                   native text of the PDF, else the Phase 4 OCR engine
        ->  figure type                          chart / diagram / image / figure / unknown
        ->  chart_data (charts only)             pipeline.charts
        ->  <work_dir>/metadata/figures.json

Extraction only. A figure keeps its image, its caption and the words printed in it; nothing here
describes or interprets what a figure shows. No reading order, no sections, no summary.
"""
import glob
import os
import re
import time

from .charts import (
    CHART_STATUSES,
    CHART_TYPES,
    CHART_UNKNOWN,
    STATUS_FAILED as CHART_FAILED,
    STATUS_PARSED as CHART_PARSED,
    TEXT_NATIVE,
    TEXT_OCR,
    TEXT_ROLES,
    VALUE_SOURCES,
    VERTICAL_RATIO,
    ChartText,
    RuleBasedChartExtractor,
    empty_chart_data,
)
from .common import OutputDirectoryError, PdfPipelineError, ensure_directory
from .layout import COORDINATE_SYSTEM, LAYOUT_JSON, RegionType, clip_bbox, image_bbox_to_pdf, load_pages_document
from .layout_eval import is_valid_bbox
from .ocr import (
    DEFAULT_OCR_PADDING,
    OCR_JSON,
    _is_confidence,
    _open_page_image,
    check_padding,
    clean_line,
    crop_box,
    is_blank,
    is_usable_text,
    load_layout_document,
    native_lines_in_region,
)
from .page_pipeline import METADATA_DIR, PAGES_JSON, _write_json
from .render import pdf_bbox_to_pixels
from .tables import CAPTION_AMBIGUITY_RATIO, MAX_CAPTION_GAP, _remove_stale, load_ocr_document

FIGURES_SCHEMA_VERSION = 1
FIGURE_RULES_VERSION = "1.0"
FIGURES_JSON = "figures.json"
FIGURE_IMAGES_DIR = "figure_images"                 # the crop of every figure: part of the result
FIGURE_DEBUG_DIR = "figure_crops"                   # padded crops and chart overlays: on request only
FIGURE_IMAGE_PATTERN = "page_{:03d}_fig_{:03d}.png"
FIGURE_OVERLAY_PATTERN = "page_{:03d}_fig_{:03d}_overlay.png"

DEFAULT_FIGURE_PADDING = DEFAULT_OCR_PADDING        # pixels of page added around a figure region

TYPE_FIGURE = "figure"          # a figure that is neither clearly a drawing nor clearly a picture
TYPE_CHART = "chart"
TYPE_DIAGRAM = "diagram"        # a drawing: flat colours on a plain background
TYPE_IMAGE = "image"            # a continuous-tone picture (photo, micrograph, rendering)
TYPE_UNKNOWN = "unknown"        # blank, or it could not be cut out
FIGURE_TYPES = (TYPE_FIGURE, TYPE_CHART, TYPE_DIAGRAM, TYPE_IMAGE, TYPE_UNKNOWN)

TYPE_FROM_LABEL = "layout_label"
TYPE_FROM_RULE = "visual_rule"

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# Layout labels that already say "chart", and CAPTION-class labels that are notes, not captions.
CHART_SOURCE_LABELS = frozenset({"chart"})
NOTE_SOURCE_LABELS = frozenset({"vision_footnote"})

# Horizontal overlap of a figure and its caption, as a share of the narrower of the two. Lower than
# for tables: a short caption set flush left under a centred figure overlaps it only in part.
FIGURE_CAPTION_OVERLAP_MIN = 0.25
CAPTION_BESIDE_TOLERANCE = 3.0  # points a caption may reach into the figure's height and still be above/below
# Share of a crop's pixels in its eight most common colours (16 levels per channel).
DIAGRAM_MIN_FLAT = 0.85
IMAGE_MAX_FLAT = 0.5
# Text read by OCR inside a figure is kept only when it is likely to be text at all: marks of a
# plot and the texture of a picture also come back as "characters".
FIGURE_OCR_MIN_CONFIDENCE = 0.5
FIGURE_OCR_MIN_HEIGHT = 0.5     # of the median height of the lines read in the same figure
# A rotated line (text running bottom to top) read by OCR is kept only when ...
VERTICAL_MIN_CONFIDENCE = 0.8
VERTICAL_MIN_CHARACTERS = 3
VERTICAL_MIN_RATIO = 1.2        # height of its box / width

# "Figure 3", "Fig. 2", "FIG. 4b", "Figure 3(a)", "Fig. S1", "Figure A.2"; then an optional separator.
_FIGURE_LABEL = re.compile(
    r"(?P<word>Figure|FIGURE|Fig\.?|FIG\.?)\s*(?P<number>(?:[A-Z]\.?)?\d{1,3}(?:\.\d{1,2})?)"
    r"(?P<part>\s?\([a-z]\)|[a-z](?![A-Za-z]))?\s*(?P<separator>[:.|–—-])?\s*")
_TABLE_LABEL = re.compile(r"(?:Table|TABLE|Tab\.)\s*[A-Z]?\.?\d")


class FigureError(PdfPipelineError):
    """Base class for figure extraction failures."""


class FigureInputError(FigureError):
    """The Phase 2 / 3 / 4 artifacts are missing, invalid or do not belong together."""


class FigureExtractionError(FigureError):
    """One figure could not be extracted. The run goes on with the other figures."""


def parse_figure_number(text):
    """Split a caption into its figure number and the caption text. Returns (number, text).

        "Figure 3: Overview of the model."  ->  ("Figure 3", "Overview of the model.")
        "Fig. 2. Results"                   ->  ("Fig. 2", "Results")
        "Fig. 4b - Detail"                  ->  ("Fig. 4b", "Detail")
        "Figure 3(a) Detail"                ->  ("Figure 3(a)", "Detail")

    The number is kept as it is written (only the spacing is made regular). A text that does not
    start with a figure label, or where the label runs on into a sentence ("Figure 3 shows that ..."),
    gives (None, the whole text). An empty remainder is None.
    """
    cleaned = clean_line(text or "")
    match = _FIGURE_LABEL.match(cleaned)
    if not match:
        return None, cleaned or None
    rest = cleaned[match.end():]
    if not match.group("separator") and rest and not (rest[0].isupper() or rest[0].isdigit()):
        return None, cleaned
    word, part = match.group("word"), (match.group("part") or "").strip()
    return f"{word} {match.group('number')}{part}", rest or None


def is_figure_caption_text(text):
    """True for a text that starts with a figure label followed by a separator ("Figure 3: ...").
    Such a TEXT region is treated as a caption the layout model did not label as one."""
    match = _FIGURE_LABEL.match(clean_line(text or ""))
    return bool(match and match.group("separator"))


def caption_candidates(regions, texts=None):
    """What can be the caption of a figure on one page: [{"region_id", "pdf_bbox", "text"}].
    `regions` are the regions of one layout.json page and `texts` {region_id: text} what is known of
    their text.

      - Every CAPTION region, except notes ("vision_footnote") and captions that start with a table
        label ("Table 2: ...").
      - A TEXT region whose text starts like a figure caption ("Figure 3: ..."): a caption the layout
        model did not label as one.
    """
    texts = texts or {}
    candidates = []
    for region in regions:
        text = texts.get(region["region_id"]) or ""
        if region["type"] == RegionType.CAPTION.value:
            if region.get("source_label") in NOTE_SOURCE_LABELS or _TABLE_LABEL.match(clean_line(text)):
                continue
        elif not (region["type"] == RegionType.TEXT.value and is_figure_caption_text(text)):
            continue
        candidates.append({"region_id": region["region_id"], "pdf_bbox": list(region["pdf_bbox"]),
                           "text": text or None})
    return candidates


def associate_figure_captions(regions, texts=None):
    """Pair FIGURE regions with their captions, by geometry. The arguments are those of
    caption_candidates, which says what can be a caption. A caption can belong to a figure when

      - it lies above or below the figure, not beside it,
      - the two overlap horizontally by at least FIGURE_CAPTION_OVERLAP_MIN of the narrower one (a
        caption in another column does not overlap at all),
      - they are at most MAX_CAPTION_GAP points apart vertically.

    A caption is offered only to the figure or table nearest to it, and a figure takes the nearest
    caption offered to it. Whenever the runner-up is less than CAPTION_AMBIGUITY_RATIO times farther,
    nothing is assigned: an unclear pairing is no pairing.

    Returns {figure region_id: {"caption_region_id", "position": "above" | "below", "gap", "text"}}.
    """
    def gap(owner, caption):
        overlap = min(owner[2], caption[2]) - max(owner[0], caption[0])
        if overlap < FIGURE_CAPTION_OVERLAP_MIN * min(owner[2] - owner[0], caption[2] - caption[0]):
            return None
        if caption[1] >= owner[3] - CAPTION_BESIDE_TOLERANCE:
            distance, position = max(caption[1] - owner[3], 0.0), "below"
        elif caption[3] <= owner[1] + CAPTION_BESIDE_TOLERANCE:
            distance, position = max(owner[1] - caption[3], 0.0), "above"
        else:
            return None
        return (distance, position) if distance <= MAX_CAPTION_GAP else None

    def clear_winner(candidates):
        candidates = sorted(candidates, key=lambda c: c[:2])
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[1][0] < CAPTION_AMBIGUITY_RATIO * max(candidates[0][0], 1.0):
            return None
        return candidates[0]

    figures = [r for r in regions if r["type"] == RegionType.FIGURE.value]
    tables = [r for r in regions if r["type"] == RegionType.TABLE.value]
    offers = {}
    for index, caption in enumerate(caption_candidates(regions, texts)):
        owners = figures if _FIGURE_LABEL.match(clean_line(caption["text"] or "")) else figures + tables
        found = [(gap(owner["pdf_bbox"], caption["pdf_bbox"]), owner["region_id"]) for owner in owners]
        nearest = clear_winner([(g[0], index, g[1], owner_id) for g, owner_id in found if g is not None])
        if nearest:
            offers.setdefault(nearest[3], []).append((nearest[0], index, nearest[2], caption))

    pairs = {}
    for figure in figures:
        nearest = clear_winner(offers.get(figure["region_id"], []))
        if nearest:
            caption = nearest[3]
            pairs[figure["region_id"]] = {"caption_region_id": caption["region_id"], "position": nearest[2],
                                          "gap": nearest[0], "text": caption["text"]}
    return pairs


def figure_crop_boxes(image_bbox, image_width, image_height, padding=DEFAULT_FIGURE_PADDING, avoid=()):
    """Whole-pixel boxes to cut out for a figure region: (raw, padded). `raw` is the region itself;
    `padded` is the region grown by `padding` pixels (never beyond the page), so that a label the
    layout box cuts through stays whole. `avoid` are boxes in page pixels (captions) the padding must
    not reach into: the padded box stops at one that lies above or below the region.
    Raises FigureExtractionError for a region that cannot be cut out, ValueError for a bad `padding`.
    """
    check_padding(padding)
    try:
        raw = crop_box(image_bbox, image_width, image_height, 0)
        x0, y0, x1, y1 = crop_box(image_bbox, image_width, image_height, padding)
    except PdfPipelineError as exc:
        raise FigureExtractionError(str(exc)) from exc
    for other in avoid:
        if other[0] < x1 and other[2] > x0:
            if raw[3] <= other[1] < y1:
                y1 = int(other[1])
            if y0 < other[3] <= raw[1]:
                y0 = int(-(-other[3] // 1))
    return raw, (x0, y0, x1, y1)


def colour_flatness(image):
    """Share of an image's pixels that have one of its eight most common colours (16 levels per
    channel): near 1 for a drawing with flat fills, low for a photograph."""
    import numpy as np
    pixels = np.asarray(image.convert("RGB").resize((min(image.width, 256), min(image.height, 256))))
    codes = (pixels[..., 0] >> 4).astype(int) * 256 + (pixels[..., 1] >> 4) * 16 + (pixels[..., 2] >> 4)
    counts = np.sort(np.bincount(codes.ravel()))[::-1]
    return float(counts[:8].sum() / counts.sum())


def classify_figure_image(image):
    """"diagram", "image" or "figure" for a crop that is not a chart, from its colours alone:
    flat colours (colour_flatness >= DIAGRAM_MIN_FLAT) are a drawing, continuous tones
    (<= IMAGE_MAX_FLAT) a picture, anything between stays the neutral "figure"."""
    flatness = colour_flatness(image)
    if flatness >= DIAGRAM_MIN_FLAT:
        return TYPE_DIAGRAM
    return TYPE_IMAGE if flatness <= IMAGE_MAX_FLAT else TYPE_FIGURE


def usable_ocr_items(items):
    """The OCR lines of a figure that are kept as its text: read with at least
    FIGURE_OCR_MIN_CONFIDENCE, holding a letter or digit, and not much smaller than the other lines
    (FIGURE_OCR_MIN_HEIGHT of their median height). The rest is plot marks or texture, not text."""
    items = [item for item in items if item.confidence >= FIGURE_OCR_MIN_CONFIDENCE
             and any(ch.isalnum() for ch in item.text)]
    heights = sorted(item.bbox[3] - item.bbox[1] for item in items)
    if not heights:
        return []
    return [item for item in items
            if item.bbox[3] - item.bbox[1] >= FIGURE_OCR_MIN_HEIGHT * heights[len(heights) // 2]]


class FigureExtractor:
    """Extracts one figure at a time: crop, type, the text printed in it and, for a chart, its data.

    `ocr_engine` is a Phase 4 OCREngine (the same class Phase 4 uses; no second OCR system). It is
    loaded lazily, once, and only when a figure has no native text. `chart_extractor` is a
    pipeline.charts.ChartExtractor. With use_ocr=False figures without native text get no text.
    """

    name = "figure-rules"

    def __init__(self, ocr_engine=None, chart_extractor=None, padding=DEFAULT_FIGURE_PADDING, use_ocr=True):
        self.ocr_engine = ocr_engine
        self.chart_extractor = chart_extractor or RuleBasedChartExtractor()
        self.padding = check_padding(padding)
        self.use_ocr = bool(use_ocr) and ocr_engine is not None

    def describe(self):
        """The "extractor" object of figures.json."""
        engine = self.ocr_engine.describe() if self.use_ocr else None
        return {
            "name": f"{self.name} + {self.chart_extractor.name}",
            "version": FIGURE_RULES_VERSION,
            "device": (engine or {}).get("device", "cpu"),
            "chart_extractor": self.chart_extractor.describe(),
            "ocr_engine": engine,
        }

    def associate_caption(self, region, regions, texts=None):
        """The caption of one FIGURE region among the regions of its page (see
        associate_figure_captions): {"caption_region_id", "position", "gap", "text"}, or None."""
        return associate_figure_captions(regions, texts).get(region["region_id"])

    def classify(self, source_label, image, texts=(), origin=(0, 0)):
        """Type of a figure crop. Returns (figure_type, "layout_label" | "visual_rule").

          1. The layout model already called it a chart          -> chart   (layout_label)
          2. Two axis lines and numeric tick labels on a scale   -> chart   (visual_rule)
          3. Otherwise, from its colours (classify_figure_image) -> diagram / image / figure
        A blank crop is "unknown".
        """
        if source_label in CHART_SOURCE_LABELS:
            return TYPE_CHART, TYPE_FROM_LABEL
        if is_blank(image):
            return TYPE_UNKNOWN, TYPE_FROM_RULE
        looks_like_chart = getattr(self.chart_extractor, "looks_like_chart", None)
        if looks_like_chart and looks_like_chart(image, list(texts), origin):
            return TYPE_CHART, TYPE_FROM_RULE
        return classify_figure_image(image), TYPE_FROM_RULE

    def read_text(self, page, page_image, box, vertical=False):
        """Text printed inside the box `box` (page pixels) of a page. Returns (ChartText items, source).

        Native text lines of the PDF that lie in the box are used when there are any (source
        "native", no confidence). Otherwise the box is given to the OCR engine (source "ocr", with
        the engine's confidence), or nothing is read when OCR is off (source None).
        With vertical=True the OCR path also reads text that runs bottom to top (axis titles).
        """
        pdf_box = image_bbox_to_pdf(box, page["scale_x"], page["scale_y"])
        native = native_lines_in_region(pdf_box, page) if all("lines" in b for b in page["blocks"]) else []
        if is_usable_text(" ".join(text for _, text, _ in native)):
            items = []
            for bbox, text, _ in native:
                text = clean_line(text)
                if text:
                    tall = bbox[3] - bbox[1] >= VERTICAL_RATIO * (bbox[2] - bbox[0]) and len(text) >= 2
                    items.append(ChartText(text, tuple(pdf_bbox_to_pixels(bbox, page["scale_x"], page["scale_y"])),
                                           None, TEXT_NATIVE, tall))
            return items, TEXT_NATIVE
        if not self.use_ocr:
            return [], None
        lines = self.ocr_engine.recognize_region(page_image, box, 0)["lines"]
        items = usable_ocr_items([ChartText(clean_line(line.text), tuple(line.bbox), line.confidence, TEXT_OCR)
                                  for line in lines])
        if vertical:
            items = self._add_vertical_text(items, page_image, box)
        return items, TEXT_OCR

    def _add_vertical_text(self, items, page_image, box):
        """OCR the crop turned a quarter clockwise and add the lines that are rotated text in the
        upright crop: at least VERTICAL_MIN_CHARACTERS characters, a box taller than wide, read with
        VERTICAL_MIN_CONFIDENCE (upright text read sideways gives short, unsure fragments).
        What the upright pass made of the same pixels is dropped."""
        from PIL import Image
        crop = page_image.crop(box)
        height = crop.height
        for line in self.ocr_engine.recognize_page(crop.transpose(Image.ROTATE_270)):
            text = clean_line(line.text)
            x0, x1 = line.bbox[1], line.bbox[3]
            y0, y1 = height - line.bbox[2], height - line.bbox[0]
            if (len(text) < VERTICAL_MIN_CHARACTERS or line.confidence < VERTICAL_MIN_CONFIDENCE
                    or y1 - y0 < VERTICAL_MIN_RATIO * (x1 - x0)):
                continue
            items = [item for item in items
                     if not (x0 - 2 <= (item.bbox[0] + item.bbox[2]) / 2 - box[0] <= x1 + 2
                             and y0 - 2 <= (item.bbox[1] + item.bbox[3]) / 2 - box[1] <= y1 + 2)]
            items.append(ChartText(text, (x0 + box[0], y0 + box[1], x1 + box[0], y1 + box[1]), line.confidence,
                                   TEXT_OCR, True))
        return items

    def extract(self, region, page, page_image, avoid=()):
        """Everything about one FIGURE region that comes from its own pixels and text.

        `page` is the page of pages.json, `page_image` its Phase 2 image (PIL), `avoid` the boxes of
        the page's captions in page pixels. Returns {"fields": the entry fields crop_bbox,
        figure_type, figure_type_source, ocr_text, text_source, chart_data; "crop": the raw crop;
        "padded_crop"; "padded_box"; "texts"}. Raises FigureExtractionError when the region cannot be
        cut out; a chart that cannot be read is not an error (chart_data.status "failed").
        """
        raw_box, box = figure_crop_boxes(region["image_bbox"], page_image.width, page_image.height, self.padding,
                                         avoid)
        crop, padded = page_image.crop(raw_box), page_image.crop(box)
        origin = (box[0], box[1])
        is_chart_label = region.get("source_label") in CHART_SOURCE_LABELS
        blank = is_blank(padded)
        texts, text_source = ([], None) if blank else self.read_text(page, page_image, box, vertical=is_chart_label)
        figure_type, type_source = self.classify(region.get("source_label"), padded, texts, origin)
        chart_data = None
        if figure_type == TYPE_CHART:
            if text_source == TEXT_OCR and not is_chart_label:
                texts = self._add_vertical_text(texts, page_image, box)
            try:
                chart_data = self.chart_extractor.extract_chart(padded, texts, origin)
            except Exception as exc:    # the figure is kept; only its chart reading is lost
                chart_data = empty_chart_data(self.chart_extractor.name, notes=[f"{type(exc).__name__}: {exc}"])
        fields = {
            "crop_bbox": list(box),
            "figure_type": figure_type,
            "figure_type_source": type_source,
            "ocr_text": [_text_entry(item, page) for item in sorted(texts, key=lambda t: (t.bbox[1], t.bbox[0]))],
            "text_source": text_source if texts else None,
            "chart_data": chart_data,
        }
        return {"fields": fields, "crop": crop, "padded_crop": padded, "padded_box": box, "texts": texts}


def run_document_figures(work_dir, extractor, save_debug=False):
    """Extract every FIGURE region of a working directory and write metadata/figures.json.

    Needs the Phase 2, 3 and 4 artifacts in `work_dir`; none is modified, and neither are tables.json
    and equations.json. Only FIGURE regions of layout.json are looked at; pages are not searched for
    figures. Each region is handled on its own: one that fails is kept with status "failed", an
    "error" and, whenever it can be cut out, its crop.

    The crop of every figure is saved as <work_dir>/figure_images/page_NNN_fig_NNN.png (`image_path`,
    relative to `work_dir`). With save_debug=True the padded crop that was analysed is also saved as
    <work_dir>/figure_crops/page_NNN_fig_NNN.png and, for charts, an overlay
    page_NNN_fig_NNN_overlay.png with the plot area and the detected bars, lines and points.

    Returns {"figures_path", "work_dir", "document", "timings", "images", "debug"}. `timings` has
    engine_init_s (0.0 if the OCR engine was already loaded or not needed), figure_s ({figure_id:
    seconds}) and total_s; it is not written to the file.
    Raises a PdfPipelineError for unusable input; no figures.json is left behind then.
    """
    started = time.perf_counter()
    work_dir = os.path.abspath(os.fspath(work_dir))
    figures_path = os.path.join(work_dir, METADATA_DIR, FIGURES_JSON)
    _remove_stale(figures_path)
    pages_document = load_pages_document(work_dir)
    try:
        layout_document = load_layout_document(work_dir, pages_document)
        ocr_document = load_ocr_document(work_dir, layout_document)
    except PdfPipelineError as exc:
        raise FigureInputError(str(exc)) from exc
    for directory, pattern in ((FIGURE_IMAGES_DIR, "page_*_fig_*.png"), (FIGURE_DEBUG_DIR, "page_*_fig_*.png")):
        _remove_old_images(os.path.join(work_dir, directory), pattern)

    engine = extractor.ocr_engine if extractor.use_ocr else None
    was_loaded = bool(engine and engine.loaded)
    pages, figure_seconds, images, debug = [], {}, [], []
    for page, layout_page, ocr_page in zip(pages_document["pages"], layout_document["pages"],
                                           ocr_document["pages"]):
        regions = [region for region in layout_page["regions"] if region["type"] == RegionType.FIGURE.value]
        text_regions = {region["region_id"]: region for region in ocr_page["regions"]}
        texts = {key: value["text"] for key, value in text_regions.items()}
        by_id = {region["region_id"]: region for region in layout_page["regions"]}
        captions = associate_figure_captions(layout_page["regions"], texts)
        avoid = [by_id[candidate["region_id"]]["image_bbox"]
                 for candidate in caption_candidates(layout_page["regions"], texts)]
        page_image = _open_page_image(work_dir, page) if regions else None
        figures = []
        for position, region in enumerate(regions, 1):
            t0 = time.perf_counter()
            figure = _figure_entry(region, position)
            match = captions.get(region["region_id"])
            if match:
                source = text_regions[match["caption_region_id"]]
                number, text = parse_figure_number(match["text"])
                figure.update(caption=text, caption_raw=match["text"], figure_number=number,
                              caption_region_id=match["caption_region_id"], caption_position=match["position"],
                              caption_pdf_bbox=list(by_id[match["caption_region_id"]]["pdf_bbox"]),
                              caption_text_source=source["text_source"], caption_confidence=source["confidence"])
            result = None
            try:
                result = extractor.extract(region, page, page_image, avoid)
                figure.update(result["fields"], status=STATUS_OK)
            except Exception as exc:    # one figure must not lose the others
                figure.update(status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
            crop = result["crop"] if result else _fallback_crop(page_image, region)
            if crop is not None:
                name = FIGURE_IMAGE_PATTERN.format(page["page_number"], position)
                images.append(_save_image(crop, os.path.join(work_dir, FIGURE_IMAGES_DIR), name))
                figure["image_path"] = f"{FIGURE_IMAGES_DIR}/{name}"
            if save_debug and result:
                debug_dir = os.path.join(work_dir, FIGURE_DEBUG_DIR)
                debug.append(_save_image(result["padded_crop"], debug_dir,
                                         FIGURE_IMAGE_PATTERN.format(page["page_number"], position)))
                if figure["chart_data"]:
                    from .figures_visualize import draw_chart_overlay
                    overlay = draw_chart_overlay(result["padded_crop"], result["padded_box"][:2],
                                                 figure["chart_data"], result["texts"])
                    debug.append(_save_image(overlay, debug_dir,
                                             FIGURE_OVERLAY_PATTERN.format(page["page_number"], position)))
            figure_seconds[figure["figure_id"]] = time.perf_counter() - t0
            figures.append(figure)
        pages.append({"page_number": page["page_number"], "figures": figures})

    document = {
        "schema_version": FIGURES_SCHEMA_VERSION,
        "extractor": extractor.describe(),
        "settings": {"padding_px": extractor.padding, "figure_ocr": extractor.use_ocr},
        "source": {
            "pages_json": f"{METADATA_DIR}/{PAGES_JSON}",
            "layout_json": f"{METADATA_DIR}/{LAYOUT_JSON}",
            "ocr_json": f"{METADATA_DIR}/{OCR_JSON}",
            "source_pdf": pages_document["document"].get("source_pdf"),
            "source_sha256": pages_document["document"].get("source_sha256"),
        },
        "coordinate_system": dict(COORDINATE_SYSTEM),
        "pages": pages,
    }
    document["summary"] = summarize_figures_document(document)
    problems = validate_figures_document(document)
    if problems:    # a bug in this module, not a property of the document
        raise AssertionError("figures.json would be invalid: " + "; ".join(problems))

    _write_json(figures_path, document)
    return {
        "figures_path": figures_path,
        "work_dir": work_dir,
        "document": document,
        "timings": {
            "engine_init_s": engine.load_seconds if engine and engine.loaded and not was_loaded else 0.0,
            "figure_s": figure_seconds,
            "total_s": time.perf_counter() - started,
        },
        "images": images,
        "debug": debug,
    }


def has_chart_numbers(chart_data):
    """True when a chart_data object holds at least one recovered number."""
    return any(bar["value"] is not None for series in chart_data["series"] for bar in series.get("bars", ())) or any(
        point["y"] is not None for series in chart_data["series"] for point in series.get("points", ()))


def summarize_figures_document(document):
    """Counts over all figures of a figures.json object. figure_regions = generic_figures + charts.
    charts_parsed counts the charts whose type was determined; chart_data_extracted those with at
    least one number."""
    figures = [figure for page in document["pages"] for figure in page["figures"]]
    charts = [f for f in figures if f["figure_type"] == TYPE_CHART]
    typed = [f["chart_data"]["chart_type"] for f in charts if f["chart_data"]["chart_type"] != CHART_UNKNOWN]
    return {
        "pages": len(document["pages"]),
        "figure_regions": len(figures),
        "generic_figures": len(figures) - len(charts),
        "charts": len(charts),
        "figure_types": {kind: sum(1 for f in figures if f["figure_type"] == kind) for kind in FIGURE_TYPES
                         if any(f["figure_type"] == kind for f in figures)},
        "captions_matched": sum(1 for f in figures if f["caption_region_id"] is not None),
        "figure_numbers_found": sum(1 for f in figures if f["figure_number"] is not None),
        "with_text": sum(1 for f in figures if f["ocr_text"]),
        "charts_parsed": len(typed),
        "chart_types": {kind: typed.count(kind) for kind in CHART_TYPES if kind in typed},
        "chart_data_extracted": sum(1 for f in charts if has_chart_numbers(f["chart_data"])),
        "chart_failures": sum(1 for f in charts if f["chart_data"]["status"] == CHART_FAILED),
        "failures": sum(1 for f in figures if f["status"] == STATUS_FAILED),
    }


def validate_figures_document(document):
    """Structural check of a figures.json object. Returns a list of problems (empty = valid)."""
    if not isinstance(document, dict):
        return ["root is not an object"]
    problems = []
    extractor = document.get("extractor")
    if not isinstance(extractor, dict) or not isinstance(extractor.get("name"), str) or not extractor.get("name"):
        problems.append("extractor.name missing")
    pages = document.get("pages")
    if not isinstance(pages, list) or not pages:
        return problems + ["pages missing or empty"]

    for i, page in enumerate(pages):
        if not isinstance(page, dict) or page.get("page_number") != i + 1 or not isinstance(page.get("figures"), list):
            problems.append(f"pages[{i}] has no page_number {i + 1} or no figures")
            continue
        for j, figure in enumerate(page["figures"]):
            where = f"pages[{i}].figures[{j}]"
            if not isinstance(figure, dict):
                problems.append(f"{where} is not an object")
                continue
            if not isinstance(figure.get("figure_id"), str) or figure.get("page_number") != page["page_number"]:
                problems.append(f"{where} has no figure_id or a wrong page_number")
            if not all(isinstance(figure.get(key), str) for key in ("source_region_id", "source_label")):
                problems.append(f"{where} has no source_region_id or no source_label")
            for key in ("image_bbox", "pdf_bbox"):
                if not is_valid_bbox(figure.get(key)):
                    problems.append(f"{where}.{key} is not a valid box")
            problems.extend(f"{where}: {problem}" for problem in _figure_problems(figure))
    return problems


def _figure_problems(figure):
    problems = []
    status, figure_type, chart = figure.get("status"), figure.get("figure_type"), figure.get("chart_data")
    if figure_type not in FIGURE_TYPES:
        problems.append("figure_type is not figure, chart, diagram, image or unknown")
    if status == STATUS_FAILED:
        if not isinstance(figure.get("error"), str) or figure_type != TYPE_UNKNOWN or chart is not None:
            problems.append("a failed figure needs an error, type unknown and no chart_data")
    elif status != STATUS_OK or "error" in figure:
        problems.append("status is not ok or failed, or an extracted figure has an error")
    if (figure_type == TYPE_CHART) != (chart is not None):
        problems.append("chart_data must be set for a chart and null for every other figure")
    if status == STATUS_OK and not figure.get("image_path"):
        problems.append("an extracted figure must keep its crop image")
    for key in ("caption", "figure_number", "image_path", "caption_region_id"):
        if figure.get(key) is not None and not (isinstance(figure[key], str) and figure[key]):
            problems.append(f"{key} is not null or text")
    if figure.get("caption_region_id") is None and (figure.get("caption") or figure.get("figure_number")):
        problems.append("a caption or figure number without a caption region")
    if figure.get("crop_bbox") is not None and not is_valid_bbox(figure["crop_bbox"]):
        problems.append("crop_bbox is not a valid box")

    texts = figure.get("ocr_text")
    if not isinstance(texts, list):
        return problems + ["ocr_text is not a list"]
    for item in texts:
        native = isinstance(item, dict) and item.get("text_source") == TEXT_NATIVE and item.get("confidence") is None
        read = isinstance(item, dict) and item.get("text_source") == TEXT_OCR and _is_confidence(item.get("confidence"))
        if not ((native or read) and isinstance(item.get("text"), str) and item["text"]
                and is_valid_bbox(item.get("image_bbox")) and is_valid_bbox(item.get("pdf_bbox"))
                and (item.get("role") is None or item.get("role") in TEXT_ROLES)):
            problems.append("an ocr_text item needs text, boxes, a source and a confidence that fits the source")
            break
    if chart is not None:
        problems.extend(_chart_problems(chart))
    return problems


def _chart_problems(chart):
    if not isinstance(chart, dict):
        return ["chart_data is not an object"]
    problems = []
    if chart.get("status") not in CHART_STATUSES or chart.get("chart_type") not in CHART_TYPES:
        problems.append("chart_data.status or chart_data.chart_type is not a known value")
    for key in ("x_axis", "y_axis"):
        axis = chart.get(key)
        if not (isinstance(axis, dict) and isinstance(axis.get("ticks"), list)
                and (axis.get("title") is None or isinstance(axis["title"], str))):
            problems.append(f"chart_data.{key} needs a title (or null) and a list of ticks")
    if not all(isinstance(chart.get(key), list) for key in ("legend", "series", "annotations", "notes")):
        return problems + ["chart_data needs the lists legend, series, annotations and notes"]
    numbers = False
    for series in chart["series"]:
        if not isinstance(series, dict) or ("bars" in series) == ("points" in series):
            problems.append("a series needs either bars or points")
            continue
        for bar in series.get("bars", ()):
            numbers = numbers or bar.get("value") is not None
            if (bar.get("value") is None) != (bar.get("value_source") is None) or (
                    bar.get("value") is not None and bar.get("value_source") not in VALUE_SOURCES):
                problems.append("a bar value needs a value_source, and only a value has one")
        for point in series.get("points", ()):
            numbers = numbers or point.get("y") is not None
            if (point.get("y") is None) != (point.get("value_source") is None) or (
                    point.get("y") is not None and point.get("value_source") not in VALUE_SOURCES):
                problems.append("a point needs a value_source, and only a point with a y value has one")
    if (chart.get("status") == CHART_PARSED) != numbers:
        problems.append("chart_data.status is parsed exactly when a number was recovered")
    if chart.get("status") == CHART_FAILED and (chart["series"] or chart.get("chart_type") != CHART_UNKNOWN):
        problems.append("a failed chart has no type and no series")
    return problems


def _figure_entry(region, position):
    """A figure entry before extraction: identity and geometry of its layout region, unchanged."""
    return {
        "figure_id": f"p{region['page_number']:03d}_f{position:03d}",
        "page_number": region["page_number"],
        "source_region_id": region["region_id"],
        "source_label": region["source_label"],
        "source_model": region.get("source_model"),
        "image_bbox": list(region["image_bbox"]),
        "pdf_bbox": list(region["pdf_bbox"]),
        "crop_bbox": None,
        "figure_type": TYPE_UNKNOWN,
        "figure_type_source": None,
        "caption": None,
        "caption_raw": None,
        "figure_number": None,
        "caption_region_id": None,
        "caption_position": None,
        "caption_pdf_bbox": None,
        "caption_text_source": None,
        "caption_confidence": None,
        "image_path": None,
        "ocr_text": [],
        "text_source": None,
        "chart_data": None,
        "status": STATUS_FAILED,
    }


def _text_entry(item, page):
    pdf_bbox = clip_bbox(image_bbox_to_pdf(item.bbox, page["scale_x"], page["scale_y"]),
                         page["pdf_width"], page["pdf_height"])
    return {"text": item.text, "confidence": item.confidence, "text_source": item.source, "role": item.role,
            "vertical": item.vertical, "image_bbox": [float(v) for v in item.bbox],
            "pdf_bbox": list(pdf_bbox or image_bbox_to_pdf(item.bbox, page["scale_x"], page["scale_y"]))}


def _fallback_crop(page_image, region):
    """The plain crop of a region whose extraction failed, or None when it cannot be cut out."""
    try:
        return page_image.crop(crop_box(region["image_bbox"], page_image.width, page_image.height, 0))
    except PdfPipelineError:
        return None


def _save_image(image, directory, name):
    target = os.path.join(ensure_directory(directory), name)
    try:
        image.save(target, format="PNG")
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot write figure image: {target} ({exc})") from exc
    return target


def _remove_old_images(directory, pattern):
    # Images of an earlier run must not be mistaken for images of this one.
    for old in glob.glob(os.path.join(glob.escape(directory), pattern)):
        try:
            os.remove(old)
        except OSError as exc:
            raise OutputDirectoryError(f"Cannot replace: {old} ({exc})") from exc

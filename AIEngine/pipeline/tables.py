"""Phase 5 table structure extraction: TABLE region -> rows, columns, cells, headers -> tables.json.

    Phase 3 TABLE region
        native text lines in it form a regular grid?      -> "native"  geometry of the lines + native text
        native text, but no regular grid (merged cells..) -> "hybrid"  structure model   + native text
        no native text (scan, picture)                    -> "vision"  structure model   + OCR text

Only the structure-model class knows the model library, and OCR goes through the Phase 4 OCREngine.
Cell text is kept exactly as read ("94.70" stays "94.70").

Tables only: no formula recognition, no figure or chart understanding, no reading order.
"""
import json
import os
import re
import statistics
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
)
from .layout_eval import is_valid_bbox
from .ocr import (
    DEFAULT_OCR_PADDING,
    OCR_JSON,
    TEXT_SOURCE_NATIVE,
    TEXT_SOURCE_OCR,
    OcrLine,
    _is_confidence,
    _open_page_image,
    clean_line,
    crop_box,
    group_rows,
    is_usable_text,
    join_lines,
    load_layout_document,
    native_lines_in_region,
    region_confidence,
    validate_ocr_document,
)
from .page_pipeline import METADATA_DIR, PAGES_JSON, _write_json
from .render import pdf_bbox_to_pixels

TABLES_SCHEMA_VERSION = 1
TABLES_JSON = "tables.json"
TABLES_DEBUG_DIR = "tables_debug"

SOURCE_NATIVE = "native"        # geometry of native lines + native text
SOURCE_VISION = "vision"        # structure model + OCR text
SOURCE_HYBRID = "hybrid"        # structure model + native text
MODES = ("auto", SOURCE_NATIVE, SOURCE_VISION, SOURCE_HYBRID)

STRUCTURE_GEOMETRY = "geometry"
STRUCTURE_MODEL = "model"

# Native geometry (PDF points).
COLUMN_GAP_MIN = 2.0            # texts closer than this horizontally are in the same column
ALIGN_TOLERANCE = 3.0           # a cell lines up with its column when left edge, right edge or centre is this close
NATIVE_MIN_FILL = 0.5           # share of the grid positions that must hold text

# A text line goes to the model cell holding the largest share of it, if that share is at least this.
CELL_MATCH_MIN = 0.3

MAX_HEADER_ROWS = 3

# Caption association (PDF points).
MAX_CAPTION_GAP = 36.0          # largest vertical distance between a table and its caption
CAPTION_OVERLAP_MIN = 0.5       # horizontal overlap, as a share of the narrower of the two
CAPTION_AMBIGUITY_RATIO = 1.5   # the nearest candidate wins only if the next one is this many times farther

_LETTER_OR_DIGIT = re.compile(r"[^\W_]", re.UNICODE)
_TAG = re.compile(r"<tr\b[^>]*>|<td\b([^>]*)>")
_SPAN = re.compile(r"(rowspan|colspan)\s*=\s*[\"']?(\d+)")


class TableError(PdfPipelineError):
    """Base class for table extraction failures."""


class TableInputError(TableError):
    """The Phase 2 / 3 / 4 artifacts are missing, invalid or do not belong together."""


class TableModelError(TableError):
    """The table structure model could not be loaded."""


class TableExtractionError(TableError):
    """One table could not be extracted. The run goes on with the other tables."""


@dataclass(frozen=True)
class StructureCell:
    """One cell of a recognized table: grid position, spans, and box [x0, y0, x1, y1] in pixels."""
    row: int
    column: int
    row_span: int
    col_span: int
    bbox: tuple


@dataclass(frozen=True)
class TableStructure:
    cells: tuple
    n_rows: int
    n_columns: int
    score: float            # the model's own confidence in the structure, 0-1


class TableStructureModel:
    """Interface of a table structure model. Subclasses set `name` and implement `_load_model` and
    `_recognize`; nothing outside the subclass touches the model library."""

    name = ""

    def __init__(self):
        self._model = None
        self.load_count = 0
        self.load_seconds = 0.0

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        """Load the model once; later calls do nothing. Raises TableModelError."""
        if self.loaded:
            return self
        started = time.perf_counter()
        try:
            self._model = self._load_model()
        except TableError:
            raise
        except Exception as exc:
            raise TableModelError(f"Cannot load table structure model {self.name}: {exc}") from exc
        self.load_count += 1
        self.load_seconds = time.perf_counter() - started
        return self

    def recognize(self, image):
        """Structure of the table shown by a PIL image, with cell boxes in pixels of that image.
        Raises TableExtractionError when the model's answer is not a table."""
        self.load()
        return check_structure(self._recognize(image), image.width, image.height)

    def describe(self):
        return {"name": self.name}

    def _load_model(self):
        raise NotImplementedError

    def _recognize(self, image):
        raise NotImplementedError


class SLANetPlusStructureModel(TableStructureModel):
    """SLANet_plus through PaddleOCR's standalone TableStructureRecognition module."""

    name = "SLANet_plus"

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
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TableStructureRecognition
        return TableStructureRecognition(model_name=self.name, device=self.device)

    def _recognize(self, image):
        import numpy as np
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])     # BGR, as OpenCV
        result = list(self._model.predict(pixels, batch_size=1))[0]
        return parse_structure_tokens(result["structure"], result["bbox"], float(result["structure_score"]))


def parse_structure_tokens(tokens, boxes, score):
    """Turn the HTML tokens of a structure model ("<tr>", "<td colspan="2">", ...) and one box per <td>
    (a rectangle or a polygon, as a flat list of numbers) into a TableStructure.

    rowspan and colspan are taken from the tokens; a cell is placed at the first free column of its row.
    Raises TableExtractionError when the tokens and boxes do not describe a table.
    """
    cells, occupied, row, column, index = [], set(), -1, 0, 0
    for tag in _TAG.finditer("".join(tokens)):
        if tag.group(1) is None:        # <tr>
            row, column = row + 1, 0
            continue
        if row < 0 or index >= len(boxes):
            raise TableExtractionError("table structure is malformed: cells outside a row, or more cells than boxes")
        spans = {name: int(value) for name, value in _SPAN.findall(tag.group(1))}
        row_span, col_span = max(spans.get("rowspan", 1), 1), max(spans.get("colspan", 1), 1)
        while (row, column) in occupied:
            column += 1
        occupied.update((row + r, column + c) for r in range(row_span) for c in range(col_span))
        points = [float(v) for v in boxes[index]]
        if len(points) < 4 or len(points) % 2:
            raise TableExtractionError("table structure is malformed: a cell box is not a list of points")
        xs, ys = points[0::2], points[1::2]
        cells.append(StructureCell(row, column, row_span, col_span, (min(xs), min(ys), max(xs), max(ys))))
        column += col_span
        index += 1
    if not cells or index != len(boxes):
        raise TableExtractionError("table structure is malformed: no cells, or fewer cells than boxes")
    return TableStructure(tuple(cells), max(c.row + c.row_span for c in cells),
                          max(c.column + c.col_span for c in cells), score)


def check_structure(structure, width, height):
    """Clip the cell boxes of a TableStructure to its image and check the grid. Raises TableExtractionError."""
    cells, seen = [], set()
    for cell in structure.cells:
        bbox = clip_bbox(cell.bbox, width, height)
        positions = {(cell.row + r, cell.column + c) for r in range(cell.row_span) for c in range(cell.col_span)}
        if (bbox is None or min(cell.row, cell.column) < 0 or min(cell.row_span, cell.col_span) < 1
                or positions & seen):
            raise TableExtractionError("table structure is malformed: a cell has no box or overlaps another cell")
        seen |= positions
        cells.append(StructureCell(cell.row, cell.column, cell.row_span, cell.col_span, bbox))
    if not cells:
        raise TableExtractionError("table structure is malformed: no cells")
    score = structure.score if _is_confidence(structure.score) else 0.0
    return TableStructure(tuple(cells), max(r for r, _ in seen) + 1, max(c for _, c in seen) + 1, score)


def parse_native_table(lines):
    """Rebuild a table from the positions of its native text lines, given as (bbox, text) in PDF points.

    Rows are the lines on one height (group_rows). Columns are the gaps of the horizontal projection:
    texts closer than COLUMN_GAP_MIN are one column. This is only trusted for a REGULAR grid; `problems`
    lists every reason not to trust it:

      - fewer than two rows or two columns;
      - two texts in one row and column (typically a merged cell bridging two columns);
      - a text that does not line up with the other texts of its column, by left edge, right edge or
        centre within ALIGN_TOLERANCE (typically a header centred over several columns);
      - less than NATIVE_MIN_FILL of the grid positions hold text (typically cells wrapped over lines).

    Returns {"n_rows", "n_columns", "cells": [{"row", "column", "text", "bbox"}], "confidence", "problems"}.
    Every grid position is a cell; its box is the column's width by the row's height. No spans are produced.
    """
    items = [(tuple(bbox), clean_line(text)) for bbox, text in lines]
    items = [item for item in items if item[1]]
    if not items:
        return {"n_rows": 0, "n_columns": 0, "cells": [], "confidence": 0.0, "problems": ["no text"]}

    rows = group_rows(items)
    columns = []
    for bbox, _ in sorted(items, key=lambda item: item[0][0]):
        if columns and bbox[0] - columns[-1][1] < COLUMN_GAP_MIN:
            columns[-1][1] = max(columns[-1][1], bbox[2])
        else:
            columns.append([bbox[0], bbox[2]])

    def column_of(bbox):
        return next(i for i, (x0, x1) in enumerate(columns) if x0 <= bbox[0] and bbox[2] <= x1)

    problems, grid, by_column = [], {}, {}
    for r, row in enumerate(rows):
        for bbox, text in row:
            c = column_of(bbox)
            if (r, c) in grid:
                problems.append(f"row {r} has two texts in column {c}")
            grid.setdefault((r, c), []).append((bbox, text))
            by_column.setdefault(c, []).append(bbox)

    misaligned = 0
    for c, boxes in by_column.items():
        for bbox in boxes:
            others = [other for other in boxes if other is not bbox]
            if len(others) < 2:
                continue
            anchors = (lambda b: b[0], lambda b: b[2], lambda b: (b[0] + b[2]) / 2)
            if min(abs(a(bbox) - statistics.median(a(o) for o in others)) for a in anchors) > ALIGN_TOLERANCE:
                misaligned += 1
                problems.append(f"a text of column {c} does not line up with the column")

    n_rows, n_columns = len(rows), len(columns)
    fill = len(grid) / (n_rows * n_columns)
    if n_rows < 2 or n_columns < 2:
        problems.append(f"not a grid: {n_rows} row(s), {n_columns} column(s)")
    if fill < NATIVE_MIN_FILL:
        problems.append(f"only {fill:.0%} of the grid positions hold text")

    cells = []
    for r, row in enumerate(rows):
        y0, y1 = min(b[1] for b, _ in row), max(b[3] for b, _ in row)
        for c, (x0, x1) in enumerate(columns):
            cells.append({"row": r, "column": c, "bbox": (x0, y0, x1, y1),
                          "text": " ".join(text for _, text in grid.get((r, c), []))})
    confidence = 0.0 if n_rows < 2 or n_columns < 2 else fill * (1 - misaligned / len(items))
    return {"n_rows": n_rows, "n_columns": n_columns, "cells": cells, "confidence": confidence, "problems": problems}


def fill_structure(structure, lines):
    """Put text lines into the cells of a TableStructure. Lines and cell boxes share one coordinate system.

    `lines` are OcrLine objects (confidence None for native text). A line goes to the cell that holds
    the largest share of the line's area, if that share is at least CELL_MATCH_MIN. Returns
    (texts, confidences, unassigned): two lists parallel to structure.cells and the texts that fit no cell.
    """
    per_cell, unassigned = [[] for _ in structure.cells], []
    for line in lines:
        area = (line.bbox[2] - line.bbox[0]) * (line.bbox[3] - line.bbox[1])
        best, best_share = None, 0.0
        for index, cell in enumerate(structure.cells):
            width = min(line.bbox[2], cell.bbox[2]) - max(line.bbox[0], cell.bbox[0])
            height = min(line.bbox[3], cell.bbox[3]) - max(line.bbox[1], cell.bbox[1])
            share = width * height / area if width > 0 and height > 0 and area > 0 else 0.0
            if share > best_share:
                best, best_share = index, share
        if best is not None and best_share >= CELL_MATCH_MIN:
            per_cell[best].append(line)
        elif clean_line(line.text):
            unassigned.append(clean_line(line.text))
    texts = [join_lines([(line.bbox, line.text) for line in cell_lines]).replace("\n", " ") for cell_lines in per_cell]
    confidences = [region_confidence(cell_lines) if cell_lines and cell_lines[0].confidence is not None else None
                   for cell_lines in per_cell]
    return texts, confidences, unassigned


def is_numeric_cell(text):
    """True for a cell that holds a value and no word: at least one digit and no letter ("94.70", "1,024",
    "-3.5%", "12 ± 0.4"). Only used to find header rows; the text itself is never converted."""
    return any(ch.isdigit() for ch in text) and not any(ch.isalpha() for ch in text)


def detect_header_rows(grid):
    """Indexes of the header rows of a table given as a list of rows of cell texts, or None when unknown.

    Header rows are the leading rows without any numeric cell, PROVIDED every row after them has at
    least one numeric cell and there are at most MAX_HEADER_ROWS of them. So a table of words only, or
    one whose first row already holds numbers, gets None: the first row is not assumed to be a header.
    """
    count = 0
    for row in grid:
        if any(is_numeric_cell(text) for text in row) or not any(row):
            break
        count += 1
    body = grid[count:]
    if not 0 < count <= MAX_HEADER_ROWS or not body:
        return None
    if not all(any(is_numeric_cell(text) for text in row) for row in body):
        return None
    return list(range(count))


def table_grid(table):
    """All rows of a table (header rows included) as lists of n_columns cell texts, built from its
    cells. The text of a merged cell is at its top-left position; the positions it covers hold ""."""
    grid = [[""] * table["n_columns"] for _ in range(table["n_rows"])]
    for cell in table["cells"]:
        grid[cell["row"]][cell["column"]] = cell["text"]
    return grid


def associate_captions(regions):
    """Pair TABLE regions with CAPTION regions of the same page, by geometry only.

    `regions` are the regions of one layout.json page. A caption can belong to a table when the two
    overlap horizontally by at least CAPTION_OVERLAP_MIN of the narrower one and are at most
    MAX_CAPTION_GAP points apart vertically. A caption is a candidate only for the table or figure
    nearest to it; a table takes its nearest candidate. Whenever the runner-up is less than
    CAPTION_AMBIGUITY_RATIO times farther, nothing is assigned. Returns {table region_id: caption region_id}.
    """
    def gap(a, b):
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        if overlap < CAPTION_OVERLAP_MIN * min(a[2] - a[0], b[2] - b[0]):
            return None
        distance = max(b[1] - a[3], a[1] - b[3], 0.0)
        return distance if distance <= MAX_CAPTION_GAP else None

    def clear_winner(candidates):
        """The nearest of [(gap, id)], or None when there is none or the next one is about as near."""
        candidates = sorted(candidates)
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[1][0] < CAPTION_AMBIGUITY_RATIO * max(candidates[0][0], 1.0):
            return None
        return candidates[0]

    owners = [r for r in regions if r["type"] in (RegionType.TABLE.value, RegionType.FIGURE.value)]
    by_owner = {}
    for caption in (r for r in regions if r["type"] == RegionType.CAPTION.value):
        gaps = [(gap(owner["pdf_bbox"], caption["pdf_bbox"]), owner["region_id"]) for owner in owners]
        nearest = clear_winner([(distance, owner_id) for distance, owner_id in gaps if distance is not None])
        if nearest:
            by_owner.setdefault(nearest[1], []).append((nearest[0], caption["region_id"]))

    pairs = {}
    for table in (r for r in regions if r["type"] == RegionType.TABLE.value):
        nearest = clear_winner(by_owner.get(table["region_id"], []))
        if nearest:
            pairs[table["region_id"]] = nearest[1]
    return pairs


class TableExtractor:
    """Extracts one table at a time, choosing between the native, hybrid and vision paths.

    `structure_model` is a TableStructureModel and `ocr_engine` a Phase 4 OCREngine. Both are loaded
    lazily, once, and only when a table needs them: a document of regular native tables loads neither.
    """

    def __init__(self, structure_model, ocr_engine, padding=DEFAULT_OCR_PADDING):
        self.structure_model = structure_model
        self.ocr_engine = ocr_engine
        self.padding = padding

    def load(self):
        """Load both models now instead of at the first table that needs them."""
        self.structure_model.load()
        self.ocr_engine.load()
        return self

    def describe(self):
        """The "extractor" object of tables.json."""
        return {
            "name": "native-geometry + " + self.structure_model.name,
            "structure_model": self.structure_model.describe(),
            "ocr_engine": self.ocr_engine.describe(),
        }

    def extract(self, region, page, open_page_image, mode="auto", ocr_lines=None):
        """Structure and cell text of one TABLE region, by the path `mode` names or, with "auto":

            native  when the region's native lines form a regular grid (parse_native_table has no problems)
            hybrid  when the region has usable native text that does not
            vision  when it has no usable native text

        `open_page_image` returns the Phase 2 page image (PIL) and is only called for the model paths.
        `ocr_lines` are OcrLine objects in page pixels already recognized for this region, if any.
        Returns the structural part of a table entry. Raises TableExtractionError (or ValueError for a bad mode).
        """
        if mode not in MODES:
            raise ValueError(f"table mode must be one of {', '.join(MODES)}, got {mode!r}")
        native_lines = [(bbox, text) for bbox, text, _ in native_lines_in_region(region["pdf_bbox"], page)]
        if not is_usable_text(join_lines(native_lines)):
            native_lines = []

        if mode in ("auto", SOURCE_NATIVE) and native_lines:
            table = self.extract_native(region, page, native_lines)
            if mode == SOURCE_NATIVE or not table["notes"]:
                return table
        if mode == SOURCE_NATIVE or (mode == SOURCE_HYBRID and not native_lines):
            raise TableExtractionError("the table region has no usable native text")
        use_native = native_lines and mode != SOURCE_VISION
        return self.extract_image(region, page, open_page_image(), native_lines=native_lines if use_native else None,
                                  ocr_lines=ocr_lines)

    def extract_native(self, region, page, native_lines=None):
        """The native path: grid from the positions of the native lines, text from the PDF. No model.
        The result's "notes" list why the grid may be wrong (empty for a regular grid)."""
        if native_lines is None:
            native_lines = [(bbox, text) for bbox, text, _ in native_lines_in_region(region["pdf_bbox"], page)]
        parsed = parse_native_table(native_lines)
        if not parsed["cells"]:
            raise TableExtractionError("the table region has no usable native text")
        cells = [{
            "row": cell["row"], "column": cell["column"], "row_span": 1, "col_span": 1, "text": cell["text"],
            "text_source": TEXT_SOURCE_NATIVE if cell["text"] else None, "confidence": None,
            "image_bbox": pdf_bbox_to_pixels(cell["bbox"], page["scale_x"], page["scale_y"]),
        } for cell in parsed["cells"]]
        return _table_fields(cells, parsed["n_rows"], parsed["n_columns"], page, SOURCE_NATIVE, STRUCTURE_GEOMETRY,
                             parsed["confidence"], notes=parsed["problems"])

    def extract_image(self, region, page, page_image, native_lines=None, ocr_lines=None):
        """The model paths: structure from the model on the region's crop of the page image, and text

            hybrid  from `native_lines` ((bbox in PDF points, text) pairs) when given: nothing is OCR'd;
            vision  from `ocr_lines` when given, else from the OCR engine on the same crop.
        """
        try:
            box = crop_box(region["image_bbox"], page_image.width, page_image.height, self.padding)
        except PdfPipelineError as exc:
            raise TableExtractionError(str(exc)) from exc
        structure = self.structure_model.recognize(page_image.crop(box))
        structure = TableStructure(
            tuple(StructureCell(c.row, c.column, c.row_span, c.col_span,
                                (c.bbox[0] + box[0], c.bbox[1] + box[1], c.bbox[2] + box[0], c.bbox[3] + box[1]))
                  for c in structure.cells), structure.n_rows, structure.n_columns, structure.score)

        if native_lines:
            source, text_source = SOURCE_HYBRID, TEXT_SOURCE_NATIVE
            lines = [OcrLine(text, None, tuple(pdf_bbox_to_pixels(bbox, page["scale_x"], page["scale_y"])))
                     for bbox, text in native_lines]
        else:
            source, text_source = SOURCE_VISION, TEXT_SOURCE_OCR
            lines = ocr_lines
            if lines is None:
                lines = self.ocr_engine.recognize_region(page_image, region["image_bbox"], self.padding)["lines"]
        texts, confidences, unassigned = fill_structure(structure, lines)
        if not any(texts):
            raise TableExtractionError("no text found in the table region")

        cells = [{
            "row": cell.row, "column": cell.column, "row_span": cell.row_span, "col_span": cell.col_span,
            "text": text, "text_source": text_source if text else None, "confidence": confidence,
            "image_bbox": list(cell.bbox),
        } for cell, text, confidence in zip(structure.cells, texts, confidences)]
        matched = len(lines) - len(unassigned)
        return _table_fields(cells, structure.n_rows, structure.n_columns, page, source, STRUCTURE_MODEL,
                             structure.score * matched / len(lines), unassigned_text=unassigned)


def load_ocr_document(work_dir, layout_document):
    """Read <work_dir>/metadata/ocr.json and check it belongs to `layout_document`. Raises TableInputError."""
    path = os.path.join(os.fspath(work_dir), METADATA_DIR, OCR_JSON)
    if not os.path.isfile(path):
        raise TableInputError(f"No Phase 4 result found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
    except (OSError, ValueError) as exc:
        raise TableInputError(f"Cannot read {path}: {exc}") from exc
    problems = validate_ocr_document(document)
    if problems:
        raise TableInputError(f"Invalid {path}: " + "; ".join(problems[:5]))

    def region_ids(doc):
        return [[region["region_id"] for region in page["regions"]] for page in doc["pages"]]

    if (document.get("source", {}).get("source_sha256") != layout_document.get("source", {}).get("source_sha256")
            or region_ids(document) != region_ids(layout_document)):
        raise TableInputError(f"{path} was made from another layout.json; run Phase 4 again")
    return document


def run_document_tables(work_dir, extractor, mode="auto", visualize=False):
    """Extract every TABLE region of a working directory and write metadata/tables.json.

    Needs the Phase 2, 3 and 4 artifacts in `work_dir`; none is modified. Each TABLE region is handled
    on its own: a table that cannot be extracted is kept with an "error" and the run goes on. Text
    lines that Phase 4 already recognized for a table region (--ocr-tables) are reused, not OCR'd again.
    The caption is the text Phase 4 stored for the CAPTION region that associate_captions pairs with
    the table. With visualize=True each extracted table is also drawn on a copy of its page as
    <work_dir>/tables_debug/page_NNN_table_NNN_debug.png.

    Returns {"tables_path", "work_dir", "document", "timings", "visualizations"}. `timings` has
    model_init_s (models loaded during this run), table_s ({table_id: seconds}) and total_s; it is not
    written to the file. Raises a PdfPipelineError for unusable input; no tables.json is left behind then.
    """
    started = time.perf_counter()
    if mode not in MODES:
        raise ValueError(f"table mode must be one of {', '.join(MODES)}, got {mode!r}")
    work_dir = os.path.abspath(os.fspath(work_dir))
    tables_path = os.path.join(work_dir, METADATA_DIR, TABLES_JSON)
    _remove_stale(tables_path)      # first: a tables.json of other inputs must not survive a failed run
    pages_document = load_pages_document(work_dir)
    try:
        layout_document = load_layout_document(work_dir, pages_document)
    except PdfPipelineError as exc:
        raise TableInputError(str(exc)) from exc
    ocr_document = load_ocr_document(work_dir, layout_document)
    if any("lines" not in block for page in pages_document["pages"] for block in page["blocks"]):
        raise TableInputError("pages.json has no native text lines (written before Phase 4); run Phase 2 again")

    models = (extractor.structure_model, extractor.ocr_engine)
    loaded_before = [model.loaded for model in models]

    def loaded_seconds():
        return sum(model.load_seconds for model in models if model.loaded)

    pages, table_seconds, drawn = [], {}, []
    for page, layout_page, ocr_page in zip(pages_document["pages"], layout_document["pages"], ocr_document["pages"]):
        ocr_regions = {region["region_id"]: region for region in ocr_page["regions"]}
        captions = associate_captions(layout_page["regions"])
        image = []

        def open_page_image():
            if not image:
                image.append(_open_page_image(work_dir, page))
            return image[0]

        tables = []
        for region in layout_page["regions"]:
            if region["type"] != RegionType.TABLE.value:
                continue
            table = {
                "table_id": f"p{page['page_number']:03d}_t{len(tables) + 1:03d}",
                "region_id": region["region_id"],
                "page_number": page["page_number"],
                "image_bbox": list(region["image_bbox"]),
                "pdf_bbox": list(region["pdf_bbox"]),
                "caption": None,
                "caption_region_id": None,
            }
            caption = ocr_regions.get(captions.get(region["region_id"]))
            if caption and caption["text"]:
                table.update(caption=caption["text"].replace("\n", " "), caption_region_id=caption["region_id"])

            recognized = ocr_regions[region["region_id"]]
            ocr_lines = None
            if recognized["text_source"] == TEXT_SOURCE_OCR:
                ocr_lines = [OcrLine(line["text"], line["confidence"], tuple(line["image_bbox"]))
                             for line in recognized["lines"]]
            t0, init_before = time.perf_counter(), loaded_seconds()
            try:
                table.update(extractor.extract(region, page, open_page_image, mode, ocr_lines))
            except TableInputError:
                raise
            except Exception as exc:    # one unreadable table must not lose the others
                table.update(_failed_fields(f"{type(exc).__name__}: {exc}"))
            # A model loaded for this table is counted in model_init_s, not in the table's own time.
            table_seconds[table["table_id"]] = time.perf_counter() - t0 - (loaded_seconds() - init_before)
            tables.append(table)

        if visualize and any("error" not in table for table in tables):
            from .tables_visualize import draw_table, table_image_name
            debug_dir = ensure_directory(os.path.join(work_dir, TABLES_DEBUG_DIR))
            for index, table in enumerate(tables, 1):
                if "error" not in table:
                    target = os.path.join(debug_dir, table_image_name(page["page_number"], index))
                    draw_table(open_page_image(), table, target)
                    drawn.append(target)
        pages.append({"page_number": page["page_number"], "tables": tables})

    document = {
        "schema_version": TABLES_SCHEMA_VERSION,
        "extractor": extractor.describe(),
        "settings": {"mode": mode, "padding_px": extractor.padding},
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
    document["summary"] = summarize_tables_document(document)
    problems = validate_tables_document(document)
    if problems:    # a bug in this module, not a property of the document
        raise AssertionError("tables.json would be invalid: " + "; ".join(problems))

    _write_json(tables_path, document)
    return {
        "tables_path": tables_path,
        "work_dir": work_dir,
        "document": document,
        "timings": {
            "model_init_s": sum(model.load_seconds for model, before in zip(models, loaded_before)
                                if model.loaded and not before),
            "table_s": table_seconds,
            "total_s": time.perf_counter() - started,
        },
        "visualizations": drawn,
    }


def summarize_tables_document(document):
    """Counts over all tables of a tables.json object."""
    tables = [table for page in document["pages"] for table in page["tables"]]
    extracted = [table for table in tables if "error" not in table]
    return {
        "pages": len(document["pages"]),
        "table_regions": len(tables),
        "tables_extracted": len(extracted),
        SOURCE_NATIVE: sum(1 for t in extracted if t["source"] == SOURCE_NATIVE),
        SOURCE_VISION: sum(1 for t in extracted if t["source"] == SOURCE_VISION),
        SOURCE_HYBRID: sum(1 for t in extracted if t["source"] == SOURCE_HYBRID),
        "cells": sum(len(t["cells"]) for t in extracted),
        "with_headers": sum(1 for t in extracted if t["headers"] is not None),
        "with_caption": sum(1 for t in extracted if t["caption"] is not None),
        "extraction_failures": len(tables) - len(extracted),
    }


def validate_tables_document(document):
    """Structural check of a tables.json object. Returns a list of problems (empty = valid)."""
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
        if not isinstance(page, dict) or page.get("page_number") != i + 1 or not isinstance(page.get("tables"), list):
            problems.append(f"pages[{i}] has no page_number {i + 1} or no tables")
            continue
        for j, table in enumerate(page["tables"]):
            where = f"pages[{i}].tables[{j}]"
            if not isinstance(table, dict):
                problems.append(f"{where} is not an object")
                continue
            if not isinstance(table.get("table_id"), str) or table.get("page_number") != page["page_number"]:
                problems.append(f"{where} has no table_id or a wrong page_number")
            for key in ("image_bbox", "pdf_bbox"):
                if not is_valid_bbox(table.get(key)):
                    problems.append(f"{where}.{key} is not a valid box")
            if table.get("caption") is not None and not isinstance(table.get("caption"), str):
                problems.append(f"{where}.caption is not text")
            problems.extend(f"{where}: {problem}" for problem in _table_problems(table))
    return problems


def _table_problems(table):
    cells, rows, headers = table.get("cells"), table.get("rows"), table.get("headers")
    n_rows, n_columns, header_rows = table.get("n_rows"), table.get("n_columns"), table.get("header_rows")
    if not isinstance(cells, list) or not isinstance(rows, list):
        return ["cells or rows missing"]
    if "error" in table:
        ok = table.get("source") is None and not cells and not rows and headers is None
        return [] if ok else ["a failed table must be empty"]
    if table.get("source") not in (SOURCE_NATIVE, SOURCE_VISION, SOURCE_HYBRID):
        return ["source is not native, vision or hybrid"]
    if not _is_confidence(table.get("structure_confidence")):
        return ["structure_confidence is not between 0 and 1"]
    if not all(isinstance(v, int) and v > 0 for v in (n_rows, n_columns)):
        return ["n_rows or n_columns missing"]

    problems, covered = [], set()
    for cell in cells:
        spans = [cell.get(k) for k in ("row", "column", "row_span", "col_span")] if isinstance(cell, dict) else []
        if (len(spans) != 4 or not all(isinstance(v, int) for v in spans) or min(spans[:2]) < 0 or min(spans[2:]) < 1
                or spans[0] + spans[2] > n_rows or spans[1] + spans[3] > n_columns):
            problems.append("a cell lies outside the grid")
            continue
        positions = {(spans[0] + r, spans[1] + c) for r in range(spans[2]) for c in range(spans[3])}
        if positions & covered:
            problems.append("two cells cover the same position")
        covered |= positions
        if not isinstance(cell.get("text"), str) or not is_valid_bbox(cell.get("image_bbox")) \
                or not is_valid_bbox(cell.get("pdf_bbox")):
            problems.append("a cell has no text or no box")
        if cell.get("confidence") is not None and not _is_confidence(cell["confidence"]):
            problems.append("a cell confidence is not between 0 and 1")
    if header_rows is None:
        if headers is not None or len(rows) != n_rows:
            problems.append("without header rows, headers must be null and rows must hold every row")
    elif (not isinstance(headers, list) or len(headers) != n_columns
          or header_rows != list(range(len(header_rows))) or len(rows) != n_rows - len(header_rows)):
        problems.append("headers, header_rows and rows do not fit together")
    if not all(isinstance(row, list) and len(row) == n_columns for row in rows):
        problems.append("a row does not have n_columns cells")
    return problems


def _table_fields(cells, n_rows, n_columns, page, source, structure_method, confidence, notes=(), unassigned_text=()):
    """The structural part of a table entry from its cells (which have image_bbox but no pdf_bbox yet)."""
    for cell in cells:
        image_bbox = clip_bbox(cell["image_bbox"], page["image_width"], page["image_height"])
        pdf_bbox = image_bbox and clip_bbox(image_bbox_to_pdf(image_bbox, page["scale_x"], page["scale_y"]),
                                            page["pdf_width"], page["pdf_height"])
        if not pdf_bbox:
            raise TableExtractionError("a cell lies outside the page")
        cell.update(image_bbox=list(image_bbox), pdf_bbox=list(pdf_bbox))
    cells.sort(key=lambda cell: (cell["row"], cell["column"]))

    grid = table_grid({"n_rows": n_rows, "n_columns": n_columns, "cells": cells})
    header_rows = detect_header_rows(grid)
    headers = None
    if header_rows is not None:
        # A merged header cell labels every column it covers: "Accuracy Top-1", "Accuracy Top-5".
        labels = [[] for _ in range(n_columns)]
        for cell in cells:
            if cell["row"] in header_rows and cell["text"]:
                for column in range(cell["column"], cell["column"] + cell["col_span"]):
                    labels[column].append(cell["text"])
        headers = [" ".join(parts) for parts in labels]
    for cell in cells:
        cell["is_header"] = header_rows is not None and cell["row"] in header_rows
    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "header_rows": header_rows,
        "headers": headers,
        "rows": grid[len(header_rows or ()):],
        "cells": cells,
        "source": source,
        "structure_method": structure_method,
        "structure_confidence": confidence,
        "notes": list(notes),
        "unassigned_text": list(unassigned_text),
    }


def _failed_fields(error):
    return {"n_rows": 0, "n_columns": 0, "header_rows": None, "headers": None, "rows": [], "cells": [],
            "source": None, "structure_method": None, "structure_confidence": None, "notes": [],
            "unassigned_text": [], "error": error}


def _remove_stale(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot replace: {path} ({exc})") from exc

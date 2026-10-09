"""Phase 5 tests: native table geometry, structure tokens, cell filling, headers, captions, tables.json,
metrics.

    python -m unittest discover -s tests/python -v

The unit tests use a fake layout detector, a fake structure model and a fake OCR engine and need neither
PaddleOCR nor a model download. The classes named *IntegrationTests run the real PP-DocLayoutV3,
SLANet_plus and PP-OCRv5 models (first run downloads about 8 MB for SLANet_plus); they are skipped when
paddleocr is not installed or when AIRPA_SKIP_LAYOUT_MODEL=1, AIRPA_SKIP_OCR_MODEL=1 or
AIRPA_SKIP_TABLE_MODEL=1.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from test_layout import FakeDetector, TempDirTestCase, sha256
from test_ocr import DIGITAL, HAVE_OCR_MODEL, TABLE_TEXT, FakeEngine, OcrTestCase

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)
sys.path.insert(0, FIXTURES)

import make_table_fixtures as fixture  # noqa: E402
from pipeline import (  # noqa: E402
    OCREngine,
    OcrLine,
    PaddleOCREngine,
    PPDocLayoutV3Detector,
    SLANetPlusStructureModel,
    StructureCell,
    TableExtractionError,
    TableExtractor,
    TableInputError,
    TableModelError,
    TableStructure,
    TableStructureModel,
    associate_captions,
    crop_box,
    detect_document_layout,
    detect_header_rows,
    evaluate_table,
    evaluate_tables,
    parse_native_table,
    parse_structure_tokens,
    pdf_bbox_to_pixels,
    process_pdf,
    run_document_ocr,
    run_document_tables,
    table_grid,
    validate_tables_document,
)
from pipeline.tables import check_structure, fill_structure, is_numeric_cell  # noqa: E402
from pipeline.tables_visualize import grid_boundaries  # noqa: E402

TABLES_PDF = os.path.join(FIXTURES, "tables", "synthetic_tables.pdf")                   # 3 pages, native text
TABLES_SCANNED = os.path.join(FIXTURES, "tables", "synthetic_tables_scanned.pdf")       # the same as bitmaps
TRUTH_PATH = os.path.join(FIXTURES, "tables", "synthetic_tables.json")
with open(TRUTH_PATH, encoding="utf-8") as _fh:
    TRUTH = json.load(_fh)["tables"]

HAVE_TABLE_MODEL = HAVE_OCR_MODEL and os.environ.get("AIRPA_SKIP_TABLE_MODEL") != "1"

# Caption boxes as PP-DocLayoutV3 finds them (PDF points); the table boxes are the drawn tables.
CAPTION_BOXES = {1: (87.6, 193.6, 309.1, 206.8), 2: (87.6, 379.6, 347.1, 393.2), 3: (87.6, 193.6, 312.3, 206.8)}
TABLE_REGIONS = {t["page_number"]: [("figure_title", CAPTION_BOXES[t["page_number"]]), ("table", tuple(t["pdf_bbox"]))]
                 for t in TRUTH}
GRIDS = {t["page_number"]: t["grid"] for t in TRUTH}


class FakeStructureModel(TableStructureModel):
    """Returns the given TableStructure objects, one per call; counts loads and calls."""

    name = "fake-structure"

    def __init__(self, structures=(), fail_load=False):
        super().__init__()
        self.structures = list(structures)      # an Exception instance is raised instead of returned
        self.fail_load = fail_load
        self.sizes = []

    def _load_model(self):
        if self.fail_load:
            raise RuntimeError("simulated load failure")
        return object()

    def _recognize(self, image):
        self.sizes.append(image.size)
        answer = self.structures[len(self.sizes) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer


class ScriptedEngine(OCREngine):
    """Returns the given lists of OcrLine, one list per call."""

    name = "scripted-ocr"

    def __init__(self, responses=()):
        super().__init__()
        self.responses = list(responses)
        self.sizes = []

    def _load_model(self):
        return object()

    def _recognize(self, image):
        self.sizes.append(image.size)
        return list(self.responses[len(self.sizes) - 1])


def cell_geometry(spec):
    """[(row, column, col_span, text, box in PDF points)] of a fixture table, as it was drawn."""
    edges = [fixture.LEFT]
    for width in spec["column_widths"]:
        edges.append(edges[-1] + width)
    cells = []
    for r, row in enumerate(spec["cells"]):
        top, column = fixture.TABLE_TOP + r * fixture.ROW_HEIGHT, 0
        for cell in row:
            text, span = cell if isinstance(cell, tuple) else (cell, 1)
            cells.append((r, column, span, text, (edges[column], top, edges[column + span], top + fixture.ROW_HEIGHT)))
            column += span
    return cells


def lines_of(spec):
    """Native-like text lines [(bbox, text)] of a fixture table: left-aligned, 8 pt inside each cell."""
    return [((box[0] + 8, box[1] + 4, box[0] + 8 + 6 * len(text), box[3] - 4), text)
            for _, _, _, text, box in cell_geometry(spec) if text]


def cell(row, column, row_span=1, col_span=1, bbox=(0, 0, 10, 10)):
    return StructureCell(row, column, row_span, col_span, bbox)


def grid_structure(n_rows, n_columns, width=400, height=200, score=0.9):
    """A regular structure of equal cells filling a width x height image."""
    w, h = width / n_columns, height / n_rows
    return TableStructure(tuple(cell(r, c, bbox=(c * w, r * h, (c + 1) * w, (r + 1) * h))
                                for r in range(n_rows) for c in range(n_columns)), n_rows, n_columns, score)


def region(region_type, region_id, *pdf_bbox):
    return {"region_id": region_id, "type": region_type, "pdf_bbox": list(pdf_bbox)}


def all_tables(document):
    return [table for page in document["pages"] for table in page["tables"]]


class TablesTestCase(OcrTestCase):
    def prepare_tables(self, pdf=TABLES_PDF, name="doc", regions=None, **ocr_options):
        """Phase 2, a layout.json with the fixture's caption and table regions, and Phase 4 (fake engine)."""
        work = self.prepare(pdf, regions or TABLE_REGIONS, name)
        run_document_ocr(work, FakeEngine(), **ocr_options)
        return work

    def script(self, work, page_numbers=(1, 2, 3), confidence=0.9):
        """What the models would answer for the fixture tables of `work`: (structures, OCR line lists),
        both in pixels of the crop that the extractor cuts out for each table."""
        with open(os.path.join(work, "metadata", "pages.json"), encoding="utf-8") as fh:
            pages = json.load(fh)["pages"]
        with open(os.path.join(work, "metadata", "layout.json"), encoding="utf-8") as fh:
            layout = json.load(fh)["pages"]
        structures, responses = [], []
        for number in page_numbers:
            page, spec = pages[number - 1], fixture.TABLES[number - 1]
            table = next(r for r in layout[number - 1]["regions"] if r["type"] == "TABLE")
            origin = crop_box(table["image_bbox"], page["image_width"], page["image_height"], 8)

            def in_crop(box):
                x0, y0, x1, y1 = pdf_bbox_to_pixels(box, page["scale_x"], page["scale_y"])
                return (x0 - origin[0], y0 - origin[1], x1 - origin[0], y1 - origin[1])

            geometry = cell_geometry(spec)
            structures.append(TableStructure(
                tuple(cell(r, c, 1, span, in_crop(box)) for r, c, span, _, box in geometry),
                len(spec["cells"]), len(spec["column_widths"]), 0.95))
            responses.append([OcrLine(text, confidence, in_crop(bbox)) for bbox, text in lines_of(spec)])
        return structures, responses


class NativeGeometryTests(unittest.TestCase):
    def test_bordered_and_borderless_tables_are_rebuilt_from_line_positions(self):
        # Ruling lines are not text: both kinds of table give the same lines, so the same grid.
        for spec, truth in zip(fixture.TABLES[:2], TRUTH[:2]):
            parsed = parse_native_table(lines_of(spec))

            self.assertEqual(parsed["problems"], [])
            self.assertEqual((parsed["n_rows"], parsed["n_columns"]), (truth["n_rows"], truth["n_columns"]))
            self.assertEqual(parsed["confidence"], 1.0)
            grid = [[c["text"] for c in parsed["cells"] if c["row"] == r] for r in range(parsed["n_rows"])]
            self.assertEqual(grid, truth["grid"])

    def test_lines_in_any_order(self):
        lines = lines_of(fixture.TABLES[1])
        self.assertEqual(parse_native_table(lines[::-1])["cells"], parse_native_table(lines)["cells"])

    def test_cell_boxes_are_column_width_by_row_height(self):
        lines = [((10, 10, 40, 20), "a"), ((100, 11, 160, 21), "b"), ((10, 30, 60, 40), "c"), ((100, 30, 120, 40), "d")]
        parsed = parse_native_table(lines)

        self.assertEqual([(c["row"], c["column"], c["text"], c["bbox"]) for c in parsed["cells"]], [
            (0, 0, "a", (10, 10, 60, 21)), (0, 1, "b", (100, 10, 160, 21)),
            (1, 0, "c", (10, 30, 60, 40)), (1, 1, "d", (100, 30, 160, 40))])

    def test_numeric_text_is_kept_exactly(self):
        lines = [((10, 10, 40, 20), "Acc"), ((100, 10, 130, 20), "F1"),
                 ((10, 30, 40, 40), "94.70"), ((100, 30, 130, 40), "1,024"),
                 ((10, 50, 40, 60), "0.50"), ((100, 50, 130, 60), "-3.0%")]
        texts = [c["text"] for c in parse_native_table(lines)["cells"]]
        self.assertEqual(texts, ["Acc", "F1", "94.70", "1,024", "0.50", "-3.0%"])

    def test_empty_cells_are_kept_as_empty_text(self):
        lines = [((10, 10, 40, 20), "a"), ((100, 10, 130, 20), "b"), ((200, 10, 230, 20), "c"),
                 ((10, 30, 40, 40), "d"), ((200, 30, 230, 40), "f"),
                 ((10, 50, 40, 60), "g"), ((100, 50, 130, 60), "h"), ((200, 50, 230, 60), "i")]
        parsed = parse_native_table(lines)

        self.assertEqual(parsed["problems"], [])
        self.assertEqual([c["text"] for c in parsed["cells"]], ["a", "b", "c", "d", "", "f", "g", "h", "i"])
        self.assertAlmostEqual(parsed["confidence"], 8 / 9)

    def test_a_centred_merged_header_makes_the_grid_untrusted(self):
        parsed = parse_native_table([(bbox, text) for bbox, text in lines_of(fixture.TABLES[2]) if text != "Accuracy"]
                                    + [((275.2, 218.2, 324.8, 233.4), "Accuracy")])      # centred over two columns
        self.assertTrue(any("does not line up" in p for p in parsed["problems"]), parsed["problems"])
        self.assertLess(parsed["confidence"], 1.0)

    def test_a_cell_bridging_two_columns_makes_the_grid_untrusted(self):
        lines = [((10, 10, 150, 20), "one wide merged header"), ((200, 10, 230, 20), "c"),
                 ((10, 30, 40, 40), "1"), ((100, 30, 130, 40), "2"), ((200, 30, 230, 40), "3"),
                 ((10, 50, 40, 60), "4"), ((100, 50, 130, 60), "5"), ((200, 50, 230, 60), "6")]
        problems = parse_native_table(lines)["problems"]
        self.assertTrue(any("two texts in column" in p for p in problems), problems)

    def test_not_a_grid(self):
        one_row = parse_native_table([((10, 10, 40, 20), "a"), ((100, 10, 130, 20), "b")])
        one_column = parse_native_table([((10, 10, 40, 20), "a"), ((10, 30, 40, 40), "b")])
        for parsed in (one_row, one_column):
            self.assertTrue(any("not a grid" in p for p in parsed["problems"]))
            self.assertEqual(parsed["confidence"], 0.0)

    def test_sparse_grid(self):
        # A diagonal: 3 of 9 positions.
        lines = [((10, 10, 40, 20), "a"), ((100, 30, 130, 40), "b"), ((200, 50, 230, 60), "c")]
        self.assertTrue(any("grid positions" in p for p in parse_native_table(lines)["problems"]))

    def test_no_text(self):
        for lines in ([], [((10, 10, 40, 20), "   ")]):
            parsed = parse_native_table(lines)
            self.assertEqual((parsed["cells"], parsed["problems"]), ([], ["no text"]))


class StructureTokenTests(unittest.TestCase):
    BOX = [0, 0, 10, 0, 10, 5, 0, 5]        # a polygon: four corners

    def test_plain_grid(self):
        tokens = ["<html>", "<body>", "<table>", "<tr>", "<td></td>", "<td></td>", "</tr>", "<tr>", "<td></td>",
                  "<td></td>", "</tr>", "</table>", "</body>", "</html>"]
        structure = parse_structure_tokens(tokens, [[0, 0, 5, 5], [5, 0, 10, 5], [0, 5, 5, 10], [5, 5, 10, 10]], 0.9)

        self.assertEqual((structure.n_rows, structure.n_columns, structure.score), (2, 2, 0.9))
        self.assertEqual([(c.row, c.column, c.row_span, c.col_span) for c in structure.cells],
                         [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)])
        self.assertEqual(structure.cells[3].bbox, (5.0, 5.0, 10.0, 10.0))

    def test_spans_come_from_the_tokens(self):
        # Row 0: A (rowspan 2) | B (colspan 2).  Row 1: c | d.  Row 2: e | f | g.
        tokens = ["<tr>", "<td", ' rowspan="2"', ">", "</td>", "<td", ' colspan="2"', ">", "</td>", "</tr>",
                  "<tr>", "<td></td>", "<td></td>", "</tr>", "<tr>", "<td></td>", "<td></td>", "<td></td>", "</tr>"]
        structure = parse_structure_tokens(tokens, [self.BOX] * 7, 0.8)

        self.assertEqual((structure.n_rows, structure.n_columns), (3, 3))
        self.assertEqual([(c.row, c.column, c.row_span, c.col_span) for c in structure.cells],
                         [(0, 0, 2, 1), (0, 1, 1, 2), (1, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1)])
        self.assertEqual(structure.cells[0].bbox, (0.0, 0.0, 10.0, 5.0))        # bounding box of the polygon

    def test_malformed_structures(self):
        cases = ((["<tr>", "<td></td>", "<td></td>", "</tr>"], [self.BOX]),             # more cells than boxes
                 (["<tr>", "<td></td>", "</tr>"], [self.BOX, self.BOX]),                # fewer cells than boxes
                 (["<td></td>"], [self.BOX]),                                           # cell outside a row
                 (["<table>", "</table>"], []),                                         # no cells
                 (["<tr>", "<td></td>", "</tr>"], [[1, 2, 3]]))                         # not a box
        for tokens, boxes in cases:
            with self.subTest(tokens=tokens):
                with self.assertRaises(TableExtractionError):
                    parse_structure_tokens(tokens, boxes, 0.9)

    def test_check_structure(self):
        good = check_structure(TableStructure((cell(0, 0, bbox=(-5, 0, 50, 20)), cell(0, 1, bbox=(50, 0, 120, 20))),
                                              1, 2, 0.7), 100, 40)
        self.assertEqual([c.bbox for c in good.cells], [(0.0, 0.0, 50.0, 20.0), (50.0, 0.0, 100.0, 20.0)])   # clipped
        self.assertEqual((good.n_rows, good.n_columns), (1, 2))

        for bad in (TableStructure((), 0, 0, 0.9),
                    TableStructure((cell(0, 0), cell(0, 0)), 1, 1, 0.9),                    # same position twice
                    TableStructure((cell(0, 0, col_span=2), cell(0, 1)), 1, 2, 0.9),        # covered by a span
                    TableStructure((cell(0, 0, bbox=(500, 0, 600, 20)),), 1, 1, 0.9),       # outside the image
                    TableStructure((cell(0, 0, col_span=0),), 1, 1, 0.9)):
            with self.assertRaises(TableExtractionError):
                check_structure(bad, 100, 40)

    def test_model_load_failure(self):
        with self.assertRaises(TableModelError):
            FakeStructureModel(fail_load=True).load()


class FillStructureTests(unittest.TestCase):
    def test_lines_go_to_the_cell_that_holds_them(self):
        structure = grid_structure(2, 2)        # cells 200 x 100
        lines = [OcrLine("Method", 0.99, (10, 40, 90, 60)), OcrLine("Accuracy", 0.98, (210, 40, 300, 60)),
                 OcrLine("94.70", 0.9, (210, 140, 260, 160)), OcrLine("ViT", 0.8, (10, 140, 40, 160))]
        texts, confidences, unassigned = fill_structure(structure, lines)

        self.assertEqual(texts, ["Method", "Accuracy", "ViT", "94.70"])
        self.assertEqual(confidences, [0.99, 0.98, 0.8, 0.9])
        self.assertEqual(unassigned, [])

    def test_several_lines_of_one_cell_are_joined(self):
        structure = grid_structure(1, 2)        # cells 200 x 200
        lines = [OcrLine("second line", 0.5, (10, 100, 150, 120)), OcrLine("first", 1.0, (10, 60, 60, 80)),
                 OcrLine("right", 0.9, (220, 80, 280, 100))]
        texts, confidences, _ = fill_structure(structure, lines)

        self.assertEqual(texts, ["first second line", "right"])
        self.assertAlmostEqual(confidences[0], (5 * 1.0 + 10 * 0.5) / 15)

    def test_text_outside_every_cell_is_reported_not_lost(self):
        structure = TableStructure((cell(0, 0, bbox=(0, 0, 100, 50)),), 1, 1, 0.9)
        texts, _, unassigned = fill_structure(structure, [OcrLine("in", 0.9, (10, 10, 40, 30)),
                                                          OcrLine("far away", 0.9, (300, 10, 380, 30)),
                                                          OcrLine("mostly out", 0.9, (90, 10, 190, 30))])
        self.assertEqual(texts, ["in"])
        self.assertEqual(unassigned, ["far away", "mostly out"])

    def test_native_lines_have_no_confidence_and_empty_cells_stay_empty(self):
        structure = grid_structure(1, 2)
        texts, confidences, _ = fill_structure(structure, [OcrLine("0.50", None, (10, 80, 60, 100))])
        self.assertEqual((texts, confidences), (["0.50", ""], [None, None]))


class HeaderTests(unittest.TestCase):
    def test_numeric_cells(self):
        for text in ("94.70", "1,024", "-3.5%", "12 ± 0.4", "0", "(3)"):
            self.assertTrue(is_numeric_cell(text), text)
        for text in ("Accuracy", "Top-1", "F1", "n/a", "", "-", "3e-4"):
            self.assertFalse(is_numeric_cell(text), text)

    def test_text_row_above_numeric_rows(self):
        self.assertEqual(detect_header_rows(GRIDS[1]), [0])
        self.assertEqual(detect_header_rows(GRIDS[2]), [0])

    def test_two_header_rows(self):
        self.assertEqual(detect_header_rows(GRIDS[3]), [0, 1])

    def test_first_row_is_not_assumed_to_be_a_header(self):
        words = [["Name", "Role"], ["Ada", "Author"], ["Lin", "Editor"]]
        numbers = [["1", "2"], ["3", "4"]]
        mixed_body = [["Method", "Score"], ["ResNet", "92.1"], ["ViT", "pending"]]
        only_header = [["Method", "Score"]]
        too_many = [["a", "b"]] * 4 + [["1", "2"]]
        for grid in (words, numbers, mixed_body, only_header, too_many, []):
            with self.subTest(grid=grid):
                self.assertIsNone(detect_header_rows(grid))

    def test_a_text_column_in_the_body_does_not_prevent_a_header(self):
        self.assertIsNone(detect_header_rows([["", "2023", "2024"], ["Sales", "10", "12"]]))    # years are numbers
        self.assertEqual(detect_header_rows([["Region", "Sales"], ["North", "10"], ["South", "12"]]), [0])


class CaptionTests(unittest.TestCase):
    TABLE = region("TABLE", "t1", 90, 215, 480, 303)

    def test_caption_above_or_below(self):
        above = region("CAPTION", "c1", 88, 194, 309, 207)
        below = region("CAPTION", "c2", 88, 316, 347, 330)
        self.assertEqual(associate_captions([above, self.TABLE]), {"t1": "c1"})
        self.assertEqual(associate_captions([self.TABLE, below]), {"t1": "c2"})

    def test_no_caption(self):
        self.assertEqual(associate_captions([self.TABLE, region("TEXT", "x", 88, 194, 309, 207)]), {})
        self.assertEqual(associate_captions([self.TABLE]), {})

    def test_caption_too_far_or_not_aligned(self):
        far = region("CAPTION", "c1", 88, 100, 309, 113)            # 102 pt above
        beside = region("CAPTION", "c2", 490, 220, 560, 300)        # to the right, no horizontal overlap
        self.assertEqual(associate_captions([far, beside, self.TABLE]), {})

    def test_two_captions_at_similar_distance_are_ambiguous(self):
        above = region("CAPTION", "c1", 88, 194, 309, 207)          # 8 pt above
        below = region("CAPTION", "c2", 88, 313, 347, 327)          # 10 pt below
        self.assertEqual(associate_captions([above, self.TABLE, below]), {})

    def test_the_clearly_nearer_caption_wins(self):
        above = region("CAPTION", "c1", 88, 194, 309, 207)          # 8 pt above
        below = region("CAPTION", "c2", 88, 333, 347, 337)          # 30 pt below
        self.assertEqual(associate_captions([above, self.TABLE, below]), {"t1": "c1"})

    def test_a_caption_nearer_to_a_figure_is_not_the_tables(self):
        figure = region("FIGURE", "f1", 90, 40, 480, 180)
        caption = region("CAPTION", "c1", 88, 184, 309, 197)        # 4 pt under the figure, 18 pt above the table
        self.assertEqual(associate_captions([figure, caption, self.TABLE]), {})

    def test_a_caption_between_two_tables(self):
        upper = region("TABLE", "t0", 90, 60, 480, 180)
        midway = region("CAPTION", "c1", 88, 190, 309, 203)         # 10 pt and 12 pt
        near_lower = region("CAPTION", "c2", 88, 205, 309, 213)     # 25 pt and 2 pt
        self.assertEqual(associate_captions([upper, midway, self.TABLE]), {})
        self.assertEqual(associate_captions([upper, near_lower, self.TABLE]), {"t1": "c2"})


class ExtractorTests(TablesTestCase):
    """One table at a time, on the page-2 fixture table (regular grid, borderless)."""

    def setUp(self):
        super().setUp()
        self.work = self.prepare_tables()
        with open(os.path.join(self.work, "metadata", "pages.json"), encoding="utf-8") as fh:
            self.page = json.load(fh)["pages"][1]
        with open(os.path.join(self.work, "metadata", "layout.json"), encoding="utf-8") as fh:
            self.region = next(r for r in json.load(fh)["pages"][1]["regions"] if r["type"] == "TABLE")
        structures, responses = self.script(self.work, page_numbers=(2,))
        self.structure, self.ocr_lines = structures[0], responses[0]
        self.opened = 0

    def open_image(self):
        self.opened += 1
        with Image.open(os.path.join(self.work, "pages", "page_002.png")) as source:
            return source.convert("RGB")

    def extractor(self, structures=None, responses=None):
        self.model = FakeStructureModel([self.structure] if structures is None else structures)
        self.engine = ScriptedEngine([self.ocr_lines] if responses is None else responses)
        return TableExtractor(self.model, self.engine)

    def test_auto_uses_native_text_and_loads_no_model(self):
        table = self.extractor().extract(self.region, self.page, self.open_image)

        self.assertEqual((table["source"], table["structure_method"]), ("native", "geometry"))
        self.assertEqual(table["header_rows"], [0])
        self.assertEqual(table["headers"], GRIDS[2][0])
        self.assertEqual(table["rows"], GRIDS[2][1:])
        self.assertEqual({c["text_source"] for c in table["cells"]}, {"native"})
        self.assertEqual({c["confidence"] for c in table["cells"]}, {None})
        self.assertEqual((table["notes"], table["unassigned_text"]), ([], []))
        self.assertEqual((self.model.load_count, self.engine.load_count, self.opened), (0, 0, 0))

    def test_the_model_gets_the_crop_of_the_table_region(self):
        self.extractor().extract(self.region, self.page, self.open_image, mode="vision")

        x0, y0, x1, y1 = crop_box(self.region["image_bbox"], self.page["image_width"], self.page["image_height"], 8)
        self.assertEqual(self.model.sizes, [(x1 - x0, y1 - y0)])
        self.assertEqual(self.engine.sizes, [(x1 - x0, y1 - y0)])       # OCR runs on the same crop
        self.assertEqual(self.opened, 1)

    def test_vision_mode_structure_from_the_model_text_from_ocr(self):
        table = self.extractor().extract(self.region, self.page, self.open_image, mode="vision")

        self.assertEqual((table["source"], table["structure_method"]), ("vision", "model"))
        self.assertEqual(table_grid(table), GRIDS[2])
        self.assertEqual({c["text_source"] for c in table["cells"]}, {"ocr"})
        self.assertEqual({c["confidence"] for c in table["cells"]}, {0.9})
        self.assertAlmostEqual(table["structure_confidence"], 0.95)
        self.assertEqual((self.model.load_count, self.engine.load_count), (1, 1))

    def test_hybrid_mode_structure_from_the_model_text_from_the_pdf(self):
        table = self.extractor().extract(self.region, self.page, self.open_image, mode="hybrid")

        self.assertEqual((table["source"], table["structure_method"]), ("hybrid", "model"))
        self.assertEqual(table_grid(table), GRIDS[2])
        self.assertEqual({c["text_source"] for c in table["cells"]}, {"native"})
        self.assertEqual({c["confidence"] for c in table["cells"]}, {None})
        self.assertEqual(self.model.load_count, 1)
        self.assertEqual((self.engine.load_count, self.engine.sizes), (0, []))      # nothing is OCR'd

    def test_lines_already_recognized_are_reused(self):
        in_page = crop_box(self.region["image_bbox"], self.page["image_width"], self.page["image_height"], 8)
        known = [OcrLine(line.text, 0.77, (line.bbox[0] + in_page[0], line.bbox[1] + in_page[1],
                                           line.bbox[2] + in_page[0], line.bbox[3] + in_page[1]))
                 for line in self.ocr_lines]
        table = self.extractor().extract(self.region, self.page, self.open_image, mode="vision", ocr_lines=known)

        self.assertEqual(table_grid(table), GRIDS[2])
        self.assertEqual({c["confidence"] for c in table["cells"]}, {0.77})
        self.assertEqual((self.engine.load_count, self.engine.sizes), (0, []))

    def test_merged_cells_of_the_model_are_kept(self):
        merged = TableStructure((cell(0, 0, 1, 4, self.structure.cells[0].bbox),) + self.structure.cells[4:], 7, 4, 0.9)
        table = self.extractor([merged]).extract(self.region, self.page, self.open_image, mode="hybrid")

        first = table["cells"][0]
        self.assertEqual((first["row"], first["column"], first["row_span"], first["col_span"], first["text"]),
                         (0, 0, 1, 4, "Method"))
        self.assertEqual(table["rows"][0] if table["headers"] is None else table_grid(table)[0], ["Method", "", "", ""])
        self.assertEqual(table["unassigned_text"], ["Accuracy", "F1", "Time"])      # reported, not dropped
        self.assertAlmostEqual(table["structure_confidence"], 0.9 * 25 / 28)

    def test_cells_keep_their_position_in_both_coordinate_systems(self):
        for mode in ("native", "hybrid"):
            table = self.extractor().extract(self.region, self.page, self.open_image, mode=mode)
            tx0, ty0, tx1, ty1 = self.region["image_bbox"]
            for c in table["cells"]:
                x0, y0, x1, y1 = c["image_bbox"]
                self.assertTrue(tx0 - 1 <= x0 < x1 <= tx1 + 1 and ty0 - 1 <= y0 < y1 <= ty1 + 1, (mode, c))
                for i, scale in enumerate((self.page["scale_x"], self.page["scale_y"]) * 2):
                    self.assertAlmostEqual(c["pdf_bbox"][i], c["image_bbox"][i] / scale, places=6)
            # "94.70" is where it was drawn: second column, sixth row.
            target = next(c for c in table["cells"] if c["text"] == "94.70")
            self.assertEqual((target["row"], target["column"]), (5, 1))
            self.assertTrue(target["pdf_bbox"][0] < 232 < target["pdf_bbox"][2])
            self.assertTrue(target["pdf_bbox"][1] < 336 < target["pdf_bbox"][3])

    def test_failures(self):
        scanned_page = {**self.page, "blocks": [], "has_text_layer": False}
        with self.assertRaises(TableExtractionError):           # nothing native to read
            self.extractor().extract(self.region, scanned_page, self.open_image, mode="native")
        with self.assertRaises(TableExtractionError):
            self.extractor().extract(self.region, scanned_page, self.open_image, mode="hybrid")
        with self.assertRaises(TableExtractionError) as caught:     # the model sees a table, OCR finds no text
            self.extractor(responses=[[]]).extract(self.region, scanned_page, self.open_image)
        self.assertIn("no text found", str(caught.exception))
        with self.assertRaises(TableExtractionError):           # the model's answer is not a table
            self.extractor([TableStructure((), 0, 0, 0.9)]).extract(self.region, self.page, self.open_image, "vision")
        with self.assertRaises(TableExtractionError):
            self.extractor().extract({**self.region, "image_bbox": [10, 10, 12, 300]}, self.page, self.open_image,
                                     "vision")
        with self.assertRaises(ValueError):
            self.extractor().extract(self.region, self.page, self.open_image, mode="camelot")

    def test_forced_native_mode_reports_why_the_grid_is_doubtful(self):
        with open(os.path.join(self.work, "metadata", "pages.json"), encoding="utf-8") as fh:
            page = json.load(fh)["pages"][2]
        with open(os.path.join(self.work, "metadata", "layout.json"), encoding="utf-8") as fh:
            merged_region = next(r for r in json.load(fh)["pages"][2]["regions"] if r["type"] == "TABLE")

        table = self.extractor().extract(merged_region, page, self.open_image, mode="native")
        self.assertEqual(table["source"], "native")
        self.assertTrue(table["notes"])
        self.assertLess(table["structure_confidence"], 1.0)
        self.assertTrue(all(c["row_span"] == c["col_span"] == 1 for c in table["cells"]))      # no span is invented


class DocumentTablesTests(TablesTestCase):
    def run_digital(self, **options):
        self.work = self.prepare_tables()
        structures, _ = self.script(self.work, page_numbers=(3,))      # only the merged-header table needs the model
        self.model, self.engine = FakeStructureModel(structures), ScriptedEngine()
        return run_document_tables(self.work, TableExtractor(self.model, self.engine), **options)

    def run_scanned(self, **options):
        self.work = self.prepare_tables(TABLES_SCANNED)
        structures, responses = self.script(self.work)
        self.model, self.engine = FakeStructureModel(structures), ScriptedEngine(responses)
        return run_document_tables(self.work, TableExtractor(self.model, self.engine), **options)

    def test_digital_pdf(self):
        result = self.run_digital()
        tables = all_tables(result["document"])

        self.assertEqual([t["table_id"] for t in tables], ["p001_t001", "p002_t001", "p003_t001"])
        self.assertEqual([t["source"] for t in tables], ["native", "native", "hybrid"])
        self.assertEqual([(t["n_rows"], t["n_columns"]) for t in tables], [(4, 4), (7, 4), (5, 4)])
        self.assertEqual([table_grid(t) for t in tables], [GRIDS[1], GRIDS[2], GRIDS[3]])
        self.assertEqual([t["header_rows"] for t in tables], [[0], [0], [0, 1]])
        self.assertEqual(tables[0]["headers"], ["Dataset", "Classes", "Train", "Test"])
        self.assertEqual(tables[0]["rows"][0], ["Alpha", "10", "50,000", "10,000"])
        # Text comes from the PDF in all three: nothing was OCR'd.
        for t in tables:
            self.assertEqual({c["text_source"] for c in t["cells"] if c["text"]}, {"native"})
        self.assertEqual((self.engine.load_count, self.model.load_count, len(self.model.sizes)), (0, 1, 1))
        self.assertEqual(result["document"]["summary"], {
            "pages": 3, "table_regions": 3, "tables_extracted": 3, "native": 2, "vision": 0, "hybrid": 1,
            "cells": 63, "with_headers": 3, "with_caption": 3, "extraction_failures": 0})

    def test_merged_header_cell(self):
        merged = all_tables(self.run_digital()["document"])[2]

        spans = [(c["row"], c["column"], c["row_span"], c["col_span"], c["text"]) for c in merged["cells"]
                 if c["col_span"] > 1 or c["row_span"] > 1]
        self.assertEqual(spans, [(0, 1, 1, 2, "Accuracy")])
        self.assertEqual(len(merged["cells"]), 19)
        self.assertEqual(merged["headers"], ["Model", "Accuracy Top-1", "Accuracy Top-5", "Params"])
        self.assertEqual(merged["rows"], GRIDS[3][2:])
        self.assertEqual([c["is_header"] for c in merged["cells"]], [True] * 7 + [False] * 12)

    def test_numeric_cells_are_strings_exactly_as_printed(self):
        results = all_tables(self.run_digital()["document"])[1]
        self.assertEqual(results["rows"][4], ["ViT", "94.70", "94.10", "58.4"])
        self.assertEqual(results["rows"][1], ["MLP", "80.00", "79.42", "6.0"])
        for row in results["rows"]:
            for text in row:
                self.assertIsInstance(text, str)

    def test_captions(self):
        tables = all_tables(self.run_digital()["document"])
        self.assertEqual([t["caption"] for t in tables], [t["caption"] for t in TRUTH])
        self.assertEqual([t["caption_region_id"] for t in tables], ["p001_r001", "p002_r001", "p003_r001"])

    def test_table_regions_keep_their_identity_and_boxes(self):
        result = self.run_digital()
        with open(os.path.join(self.work, "metadata", "layout.json"), encoding="utf-8") as fh:
            regions = [r for page in json.load(fh)["pages"] for r in page["regions"] if r["type"] == "TABLE"]
        for table, layout_region in zip(all_tables(result["document"]), regions):
            self.assertEqual((table["region_id"], table["page_number"], table["image_bbox"], table["pdf_bbox"]),
                             (layout_region["region_id"], layout_region["page_number"], layout_region["image_bbox"],
                              layout_region["pdf_bbox"]))

    def test_scanned_pdf(self):
        result = self.run_scanned()
        tables = all_tables(result["document"])

        self.assertEqual([t["source"] for t in tables], ["vision"] * 3)
        self.assertEqual([table_grid(t) for t in tables], [GRIDS[1], GRIDS[2], GRIDS[3]])
        self.assertEqual([t["header_rows"] for t in tables], [[0], [0], [0, 1]])
        for t in tables:
            self.assertEqual({c["text_source"] for c in t["cells"] if c["text"]}, {"ocr"})
            self.assertEqual({c["confidence"] for c in t["cells"] if c["text"]}, {0.9})
            self.assertEqual(t["structure_method"], "model")
        self.assertEqual([t["caption"] for t in tables], ["recognized text 1", "recognized text 2", "recognized text 3"])
        self.assertEqual((len(self.model.sizes), len(self.engine.sizes)), (3, 3))

    def test_models_are_loaded_once_for_all_tables_and_documents(self):
        result = self.run_scanned()
        self.assertEqual((self.model.load_count, self.engine.load_count), (1, 1))
        self.assertEqual(set(result["timings"]), {"model_init_s", "table_s", "total_s"})
        self.assertEqual(sorted(result["timings"]["table_s"]), ["p001_t001", "p002_t001", "p003_t001"])

        other = self.prepare_tables(TABLES_SCANNED, "other")
        structures, responses = self.script(other)
        self.model.structures += structures
        self.engine.responses += responses
        again = run_document_tables(other, TableExtractor(self.model, self.engine))
        self.assertEqual((self.model.load_count, self.engine.load_count), (1, 1))
        self.assertEqual(again["timings"]["model_init_s"], 0.0)
        self.assertEqual(again["document"]["summary"]["tables_extracted"], 3)

    def test_table_text_recognized_in_phase_4_is_not_recognized_again(self):
        work = self.prepare(TABLES_SCANNED, TABLE_REGIONS)
        structures, responses = self.script(work)
        # Phase 4 with --ocr-tables: two regions per page, the caption and then the table.
        phase_4 = ScriptedEngine([lines for table_lines in responses
                                  for lines in ([OcrLine("a caption", 0.9, (10, 10, 200, 30))], table_lines)])
        run_document_ocr(work, phase_4, ocr_tables=True)
        self.assertEqual(len(phase_4.sizes), 6)

        tables_engine = ScriptedEngine()
        document = run_document_tables(work, TableExtractor(FakeStructureModel(structures), tables_engine))["document"]

        self.assertEqual([t["source"] for t in all_tables(document)], ["vision"] * 3)
        self.assertEqual([table_grid(t) for t in all_tables(document)], [GRIDS[1], GRIDS[2], GRIDS[3]])
        self.assertEqual((tables_engine.load_count, tables_engine.sizes), (0, []))

    def test_tables_json(self):
        result = self.run_digital()

        self.assertEqual(result["tables_path"], os.path.join(self.work, "metadata", "tables.json"))
        with open(result["tables_path"], encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data, result["document"])
        self.assertEqual(validate_tables_document(data), [])
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["extractor"], {"name": "native-geometry + fake-structure",
                                             "structure_model": {"name": "fake-structure"},
                                             "ocr_engine": {"name": "scripted-ocr"}})
        self.assertEqual(data["settings"], {"mode": "auto", "padding_px": 8})
        self.assertEqual(data["source"]["source_pdf"], "synthetic_tables.pdf")
        self.assertEqual(data["source"]["ocr_json"], "metadata/ocr.json")
        self.assertEqual([p["page_number"] for p in data["pages"]], [1, 2, 3])
        table = data["pages"][1]["tables"][0]
        self.assertEqual(set(table), {
            "table_id", "region_id", "page_number", "image_bbox", "pdf_bbox", "caption", "caption_region_id",
            "n_rows", "n_columns", "header_rows", "headers", "rows", "cells", "source", "structure_method",
            "structure_confidence", "notes", "unassigned_text"})
        self.assertEqual(set(table["cells"][0]), {"row", "column", "row_span", "col_span", "text", "text_source",
                                                  "confidence", "is_header", "image_bbox", "pdf_bbox"})

    def test_earlier_artifacts_are_not_modified(self):
        work = self.prepare_tables()
        names = ("metadata/pages.json", "metadata/layout.json", "metadata/ocr.json", "pages/page_001.png")
        before = {name: sha256(os.path.join(work, *name.split("/"))) for name in names}
        structures, _ = self.script(work, page_numbers=(3,))

        run_document_tables(work, TableExtractor(FakeStructureModel(structures), ScriptedEngine()), visualize=True)

        self.assertEqual(before, {name: sha256(os.path.join(work, *name.split("/"))) for name in names})
        self.assertEqual(sorted(os.listdir(os.path.join(work, "metadata"))),
                         ["layout.json", "ocr.json", "pages.json", "tables.json"])

    def test_forced_modes(self):
        work = self.prepare_tables()
        structures, _ = self.script(work)
        hybrid = run_document_tables(work, TableExtractor(FakeStructureModel(structures), ScriptedEngine()),
                                     mode="hybrid")["document"]
        self.assertEqual([t["source"] for t in all_tables(hybrid)], ["hybrid"] * 3)
        self.assertEqual([table_grid(t) for t in all_tables(hybrid)], [GRIDS[1], GRIDS[2], GRIDS[3]])
        self.assertEqual(hybrid["settings"]["mode"], "hybrid")
        with self.assertRaises(ValueError):
            run_document_tables(work, TableExtractor(FakeStructureModel(), ScriptedEngine()), mode="fast")

    def test_a_table_region_without_text_is_a_reported_failure(self):
        regions = {1: TABLE_REGIONS[1] + [("table", (90.0, 600.0, 480.0, 700.0))]}      # empty paper
        work = self.prepare_tables(regions=regions)
        model = FakeStructureModel([grid_structure(2, 2, 300, 150)])
        document = run_document_tables(work, TableExtractor(model, ScriptedEngine([[]])))["document"]

        good, empty = document["pages"][0]["tables"]
        self.assertEqual(good["source"], "native")
        self.assertEqual((empty["table_id"], empty["source"], empty["cells"], empty["rows"], empty["headers"]),
                         ("p001_t002", None, [], [], None))
        self.assertIn("no text found in the table region", empty["error"])
        self.assertEqual((document["summary"]["tables_extracted"], document["summary"]["extraction_failures"]), (1, 1))
        self.assertEqual(validate_tables_document(document), [])

    def test_a_malformed_table_fails_alone(self):
        work = self.prepare_tables(TABLES_SCANNED)
        structures, responses = self.script(work)
        structures[1] = RuntimeError("simulated model failure")
        model = FakeStructureModel(structures)
        # The failed table never reaches OCR, so the engine is asked twice.
        document = run_document_tables(work, TableExtractor(model, ScriptedEngine([responses[0], responses[2]])))["document"]

        tables = all_tables(document)
        self.assertEqual([t["source"] for t in tables], ["vision", None, "vision"])
        self.assertEqual(tables[1]["error"], "RuntimeError: simulated model failure")
        self.assertEqual(tables[1]["caption"], "recognized text 2")     # what is known is kept
        self.assertEqual(table_grid(tables[2]), GRIDS[3])
        self.assertEqual(document["summary"]["extraction_failures"], 1)
        self.assertEqual(validate_tables_document(document), [])

    def test_page_without_tables(self):
        work = self.prepare(DIGITAL, {1: [("text", (68.4, 78.4, 513.2, 134.4))]})
        run_document_ocr(work, FakeEngine())
        document = run_document_tables(work, TableExtractor(FakeStructureModel(), ScriptedEngine()))["document"]

        self.assertEqual([p["tables"] for p in document["pages"]], [[], []])
        self.assertEqual(document["summary"]["table_regions"], 0)
        self.assertEqual(validate_tables_document(document), [])

    def test_missing_or_mismatched_artifacts_and_no_file_after_a_failed_run(self):
        work = self.prepare(TABLES_PDF, TABLE_REGIONS)                  # no ocr.json yet
        extractor = TableExtractor(FakeStructureModel(), ScriptedEngine())
        with self.assertRaises(TableInputError) as caught:
            run_document_tables(work, extractor)
        self.assertIn("ocr.json", str(caught.exception))

        run_document_ocr(work, FakeEngine())
        structures, _ = self.script(work, page_numbers=(3,))
        run_document_tables(work, TableExtractor(FakeStructureModel(structures), ScriptedEngine()))
        self.assertTrue(os.path.isfile(os.path.join(work, "metadata", "tables.json")))

        # A new layout.json with other regions: ocr.json no longer belongs to it.
        detect_document_layout(work, FakeDetector({}))
        with self.assertRaises(TableInputError):
            run_document_tables(work, extractor)
        self.assertFalse(os.path.exists(os.path.join(work, "metadata", "tables.json")))

    def test_visualization_only_on_request(self):
        result = self.run_digital()
        self.assertEqual(result["visualizations"], [])
        self.assertFalse(os.path.exists(os.path.join(self.work, "tables_debug")))

        result = self.run_digital(visualize=True)
        self.assertEqual(sorted(os.listdir(os.path.join(self.work, "tables_debug"))),
                         ["page_001_table_001_debug.png", "page_002_table_001_debug.png", "page_003_table_001_debug.png"])
        with Image.open(os.path.join(self.work, "pages", "page_001.png")) as original, \
                Image.open(result["visualizations"][0]) as drawn:
            self.assertEqual(drawn.size, original.size)
            self.assertNotEqual(drawn.convert("RGB").tobytes(), original.convert("RGB").tobytes())
        table = all_tables(result["document"])[0]
        ys, xs = grid_boundaries(table)
        self.assertEqual((len(ys), len(xs)), (table["n_rows"] + 1, table["n_columns"] + 1))
        self.assertEqual((ys[0], ys[-1], xs[0], xs[-1]), (table["image_bbox"][1], table["image_bbox"][3],
                                                          table["image_bbox"][0], table["image_bbox"][2]))
        self.assertEqual(ys, sorted(ys))
        self.assertEqual(xs, sorted(xs))

    def test_validator_reports_problems(self):
        data = self.run_digital()["document"]
        self.assertTrue(validate_tables_document([]))
        self.assertTrue(validate_tables_document({"extractor": {"name": "x"}, "pages": []}))

        def broken(change):
            copy = json.loads(json.dumps(data))
            change(copy["pages"][0]["tables"][0])
            return validate_tables_document(copy)

        self.assertTrue(broken(lambda t: t.update(source="ocr")))
        self.assertTrue(broken(lambda t: t["rows"].pop()))
        self.assertTrue(broken(lambda t: t.update(headers=None)))
        self.assertTrue(broken(lambda t: t["cells"][0].update(column=9)))
        self.assertTrue(broken(lambda t: t["cells"][1].update(column=0)))       # two cells on one position
        self.assertTrue(broken(lambda t: t["cells"][0].update(col_span=0)))
        self.assertTrue(broken(lambda t: t["cells"][0].pop("pdf_bbox")))
        self.assertTrue(broken(lambda t: t.update(structure_confidence=1.4)))
        self.assertTrue(broken(lambda t: t.update(error="x")))                  # a failed table must be empty


class TableMetricTests(unittest.TestCase):
    @staticmethod
    def table(grid, spans=(), header_rows=None, caption=None, page_number=1, pdf_bbox=(0, 0, 100, 100)):
        """A tables.json entry for `grid`; `spans` are (row, column, row_span, col_span)."""
        merged = {(r, c): (rs, cs) for r, c, rs, cs in spans}
        covered = {(r + i, c + j) for (r, c), (rs, cs) in merged.items() for i in range(rs) for j in range(cs)}
        cells = [{"row": r, "column": c, "row_span": merged.get((r, c), (1, 1))[0],
                  "col_span": merged.get((r, c), (1, 1))[1], "text": text}
                 for r, row in enumerate(grid) for c, text in enumerate(row) if (r, c) in merged or (r, c) not in covered]
        return {"table_id": "p001_t001", "page_number": page_number, "pdf_bbox": list(pdf_bbox), "source": "native",
                "n_rows": len(grid), "n_columns": len(grid[0]), "cells": cells, "header_rows": header_rows,
                "caption": caption}

    EXPECTED = {"grid": [["Method", "Accuracy", "F1"], ["ResNet", "92.10", "91.40"], ["ViT", "94.70", "94.10"]],
                "header_rows": [0], "caption": "Table 2. Results."}

    def test_perfect_table(self):
        result = evaluate_table(self.EXPECTED, self.table(self.EXPECTED["grid"], header_rows=[0],
                                                          caption="Table 2. Results."))
        self.assertEqual(result, {
            "expected_rows": 3, "predicted_rows": 3, "rows_match": True, "expected_columns": 3,
            "predicted_columns": 3, "columns_match": True, "total_cells": 9, "exact_cells": 9,
            "exact_cell_accuracy": 1.0, "cell_text_accuracy": 1.0, "structure_accuracy": 1.0,
            "header_rows_match": True, "caption_match": True})

    def test_wrong_cells(self):
        grid = [["Method", "Accuracy", "F1"], ["ResNet", "92.10", "91.40"], ["ViT", "94.7", ""]]
        result = evaluate_table(self.EXPECTED, self.table(grid))

        self.assertEqual((result["exact_cells"], result["total_cells"]), (7, 9))
        self.assertAlmostEqual(result["exact_cell_accuracy"], 7 / 9)
        # "94.7" for "94.70" is not an exact match (1 of 5 characters lost); the empty cell scores 0.
        self.assertAlmostEqual(result["cell_text_accuracy"], (7 + 0.8 + 0.0) / 9)
        self.assertEqual((result["rows_match"], result["columns_match"], result["structure_accuracy"]), (True, True, 1.0))
        self.assertEqual((result["header_rows_match"], result["caption_match"]), (False, False))

    def test_whitespace_is_ignored_but_nothing_else(self):
        grid = [["Method ", "Accuracy", "F1"], ["ResNet", " 92.10", "91.40"], ["ViT", "94.70", "94.10"]]
        self.assertEqual(evaluate_table(self.EXPECTED, self.table(grid))["exact_cells"], 9)
        grid[0][0] = "method"
        self.assertEqual(evaluate_table(self.EXPECTED, self.table(grid))["exact_cells"], 8)

    def test_wrong_row_and_column_counts(self):
        missing_row = evaluate_table(self.EXPECTED, self.table(self.EXPECTED["grid"][:2]))
        self.assertEqual((missing_row["predicted_rows"], missing_row["rows_match"], missing_row["columns_match"]),
                         (2, False, True))
        self.assertEqual(missing_row["exact_cells"], 6)
        self.assertAlmostEqual(missing_row["structure_accuracy"], 6 / 9)

        extra_column = evaluate_table(self.EXPECTED, self.table([row + ["x"] for row in self.EXPECTED["grid"]]))
        self.assertEqual((extra_column["predicted_columns"], extra_column["columns_match"]), (4, False))
        self.assertEqual(extra_column["exact_cells"], 9)
        self.assertAlmostEqual(extra_column["structure_accuracy"], 9 / 12)

    def test_merged_cells_count_in_the_structure(self):
        expected = {"grid": [["Model", "Accuracy", ""], ["", "Top-1", "Top-5"], ["Small", "81.20", "95.10"]],
                    "merged_cells": [{"row": 0, "column": 1, "row_span": 1, "col_span": 2}]}
        with_span = evaluate_table(expected, self.table(expected["grid"], spans=[(0, 1, 1, 2)]))
        without = evaluate_table(expected, self.table(expected["grid"]))

        self.assertEqual((with_span["structure_accuracy"], with_span["exact_cells"]), (1.0, 9))
        self.assertEqual(without["exact_cells"], 9)                     # the text is right ...
        self.assertAlmostEqual(without["structure_accuracy"], 7 / 10)   # ... the structure is not
        self.assertEqual((with_span["header_rows_match"], with_span["caption_match"]), (None, None))

    def test_missing_or_failed_table(self):
        failed = {"table_id": "p001_t001", "error": "x", "source": None, "cells": [], "n_rows": 0, "n_columns": 0}
        for predicted in (None, failed):
            result = evaluate_table(self.EXPECTED, predicted)
            self.assertEqual((result["predicted_rows"], result["exact_cells"], result["exact_cell_accuracy"],
                              result["cell_text_accuracy"], result["structure_accuracy"]), (0, 0, 0.0, 0.0, 0.0))
            self.assertEqual((result["rows_match"], result["header_rows_match"], result["caption_match"]),
                             (False, False, False))

    def test_document_evaluation(self):
        truth = [{**self.EXPECTED, "page_number": 1, "pdf_bbox": [0, 0, 100, 100]},
                 {**self.EXPECTED, "page_number": 2, "pdf_bbox": [0, 0, 100, 100]},
                 {**self.EXPECTED, "page_number": 3, "pdf_bbox": [0, 0, 100, 100]}]
        wrong = [["Method", "Accuracy", "F1"], ["ResNet", "92.10", "91.40"], ["ViT", "94.7", "94.10"]]
        document = {"pages": [
            {"page_number": 1, "tables": [self.table(self.EXPECTED["grid"], header_rows=[0], caption="Table 2. Results.",
                                                     pdf_bbox=(1, 1, 100, 99))]},
            {"page_number": 2, "tables": [self.table(wrong, page_number=2)]},
            {"page_number": 3, "tables": [self.table(wrong, page_number=3, pdf_bbox=(500, 500, 600, 600))]}]}   # elsewhere

        result = evaluate_tables(truth, document)

        self.assertEqual((result["tables"], result["found"]), (3, 2))
        self.assertEqual([row["exact_cells"] for row in result["rows"]], [9, 8, 0])
        self.assertEqual((result["exact_cells"], result["total_cells"]), (17, 27))
        self.assertAlmostEqual(result["exact_cell_accuracy"], 17 / 27)
        self.assertAlmostEqual(result["row_count_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["column_count_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["header_accuracy"], 1 / 3)
        self.assertAlmostEqual(result["caption_accuracy"], 1 / 3)


class InspectTablesToolErrorTests(TempDirTestCase):
    """Failures that are reported before any model is needed."""
    TOOL = os.path.join(ENGINE, "tools", "inspect_tables.py")

    def test_bad_input_exits_with_code_2_and_no_traceback(self):
        os.makedirs(self.path("empty-dir"))
        for args in ([self.path("missing.pdf")], [self.path("empty-dir")],
                     [TABLES_PDF, "--output", self.path("out"), "--layout-threshold", "7"]):
            with self.subTest(args=args):
                done = subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=300)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn("Error:", done.stderr)
                self.assertNotIn("Traceback", done.stderr)


@unittest.skipUnless(HAVE_TABLE_MODEL, "paddleocr is not installed or a model test is switched off")
class TableModelIntegrationTests(unittest.TestCase):
    """The real layout, structure and OCR models on the fixtures. Every model is loaded once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="airpa-p5-model-")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        detector = PPDocLayoutV3Detector(device="cpu")
        cls.engine = PaddleOCREngine(device="cpu")
        cls.model = SLANetPlusStructureModel(device="cpu")
        cls.extractor = TableExtractor(cls.model, cls.engine)
        cls.results = {}
        for name, pdf in (("digital", TABLES_PDF), ("scanned", TABLES_SCANNED), ("paper", DIGITAL)):
            work = os.path.join(cls.tmp, name)
            process_pdf(pdf, work)
            detect_document_layout(work, detector)
            run_document_ocr(work, cls.engine)
            cls.results[name] = run_document_tables(work, cls.extractor, visualize=(name == "scanned"))

    def tables(self, name):
        return all_tables(self.results[name]["document"])

    def test_models_load_once(self):
        self.assertEqual((self.model.load_count, self.engine.load_count), (1, 1))

    def test_every_document_gives_a_valid_tables_json(self):
        for name, result in self.results.items():
            with self.subTest(document=name):
                with open(result["tables_path"], encoding="utf-8") as fh:
                    data = json.load(fh)
                self.assertEqual(validate_tables_document(data), [])
                self.assertEqual(data["extractor"]["structure_model"]["name"], "SLANet_plus")
                self.assertEqual(data["extractor"]["ocr_engine"]["name"], "PaddleOCR")
                self.assertEqual(data["summary"]["extraction_failures"], 0)

    def test_digital_tables(self):
        tables = self.tables("digital")
        self.assertEqual([t["source"] for t in tables], ["native", "native", "hybrid"])
        self.assertEqual([table_grid(t) for t in tables], [GRIDS[1], GRIDS[2], GRIDS[3]])
        for t in tables:
            self.assertEqual({c["text_source"] for c in t["cells"] if c["text"]}, {"native"})

        metrics = evaluate_tables(TRUTH, self.results["digital"]["document"])
        self.assertEqual((metrics["found"], metrics["exact_cells"], metrics["total_cells"]), (3, 64, 64))
        for key in ("row_count_accuracy", "column_count_accuracy", "structure_accuracy", "header_accuracy",
                    "caption_accuracy"):
            self.assertEqual(metrics[key], 1.0, key)

    def test_merged_header_from_the_real_model(self):
        merged = self.tables("digital")[2]
        self.assertEqual([(c["row"], c["column"], c["col_span"], c["text"]) for c in merged["cells"] if c["col_span"] > 1],
                         [(0, 1, 2, "Accuracy")])
        self.assertEqual(merged["headers"], ["Model", "Accuracy Top-1", "Accuracy Top-5", "Params"])
        self.assertGreater(merged["structure_confidence"], 0.9)

    def test_scanned_tables(self):
        tables = self.tables("scanned")
        self.assertEqual([t["source"] for t in tables], ["vision"] * 3)
        for t in tables:
            self.assertEqual({c["text_source"] for c in t["cells"] if c["text"]}, {"ocr"})
            self.assertTrue(all(0.5 < c["confidence"] <= 1.0 for c in t["cells"] if c["text"]))

        metrics = evaluate_tables(TRUTH, self.results["scanned"]["document"])
        self.assertEqual(metrics["found"], 3)
        self.assertEqual((metrics["row_count_accuracy"], metrics["column_count_accuracy"]), (1.0, 1.0))
        self.assertGreaterEqual(metrics["exact_cell_accuracy"], 0.95)
        self.assertGreaterEqual(metrics["cell_text_accuracy"], 0.97)
        self.assertEqual(metrics["structure_accuracy"], 1.0)
        self.assertEqual(metrics["header_accuracy"], 1.0)
        self.assertEqual(len(self.results["scanned"]["visualizations"]), 3)

    def test_the_table_of_the_phase_4_fixture(self):
        table, = self.tables("paper")
        self.assertEqual((table["source"], table["page_number"], table["n_rows"], table["n_columns"]), ("native", 2, 4, 4))
        self.assertEqual(table["headers"], ["Model", "Depth", "Accuracy", "Time"])
        self.assertEqual("\n".join(" ".join(row) for row in table_grid(table)), TABLE_TEXT)
        self.assertEqual(table["caption"], "Table 1. Accuracy and training time by depth.")

    def test_hybrid_and_vision_modes_on_native_tables(self):
        work = self.results["digital"]["work_dir"]
        for mode, text_source in (("hybrid", "native"), ("vision", "ocr")):
            document = run_document_tables(work, self.extractor, mode=mode)["document"]
            self.assertEqual([t["source"] for t in all_tables(document)], [mode] * 3)
            self.assertEqual({c["text_source"] for t in all_tables(document) for c in t["cells"] if c["text"]},
                             {text_source})
            metrics = evaluate_tables(TRUTH, document)
            self.assertEqual((metrics["row_count_accuracy"], metrics["column_count_accuracy"]), (1.0, 1.0))
            self.assertGreaterEqual(metrics["exact_cell_accuracy"], 1.0 if mode == "hybrid" else 0.95)
        self.assertEqual((self.model.load_count, self.engine.load_count), (1, 1))
        run_document_tables(work, self.extractor)       # leave the directory as the other tests expect it


@unittest.skipUnless(HAVE_TABLE_MODEL, "paddleocr is not installed or a model test is switched off")
class InspectTablesToolIntegrationTests(TempDirTestCase):
    TOOL = os.path.join(ENGINE, "tools", "inspect_tables.py")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=600)

    def test_pdf_input_then_reuse_of_the_working_directory(self):
        done = self.run_tool(TABLES_PDF, "--output", self.path("out"), "--visualize-tables")

        self.assertEqual(done.returncode, 0, done.stderr)
        for expected in ("Pages: 3", "Table regions: 3", "Tables extracted: 3", "Native: 2", "Vision: 0", "Hybrid: 1",
                         "Cells: 63", "Extraction failures: 0", "Time: ", "OCR engine: PaddleOCR, not loaded",
                         "Phase 2 pages: created; Phase 3 layout: created; Phase 4 OCR: created"):
            self.assertIn(expected, done.stdout)
        self.assertEqual(sorted(os.listdir(self.path("out", "metadata"))),
                         ["layout.json", "ocr.json", "pages.json", "tables.json"])
        self.assertEqual(len(os.listdir(self.path("out", "tables_debug"))), 3)
        with open(self.path("out", "metadata", "tables.json"), encoding="utf-8") as fh:
            first = json.load(fh)
        self.assertEqual(validate_tables_document(first), [])

        earlier = {name: sha256(self.path("out", "metadata", name)) for name in ("pages.json", "layout.json", "ocr.json")}
        done = self.run_tool(self.path("out"), "--table-mode", "hybrid")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Phase 2 pages: reused; Phase 3 layout: reused; Phase 4 OCR: reused", done.stdout)
        self.assertIn("Hybrid: 3", done.stdout)
        self.assertEqual(earlier, {name: sha256(self.path("out", "metadata", name)) for name in earlier})

        evaluated = subprocess.run([sys.executable, os.path.join(ENGINE, "tools", "evaluate_tables.py"),
                                    self.path("out"), "--truth", TRUTH_PATH], capture_output=True, text=True, timeout=120)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("Exact cell accuracy: 1.0000 (64/64)", evaluated.stdout)


if __name__ == "__main__":
    unittest.main()

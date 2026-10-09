"""Phase 4 tests: native/OCR routing, crops, cleanup, the engine interface, ocr.json, CER/WER.

    python -m unittest discover -s tests/python -v

The unit tests use a fake layout detector and a fake OCR engine and need neither PaddleOCR nor a model
download. The classes named *IntegrationTests run the real PP-DocLayoutV3 and PP-OCRv5 models (first
run downloads about 12 MB of OCR models); they are skipped when paddleocr is not installed or when
AIRPA_SKIP_LAYOUT_MODEL=1 or AIRPA_SKIP_OCR_MODEL=1.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

from test_layout import HAVE_MODEL, FakeDetector, TempDirTestCase, det, sha256

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)

from pipeline import (  # noqa: E402
    OCREngine,
    OcrCropError,
    OcrInputError,
    OcrLine,
    OcrModelError,
    PaddleOCREngine,
    PdfPipelineError,
    PPDocLayoutV3Detector,
    cer,
    clean_line,
    compare_ocr_documents,
    crop_box,
    detect_document_layout,
    edit_counts,
    has_current_pages,
    join_lines,
    pdf_bbox_to_pixels,
    process_pdf,
    route_region,
    run_document_ocr,
    should_ocr,
    validate_ocr_document,
    validate_pages_document,
    wer,
)
from pipeline.ocr import TEXT_REGION_TYPES, is_blank, is_usable_text, region_confidence  # noqa: E402

DIGITAL = os.path.join(FIXTURES, "digital", "synthetic_digital.pdf")                # 2 pages, native text
TWO_COLUMN = os.path.join(FIXTURES, "multicolumn", "synthetic_two_column.pdf")
SCANNED_TEXT = os.path.join(FIXTURES, "scanned", "synthetic_scanned_text.pdf")      # the same pages as bitmaps
MIXED = os.path.join(FIXTURES, "mixed", "synthetic_mixed.pdf")
with open(os.path.join(FIXTURES, "mixed", "synthetic_mixed.json"), encoding="utf-8") as _fh:
    MIXED_TRUTH = json.load(_fh)

HAVE_OCR_MODEL = HAVE_MODEL and os.environ.get("AIRPA_SKIP_OCR_MODEL") != "1"

# Regions of the digital fixture as PP-DocLayoutV3 finds them: {page: [(model label, PDF box)]}.
# The scanned copy has the same pages, so the same boxes.
DIGITAL_REGIONS = {
    1: [("doc_title", (70.0, 38.0, 355.6, 56.4)),
        ("paragraph_title", (70.0, 61.6, 123.6, 74.8)),     # "Abstract": same native block as the paragraph
        ("text", (68.4, 78.4, 513.2, 134.4))],
    2: [("table", (69.6, 110.8, 428.4, 173.6)),
        ("figure_title", (70.4, 184.4, 274.0, 197.6))],
}
MIXED_REGIONS = {
    1: [("paragraph_title", (70.0, 74.0, 300.0, 98.0)),
        ("text", (70.0, 108.0, 542.0, 167.0)),
        ("text", tuple(MIXED_TRUTH["image_pdf_bbox"])),     # the paragraph that is only a bitmap
        ("text", (70.0, 263.0, 542.0, 322.0))],
}
TABLE_TEXT = "Model Depth Accuracy Time\nNet-A 8 81.2 12\nNet-B 16 86.5 25\nNet-C 32 88.1 61"


class FakeEngine(OCREngine):
    """Returns one line per call, placed in the middle of the image; counts loads and calls."""

    name = "fake-ocr"
    CONFIDENCE = 0.8123456789

    def __init__(self, fail_on=None, fail_load=False, lines=None):
        super().__init__()
        self.fail_on = fail_on          # 1-based number of the call that raises
        self.fail_load = fail_load
        self.lines = lines              # fixed lines to return instead
        self.sizes = []                 # size of every image received

    def _load_model(self):
        if self.fail_load:
            raise RuntimeError("simulated load failure")
        return object()

    def _recognize(self, image):
        self.sizes.append(image.size)
        if len(self.sizes) == self.fail_on:
            raise RuntimeError("simulated OCR failure")
        if self.lines is not None:
            return list(self.lines)
        box = (10, 10, image.width - 10, image.height - 10)
        corners = ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
        return [OcrLine(f"  recognized   text {len(self.sizes)} ", self.CONFIDENCE, box, corners)]


def region(region_type, *pdf_bbox):
    return {"region_id": "p001_r001", "page_number": 1, "type": region_type, "pdf_bbox": list(pdf_bbox)}


def all_regions(document):
    return [r for page in document["pages"] for r in page["regions"]]


def page_with_ink(width=400, height=300):
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).rectangle([110, 110, 290, 190], fill="black")
    return image


class OcrTestCase(TempDirTestCase):
    def prepare(self, pdf, regions, name="doc"):
        """Phase 2 on `pdf` and a layout.json holding `regions` ({page: [(label, PDF box)]})."""
        work = self.path(name)
        pages = process_pdf(pdf, work)["document"]["pages"]
        detections = {
            f"page_{page['page_number']:03d}.png": [
                det(label, 0.9, *pdf_bbox_to_pixels(box, page["scale_x"], page["scale_y"]))
                for label, box in regions.get(page["page_number"], [])]
            for page in pages}
        detect_document_layout(work, FakeDetector(detections))
        return work


class NativeLineTests(TempDirTestCase):
    """Phase 2 now stores the lines of every native block."""

    def test_lines_rebuild_the_block_text_and_lie_inside_the_block(self):
        for pdf in (DIGITAL, TWO_COLUMN, MIXED):
            document = process_pdf(pdf, self.path(os.path.basename(pdf)))["document"]
            self.assertEqual(validate_pages_document(document), [])
            for page in document["pages"]:
                self.assertTrue(page["blocks"])
                for block in page["blocks"]:
                    with self.subTest(pdf=os.path.basename(pdf), block=block["block_index"]):
                        self.assertEqual("\n".join(line["text"] for line in block["lines"]), block["text"])
                        for line in block["lines"]:
                            x0, y0, x1, y1 = line["bbox"]
                            self.assertTrue(x0 < x1 and y0 < y1)
                            self.assertTrue(block["bbox"][0] - 0.01 <= x0 and x1 <= block["bbox"][2] + 0.01)
                            self.assertTrue(block["bbox"][1] - 0.01 <= y0 and y1 <= block["bbox"][3] + 0.01)

    def test_a_heading_and_its_paragraph_share_a_block_but_not_a_line(self):
        block = process_pdf(DIGITAL, self.path("doc"))["document"]["pages"][0]["blocks"][1]
        self.assertTrue(block["text"].startswith("Abstract\nNeural networks"))
        self.assertEqual(block["lines"][0]["text"], "Abstract")
        self.assertLess(block["lines"][0]["bbox"][3], block["lines"][1]["bbox"][3])

    def test_validator_checks_lines(self):
        document = process_pdf(TWO_COLUMN, self.path("doc"))["document"]
        document["pages"][0]["blocks"][0]["lines"][0]["bbox"] = [10, 10, 5, 20]
        self.assertTrue(any("lines[0].bbox" in p for p in validate_pages_document(document)))
        # A pages.json from before Phase 4, without lines, is still valid.
        for block in document["pages"][0]["blocks"]:
            del block["lines"]
        self.assertEqual(validate_pages_document(document), [])

    def test_has_current_pages(self):
        work = self.path("doc")
        self.assertFalse(has_current_pages(work, DIGITAL))
        process_pdf(DIGITAL, work, dpi=100)
        self.assertTrue(has_current_pages(work, DIGITAL, dpi=100))
        self.assertFalse(has_current_pages(work, DIGITAL))                  # other DPI
        self.assertFalse(has_current_pages(work, TWO_COLUMN, dpi=100))      # other PDF
        os.remove(self.path("doc", "pages", "page_002.png"))
        self.assertFalse(has_current_pages(work, DIGITAL, dpi=100))


class RoutingTests(unittest.TestCase):
    PAGE = {"has_text_layer": True, "blocks": [
        {"block_index": 0, "bbox": [72.0, 34.9, 353.8, 56.9], "text": "Effect of Depth",
         "lines": [{"bbox": [72.0, 34.9, 353.8, 56.9], "text": "Effect of Depth"}]},
        {"block_index": 1, "bbox": [72.0, 59.2, 510.3, 106.0], "text": "Abstract\nNeural networks are\nwidely used.",
         "lines": [{"bbox": [72.0, 59.2, 120.7, 75.7], "text": "Abstract"},
                   {"bbox": [72.0, 78.0, 510.3, 92.0], "text": "Neural networks are"},
                   {"bbox": [72.0, 92.0, 300.0, 106.0], "text": "widely  used. "}]},
    ]}
    SCANNED_PAGE = {"has_text_layer": False, "blocks": []}

    def test_region_with_native_text_keeps_it(self):
        routing = route_region(region("TITLE", 70.0, 38.0, 355.6, 56.4), self.PAGE)

        self.assertEqual((routing.decision, routing.reason), ("native", "native_text_in_region"))
        self.assertEqual(routing.native_text, "Effect of Depth")
        self.assertEqual(routing.native_blocks, (0,))
        self.assertFalse(should_ocr(region("TITLE", 70.0, 38.0, 355.6, 56.4), self.PAGE))

    def test_a_block_shared_by_two_regions_is_split_by_its_lines(self):
        heading = route_region(region("TITLE", 70.0, 61.6, 123.6, 74.8), self.PAGE)
        paragraph = route_region(region("TEXT", 68.4, 77.0, 513.2, 107.0), self.PAGE)

        self.assertEqual(heading.native_text, "Abstract")
        self.assertEqual(paragraph.native_text, "Neural networks are\nwidely used.")
        self.assertEqual((heading.native_blocks, paragraph.native_blocks), ((1,), (1,)))

    def test_region_without_native_text_needs_ocr(self):
        empty = region("TEXT", 68.0, 300.0, 513.0, 360.0)
        routing = route_region(empty, self.PAGE)

        self.assertEqual((routing.decision, routing.reason), ("ocr", "no_native_text_in_region"))
        self.assertIsNone(routing.native_text)
        self.assertTrue(should_ocr(empty, self.PAGE))

    def test_the_decision_is_per_region_not_per_page(self):
        # Same page, has_text_layer True: one region is native, the other needs OCR.
        self.assertFalse(should_ocr(region("TEXT", 68.4, 77.0, 513.2, 107.0), self.PAGE))
        self.assertTrue(should_ocr(region("CAPTION", 68.0, 500.0, 513.0, 520.0), self.PAGE))

    def test_page_without_text_layer(self):
        routing = route_region(region("TEXT", 68.0, 78.0, 513.0, 134.0), self.SCANNED_PAGE)
        self.assertEqual((routing.decision, routing.reason), ("ocr", "page_has_no_text_layer"))

    def test_a_line_belongs_to_the_region_holding_most_of_it(self):
        # One native line, 14 pt high and 438.3 pt wide.
        line_only = {"has_text_layer": True, "blocks": [
            {"block_index": 0, "bbox": [72.0, 78.0, 510.3, 92.0], "text": "Neural networks are",
             "lines": [{"bbox": [72.0, 78.0, 510.3, 92.0], "text": "Neural networks are"}]}]}
        self.assertTrue(should_ocr(region("TEXT", 68.0, 60.0, 513.0, 84.9), line_only))      # 6.9 of 14 pt
        self.assertFalse(should_ocr(region("TEXT", 68.0, 60.0, 513.0, 85.1), line_only))     # 7.1 of 14 pt
        self.assertTrue(should_ocr(region("TEXT", 300.0, 76.0, 513.0, 94.0), line_only))     # under half its width

    def test_unusable_native_text_is_replaced_by_ocr(self):
        for text in ("���� a", " b", "— •", "..."):
            page = {"has_text_layer": True, "blocks": [
                {"block_index": 0, "bbox": [72.0, 78.0, 510.3, 92.0], "text": text,
                 "lines": [{"bbox": [72.0, 78.0, 510.3, 92.0], "text": text}]}]}
            with self.subTest(text=text):
                routing = route_region(region("TEXT", 68.0, 76.0, 513.0, 94.0), page)
                self.assertEqual((routing.decision, routing.reason), ("ocr", "native_text_unusable"))
        self.assertTrue(is_usable_text("7"))                    # a page number
        self.assertTrue(is_usable_text("Résumé αβ"))
        self.assertFalse(is_usable_text(""))
        self.assertFalse(is_usable_text(None))

    def test_every_text_class_is_an_ocr_candidate(self):
        self.assertEqual({t.value for t in TEXT_REGION_TYPES},
                         {"TEXT", "TITLE", "CAPTION", "HEADER", "FOOTER", "PAGE_NUMBER", "LIST", "OTHER"})
        for region_type in TEXT_REGION_TYPES:
            self.assertTrue(should_ocr(region(region_type.value, 68.0, 300.0, 513.0, 360.0), self.PAGE))

    def test_equations_and_figures_get_no_text(self):
        for region_type in ("EQUATION", "FIGURE"):
            for page in (self.PAGE, self.SCANNED_PAGE):
                routing = route_region(region(region_type, 70.0, 38.0, 355.6, 56.4), page, ocr_tables=True)
                self.assertEqual((routing.decision, routing.reason), ("skip", "region_type_has_no_text"))
                self.assertIsNone(routing.native_text)

    def test_tables(self):
        with_text = region("TABLE", 70.0, 38.0, 355.6, 56.4)
        without = region("TABLE", 68.0, 300.0, 513.0, 360.0)

        self.assertEqual(route_region(with_text, self.PAGE).decision, "native")
        self.assertEqual(route_region(with_text, self.PAGE, ocr_tables=True).decision, "native")
        skipped = route_region(without, self.PAGE)
        self.assertEqual((skipped.decision, skipped.reason), ("skip", "table_ocr_not_requested"))
        self.assertEqual(route_region(without, self.PAGE, ocr_tables=True).decision, "ocr")
        self.assertEqual(route_region(without, self.SCANNED_PAGE, ocr_tables=True).reason, "page_has_no_text_layer")


class CropTests(unittest.TestCase):
    def test_box_is_rounded_outward_and_padded(self):
        self.assertEqual(crop_box([100.2, 50.7, 300.5, 90.1], 1000, 800, padding=8), (92, 42, 309, 99))
        self.assertEqual(crop_box([100.2, 50.7, 300.5, 90.1], 1000, 800, padding=0), (100, 50, 301, 91))
        self.assertEqual(crop_box((100, 50, 300, 90), 1000, 800), (92, 42, 308, 98))    # default padding 8

    def test_padding_stops_at_the_page_edges(self):
        self.assertEqual(crop_box([2, 3, 995, 798], 1000, 800, padding=8), (0, 0, 1000, 800))
        self.assertEqual(crop_box([0, 0, 1000, 800], 1000, 800, padding=50), (0, 0, 1000, 800))
        self.assertEqual(crop_box([990, 100, 1000, 200], 1000, 800, padding=8), (982, 92, 1000, 208))

    def test_partially_out_of_bounds_box_is_clipped(self):
        self.assertEqual(crop_box([-40, -10, 120, 60], 1000, 800, padding=0), (0, 0, 120, 60))
        self.assertEqual(crop_box([-40, -10, 120, 60], 1000, 800, padding=8), (0, 0, 128, 68))
        self.assertEqual(crop_box([900, 700, 1200, 900], 1000, 800, padding=8), (892, 692, 1000, 800))

    def test_invalid_and_empty_boxes(self):
        for bad in ([300, 50, 100, 90],             # inverted
                    [100, 50, 100, 90],             # no width
                    [1200, 50, 1400, 90],           # right of the page
                    [100, -90, 300, -50],           # above the page
                    [100, 50, float("nan"), 90],
                    [100, 50, 300],
                    None):
            with self.subTest(bbox=bad):
                with self.assertRaises(OcrCropError):
                    crop_box(bad, 1000, 800)

    def test_tiny_boxes(self):
        for tiny in ([10, 10, 13, 40], [10, 10, 40, 14.5], [997, 100, 1100, 200]):   # the last: 3 px left on the page
            with self.subTest(bbox=tiny):
                with self.assertRaises(OcrCropError) as caught:
                    crop_box(tiny, 1000, 800)
                self.assertIn("too small", str(caught.exception))
        self.assertEqual(crop_box([10, 10, 16, 16], 1000, 800, padding=0), (10, 10, 16, 16))

    def test_invalid_padding(self):
        for bad in (-1, 2.5, "8", None, True):
            with self.subTest(padding=bad):
                with self.assertRaises(ValueError):
                    crop_box([100, 50, 300, 90], 1000, 800, padding=bad)

    def test_blank_image(self):
        self.assertTrue(is_blank(Image.new("RGB", (50, 20), "white")))
        self.assertTrue(is_blank(Image.new("RGB", (50, 20), (40, 40, 40))))
        self.assertFalse(is_blank(page_with_ink()))


class CleanupTests(unittest.TestCase):
    def test_clean_line(self):
        self.assertEqual(clean_line("  two   words \t here\r\n"), "two words here")
        self.assertEqual(clean_line(" \n "), "")
        # Nothing is corrected.
        self.assertEqual(clean_line("teh rn0del is 1O% bet-"), "teh rn0del is 1O% bet-")

    def test_lines_are_ordered_top_to_bottom(self):
        lines = [((0, 80, 100, 100), "third"), ((0, 10, 100, 30), " first "), ((0, 45, 100, 65), "second  line")]
        self.assertEqual(join_lines(lines), "first\nsecond line\nthird")

    def test_fragments_of_one_row_are_joined_left_to_right(self):
        lines = [((300, 12, 380, 29), "Time"), ((0, 10, 80, 30), "Model"), ((150, 11, 230, 31), "Depth"),
                 ((0, 50, 80, 70), "Net-A"), ((150, 51, 170, 69), "8")]
        self.assertEqual(join_lines(lines), "Model Depth Time\nNet-A 8")

    def test_empty_fragments_are_dropped(self):
        self.assertEqual(join_lines([((0, 10, 100, 30), "  "), ((0, 40, 100, 60), "text")]), "text")
        self.assertEqual(join_lines([]), "")

    def test_region_confidence_is_weighted_by_length(self):
        lines = [OcrLine("aaaaaaaa", 1.0, (0, 0, 80, 10)), OcrLine("bb", 0.5, (0, 20, 20, 30))]
        self.assertAlmostEqual(region_confidence(lines), 0.9)
        self.assertIsNone(region_confidence([]))


class EngineTests(unittest.TestCase):
    def test_model_loads_once(self):
        engine = FakeEngine()
        self.assertFalse(engine.loaded)
        self.assertEqual(engine.load_count, 0)

        for _ in range(3):
            engine.recognize_region(page_with_ink(), [100, 100, 300, 200])
        engine.recognize_page(page_with_ink())
        engine.load()

        self.assertTrue(engine.loaded)
        self.assertEqual(engine.load_count, 1)
        self.assertEqual(len(engine.sizes), 4)

    def test_load_failure(self):
        with self.assertRaises(OcrModelError):
            FakeEngine(fail_load=True).load()

    def test_region_lines_are_returned_in_page_pixels(self):
        engine = FakeEngine()
        result = engine.recognize_region(page_with_ink(), [100, 100, 300, 200], padding=8)

        self.assertEqual(result["crop_bbox"], [92, 92, 308, 208])
        self.assertEqual(engine.sizes, [(216, 116)])
        self.assertEqual(result["crop"].size, (216, 116))
        self.assertFalse(result["blank"])
        line = result["lines"][0]
        self.assertEqual(line.bbox, (102.0, 102.0, 298.0, 198.0))            # crop (10, 10, 206, 106) + (92, 92)
        self.assertEqual(line.polygon, ((102.0, 102.0), (298.0, 102.0), (298.0, 198.0), (102.0, 198.0)))
        self.assertEqual(line.confidence, FakeEngine.CONFIDENCE)
        self.assertEqual(line.text, "  recognized   text 1 ")               # raw; cleaned when it is stored

    def test_text_of_a_neighbour_caught_by_the_padding_is_left_out(self):
        lines = [OcrLine("inside", 0.9, (10, 40, 200, 70)),
                 OcrLine("sliver above", 0.4, (0, 0, 216, 6)),              # centre at y 95, region starts at 100
                 OcrLine("sliver right", 0.4, (210, 40, 216, 70))]
        result = FakeEngine(lines=lines).recognize_region(page_with_ink(), [100, 100, 300, 200], padding=8)
        self.assertEqual([line.text for line in result["lines"]], ["inside"])

    def test_blank_region_is_not_sent_to_the_engine(self):
        engine = FakeEngine()
        result = engine.recognize_region(Image.new("RGB", (400, 300), "white"), [100, 100, 300, 200])

        self.assertTrue(result["blank"])
        self.assertEqual(result["lines"], [])
        self.assertEqual((engine.sizes, engine.load_count), ([], 0))

    def test_unusable_lines_from_the_engine_are_dropped(self):
        lines = [OcrLine("good", 0.9, (10, 40, 200, 70)),
                 OcrLine("   ", 0.9, (10, 40, 200, 70)),                    # no text
                 OcrLine("nan", float("nan"), (10, 40, 200, 70)),
                 OcrLine("high", 1.5, (10, 40, 200, 70)),
                 OcrLine("inverted", 0.9, (200, 40, 10, 70)),
                 OcrLine("outside", 0.9, (900, 40, 990, 70))]
        result = FakeEngine(lines=lines).recognize_region(page_with_ink(), [100, 100, 300, 200])
        self.assertEqual([line.text for line in result["lines"]], ["good"])

    def test_invalid_region(self):
        engine = FakeEngine()
        for bad in ([300, 100, 100, 200], [500, 100, 700, 200], [100, 100, 103, 200]):
            with self.assertRaises(OcrCropError):
                engine.recognize_region(page_with_ink(), bad)
        self.assertEqual(engine.sizes, [])

    def test_whole_page(self):
        lines = FakeEngine().recognize_page(page_with_ink())
        self.assertEqual([line.bbox for line in lines], [(10.0, 10.0, 390.0, 290.0)])

    def test_paddle_engine_describes_itself_without_loading(self):
        engine = PaddleOCREngine()
        self.assertEqual(engine.name, "PaddleOCR")
        self.assertEqual({k: engine.describe()[k] for k in ("name", "device", "detection_model", "recognition_model")},
                         {"name": "PaddleOCR", "device": "cpu", "detection_model": "PP-OCRv5_mobile_det",
                          "recognition_model": "en_PP-OCRv5_mobile_rec"})
        self.assertFalse(engine.loaded)


class DocumentOcrTests(OcrTestCase):
    def test_digital_pdf_keeps_native_text_and_never_runs_ocr(self):
        work = self.prepare(DIGITAL, DIGITAL_REGIONS)
        engine = FakeEngine(fail_load=True)         # loading it would fail the run

        result = run_document_ocr(work, engine)

        regions = all_regions(result["document"])
        self.assertEqual([r["text_source"] for r in regions], ["native"] * 5)
        self.assertEqual([r["canonical_type"] for r in regions], ["TITLE", "TITLE", "TEXT", "TABLE", "CAPTION"])
        self.assertEqual(regions[0]["text"], "Effect of Network Depth on Accuracy")
        self.assertEqual(regions[1]["text"], "Abstract")
        self.assertTrue(regions[2]["text"].startswith("Neural networks are widely used"))
        self.assertTrue(regions[2]["text"].endswith("compare their accuracy."))
        self.assertEqual(regions[3]["text"], TABLE_TEXT)
        self.assertEqual(regions[4]["text"], "Table 1. Accuracy and training time by depth.")
        for r in regions:
            self.assertIsNone(r["confidence"])
            self.assertIsNone(r["engine"])
            self.assertEqual(r["routing"], {"decision": "native", "reason": "native_text_in_region"})
            self.assertNotIn("lines", r)
        self.assertEqual([r["native_blocks"] for r in regions], [[0], [1], [1], [1], [2]])

        self.assertEqual((engine.load_count, engine.sizes), (0, []))
        self.assertEqual(result["timings"]["engine_init_s"], 0.0)
        self.assertEqual(result["timings"]["region_ocr_s"], {})
        summary = result["document"]["summary"]
        self.assertEqual((summary["text_regions"], summary["native_regions"], summary["ocr_regions"]), (5, 5, 0))
        self.assertIsNone(summary["mean_ocr_confidence"])

    def test_scanned_pdf_text_regions_are_ocred(self):
        work = self.prepare(SCANNED_TEXT, DIGITAL_REGIONS)
        engine = FakeEngine()

        result = run_document_ocr(work, engine)

        regions = all_regions(result["document"])
        self.assertEqual([p["has_text_layer"] for p in result["document"]["pages"]], [False, False])
        self.assertEqual([r["text_source"] for r in regions], ["ocr", "ocr", "ocr", "none", "ocr"])
        self.assertEqual([r["text"] for r in regions],
                         ["recognized text 1", "recognized text 2", "recognized text 3", None, "recognized text 4"])
        self.assertEqual(regions[0]["routing"], {"decision": "ocr", "reason": "page_has_no_text_layer"})
        self.assertEqual(regions[3]["routing"], {"decision": "skip", "reason": "table_ocr_not_requested"})
        self.assertEqual(len(engine.sizes), 4)
        summary = result["document"]["summary"]
        self.assertEqual((summary["text_regions"], summary["native_regions"], summary["ocr_regions"],
                          summary["skipped_regions"], summary["ocr_failures"]), (4, 0, 4, 1, 0))
        self.assertEqual(sorted(result["timings"]["region_ocr_s"]), ["p001_r001", "p001_r002", "p001_r003", "p002_r002"])
        self.assertAlmostEqual(result["timings"]["ocr_s"], sum(result["timings"]["region_ocr_s"].values()))

    def test_tables_are_ocred_as_raw_text_only_on_request(self):
        work = self.prepare(SCANNED_TEXT, DIGITAL_REGIONS)
        table = all_regions(run_document_ocr(work, FakeEngine(), ocr_tables=True)["document"])[3]

        self.assertEqual((table["canonical_type"], table["text_source"], table["text"]),
                         ("TABLE", "ocr", "recognized text 4"))
        # Text lines only: nothing about rows, columns or cells.
        self.assertFalse({"rows", "columns", "cells", "html", "latex"} & set(table))

    def test_mixed_page_routes_every_region_on_its_own(self):
        work = self.prepare(MIXED, MIXED_REGIONS)
        engine = FakeEngine()

        document = run_document_ocr(work, engine)["document"]

        regions = all_regions(document)
        self.assertTrue(document["pages"][0]["has_text_layer"])
        self.assertEqual([r["text_source"] for r in regions], ["native", "native", "ocr", "native"])
        self.assertEqual(regions[0]["text"], MIXED_TRUTH["heading"])
        self.assertEqual(" ".join(regions[1]["text"].split()), MIXED_TRUTH["native_above"])
        self.assertEqual(" ".join(regions[3]["text"].split()), MIXED_TRUTH["native_below"])
        self.assertEqual(regions[2]["text"], "recognized text 1")
        self.assertEqual(regions[2]["routing"], {"decision": "ocr", "reason": "no_native_text_in_region"})
        self.assertEqual(len(engine.sizes), 1)      # only the bitmap paragraph went to the engine

    def test_ocr_result_keeps_confidence_and_geometry(self):
        work = self.prepare(MIXED, MIXED_REGIONS)
        document = run_document_ocr(work, FakeEngine(), padding=8)["document"]
        with open(os.path.join(work, "metadata", "pages.json"), encoding="utf-8") as fh:
            page = json.load(fh)["pages"][0]
        ocr = all_regions(document)[2]

        self.assertEqual(ocr["engine"], "fake-ocr")
        self.assertAlmostEqual(ocr["confidence"], FakeEngine.CONFIDENCE, places=12)
        self.assertEqual(ocr["raw_text"], "  recognized   text 1 ")
        self.assertEqual(ocr["text"], "recognized text 1")
        self.assertEqual(ocr["pdf_bbox"], MIXED_TRUTH["image_pdf_bbox"])
        x0, y0, x1, y1 = ocr["crop_bbox"]
        self.assertEqual(ocr["crop_bbox"], [math.floor(ocr["image_bbox"][0]) - 8, math.floor(ocr["image_bbox"][1]) - 8,
                                            math.ceil(ocr["image_bbox"][2]) + 8, math.ceil(ocr["image_bbox"][3]) + 8])

        self.assertEqual(len(ocr["lines"]), 1)
        line = ocr["lines"][0]
        self.assertEqual(set(line), {"text", "confidence", "image_bbox", "pdf_bbox", "polygon"})
        self.assertEqual(line["confidence"], FakeEngine.CONFIDENCE)         # not rounded
        self.assertEqual(line["text"], "recognized text 1")
        self.assertEqual(line["image_bbox"], [x0 + 10.0, y0 + 10.0, x1 - 10.0, y1 - 10.0])
        self.assertEqual(len(line["polygon"]), 4)
        for i, scale in enumerate((page["scale_x"], page["scale_y"]) * 2):
            self.assertAlmostEqual(line["pdf_bbox"][i], line["image_bbox"][i] / scale, places=6)

    def test_every_text_says_where_it_comes_from(self):
        work = self.prepare(MIXED, MIXED_REGIONS)
        for r in all_regions(run_document_ocr(work, FakeEngine())["document"]):
            self.assertIn(r["text_source"], ("native", "ocr"))
            if r["text_source"] == "native":
                self.assertEqual((r["confidence"], r["engine"], r["routing"]["decision"]), (None, None, "native"))
                self.assertIn("native_blocks", r)
                self.assertFalse({"lines", "raw_text", "crop_bbox"} & set(r))
            else:
                self.assertEqual((r["engine"], r["routing"]["decision"]), ("fake-ocr", "ocr"))
                self.assertIsInstance(r["confidence"], float)
                self.assertNotIn("native_blocks", r)

    def test_ocr_json(self):
        work = self.prepare(MIXED, MIXED_REGIONS)
        result = run_document_ocr(work, FakeEngine(), padding=4, ocr_tables=True)

        self.assertEqual(result["ocr_path"], os.path.join(work, "metadata", "ocr.json"))
        with open(result["ocr_path"], "rb") as fh:
            raw = fh.read()
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data, result["document"])
        self.assertEqual(validate_ocr_document(data), [])
        self.assertEqual(json.loads(json.dumps(data)), data)

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["engine"], {"name": "fake-ocr"})
        self.assertEqual(data["settings"], {"padding_px": 4, "ocr_tables": True, "line_inside_min": 0.5})
        self.assertEqual(data["source"]["source_pdf"], "synthetic_mixed.pdf")
        self.assertEqual((data["source"]["pages_json"], data["source"]["layout_json"]),
                         ("metadata/pages.json", "metadata/layout.json"))
        self.assertEqual(data["coordinate_system"]["bbox_format"], "x0,y0,x1,y1")
        self.assertEqual(data["summary"], {
            "pages": 1, "regions": 4, "text_regions": 4, "native_regions": 3, "ocr_regions": 1, "ocr_empty": 0,
            "ocr_failures": 0, "skipped_regions": 0, "mean_ocr_confidence": data["pages"][0]["regions"][2]["confidence"]})
        self.assertEqual([r["region_id"] for r in data["pages"][0]["regions"]],
                         ["p001_r001", "p001_r002", "p001_r003", "p001_r004"])
        for r in data["pages"][0]["regions"]:
            self.assertLessEqual({"region_id", "page_number", "canonical_type", "source_label", "image_bbox",
                                  "pdf_bbox", "text", "text_source", "confidence", "engine", "routing"}, set(r))

    def test_ocr_json_is_identical_across_runs(self):
        work = self.prepare(MIXED, MIXED_REGIONS)
        with open(run_document_ocr(work, FakeEngine())["ocr_path"], "rb") as fh:
            first = fh.read()
        with open(run_document_ocr(work, FakeEngine())["ocr_path"], "rb") as fh:
            self.assertEqual(first, fh.read())

    def test_earlier_artifacts_are_not_modified(self):
        work = self.prepare(SCANNED_TEXT, DIGITAL_REGIONS)
        names = ("metadata/pages.json", "metadata/layout.json", "pages/page_001.png", "pages/page_002.png")
        before = {name: sha256(os.path.join(work, *name.split("/"))) for name in names}

        run_document_ocr(work, FakeEngine(), save_crops=True)

        self.assertEqual(before, {name: sha256(os.path.join(work, *name.split("/"))) for name in names})
        self.assertEqual(sorted(os.listdir(os.path.join(work, "metadata"))), ["layout.json", "ocr.json", "pages.json"])
        self.assertEqual(sorted(os.listdir(os.path.join(work, "pages"))), ["page_001.png", "page_002.png"])

    def test_engine_is_loaded_once_for_all_pages_and_documents(self):
        engine = FakeEngine()
        first = run_document_ocr(self.prepare(SCANNED_TEXT, DIGITAL_REGIONS, "a"), engine)

        self.assertEqual(engine.load_count, 1)
        self.assertEqual(len(engine.sizes), 4)                  # regions of both pages
        self.assertEqual(set(first["timings"]), {"engine_init_s", "ocr_s", "region_ocr_s", "total_s"})

        second = run_document_ocr(self.prepare(MIXED, MIXED_REGIONS, "b"), engine)
        self.assertEqual(engine.load_count, 1)
        self.assertEqual(second["timings"]["engine_init_s"], 0.0)

    def test_a_failing_region_is_reported_and_the_rest_is_kept(self):
        work = self.prepare(SCANNED_TEXT, DIGITAL_REGIONS)
        document = run_document_ocr(work, FakeEngine(fail_on=2))["document"]

        regions = all_regions(document)
        self.assertEqual([r["text_source"] for r in regions], ["ocr", "none", "ocr", "none", "ocr"])
        failed = regions[1]
        self.assertEqual((failed["text"], failed["confidence"], failed["engine"]), (None, None, None))
        self.assertEqual(failed["error"], "RuntimeError: simulated OCR failure")
        self.assertEqual(failed["routing"]["decision"], "ocr")
        self.assertNotIn("error", regions[3])                   # the skipped table is not a failure
        summary = document["summary"]
        self.assertEqual((summary["text_regions"], summary["ocr_regions"], summary["ocr_failures"]), (4, 3, 1))
        self.assertEqual(validate_ocr_document(document), [])

    def test_a_region_too_small_to_read_is_a_reported_failure(self):
        regions = {1: [("text", (70.0, 38.0, 355.6, 56.4)), ("number", (300.0, 700.0, 301.0, 720.0))]}     # 2.5 px wide
        engine = FakeEngine()
        document = run_document_ocr(self.prepare(SCANNED_TEXT, regions), engine)["document"]

        ok, tiny = all_regions(document)
        self.assertEqual(ok["text_source"], "ocr")
        self.assertEqual((tiny["text_source"], tiny["text"]), ("none", None))
        self.assertIn("too small", tiny["error"])
        self.assertEqual(len(engine.sizes), 1)
        self.assertEqual(document["summary"]["ocr_failures"], 1)

    def test_a_region_without_ink_gives_empty_ocr_text(self):
        regions = {1: [("text", (70.0, 600.0, 500.0, 700.0))]}      # empty paper at the bottom of page 1
        engine = FakeEngine()
        document = run_document_ocr(self.prepare(SCANNED_TEXT, regions), engine)["document"]

        empty = all_regions(document)[0]
        self.assertEqual((empty["text_source"], empty["text"], empty["confidence"], empty["lines"]),
                         ("ocr", "", None, []))
        self.assertNotIn("error", empty)
        self.assertEqual(engine.sizes, [])
        self.assertEqual((document["summary"]["ocr_empty"], document["summary"]["ocr_failures"]), (1, 0))
        self.assertEqual(validate_ocr_document(document), [])

    def test_crops_are_saved_only_on_request(self):
        work = self.prepare(MIXED, MIXED_REGIONS)

        result = run_document_ocr(work, FakeEngine())
        self.assertEqual(result["crops"], [])
        self.assertFalse(os.path.exists(os.path.join(work, "ocr_crops")))

        result = run_document_ocr(work, FakeEngine(), save_crops=True)
        self.assertEqual(os.listdir(os.path.join(work, "ocr_crops")), ["page_001_region_003.png"])     # the OCR'd one
        self.assertEqual(result["crops"], [os.path.join(work, "ocr_crops", "page_001_region_003.png")])
        x0, y0, x1, y1 = all_regions(result["document"])[2]["crop_bbox"]
        with Image.open(result["crops"][0]) as crop:
            self.assertEqual(crop.size, (x1 - x0, y1 - y0))

    def test_missing_or_mismatched_artifacts(self):
        with self.assertRaises(PdfPipelineError) as caught:
            run_document_ocr(self.path("nothing-here"), FakeEngine())
        self.assertIn("pages.json", str(caught.exception))

        process_pdf(DIGITAL, self.path("no-layout"))
        with self.assertRaises(OcrInputError):
            run_document_ocr(self.path("no-layout"), FakeEngine())

        # A layout.json of another document.
        work = self.prepare(DIGITAL, DIGITAL_REGIONS, "a")
        other = self.prepare(TWO_COLUMN, {}, "b")
        shutil.copyfile(os.path.join(other, "metadata", "layout.json"), os.path.join(work, "metadata", "layout.json"))
        with self.assertRaises(OcrInputError):
            run_document_ocr(work, FakeEngine())

    def test_pages_json_from_before_phase_4_is_rejected(self):
        work = self.prepare(DIGITAL, DIGITAL_REGIONS)
        path = os.path.join(work, "metadata", "pages.json")
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
        for page in document["pages"]:
            for block in page["blocks"]:
                del block["lines"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(document, fh)

        with self.assertRaises(OcrInputError) as caught:
            run_document_ocr(work, FakeEngine())
        self.assertIn("run Phase 2 again", str(caught.exception))

    def test_a_failed_run_leaves_no_ocr_json(self):
        work = self.prepare(SCANNED_TEXT, DIGITAL_REGIONS)
        run_document_ocr(work, FakeEngine())
        self.assertTrue(os.path.isfile(os.path.join(work, "metadata", "ocr.json")))

        with self.assertRaises(OcrModelError):
            run_document_ocr(work, FakeEngine(fail_load=True))
        self.assertFalse(os.path.exists(os.path.join(work, "metadata", "ocr.json")))

        with self.assertRaises(ValueError):
            run_document_ocr(work, FakeEngine(), padding=-3)

    def test_validator_reports_problems(self):
        work = self.prepare(MIXED, MIXED_REGIONS)
        data = run_document_ocr(work, FakeEngine())["document"]
        self.assertTrue(validate_ocr_document([]))
        self.assertTrue(validate_ocr_document({"engine": {"name": "x"}, "pages": []}))

        def broken(index, change):
            copy = json.loads(json.dumps(data))
            change(copy["pages"][0]["regions"][index])
            return validate_ocr_document(copy)

        self.assertTrue(broken(0, lambda r: r.update(confidence=0.9)))          # native text has no confidence
        self.assertTrue(broken(0, lambda r: r.update(text_source="pdf")))
        self.assertTrue(broken(0, lambda r: r.update(canonical_type="paragraph")))
        self.assertTrue(broken(2, lambda r: r.update(confidence=None)))         # OCR text needs one
        self.assertTrue(broken(2, lambda r: r.update(engine=None)))
        self.assertTrue(broken(2, lambda r: r["lines"][0].pop("confidence")))
        self.assertTrue(broken(2, lambda r: r["lines"][0].update(image_bbox=[5, 5, 1, 9])))
        self.assertTrue(broken(2, lambda r: r.pop("routing")))


class ErrorRateTests(unittest.TestCase):
    def test_edit_counts(self):
        self.assertEqual(edit_counts("abc", "abc"), (0, 0, 0))
        self.assertEqual(edit_counts("kitten", "sitting"), (2, 0, 1))       # k->s, e->i, +g
        self.assertEqual(edit_counts("abcd", "abd"), (0, 1, 0))             # c is missing
        self.assertEqual(edit_counts("abd", "abcd"), (0, 0, 1))
        self.assertEqual(edit_counts("abc", ""), (0, 3, 0))
        self.assertEqual(edit_counts("", "ab"), (0, 0, 2))
        self.assertEqual(edit_counts("", ""), (0, 0, 0))
        self.assertEqual(edit_counts(["a", "b"], ["a", "c"]), (1, 0, 0))    # any sequence

    def test_cer(self):
        self.assertEqual(cer("hello world", "hello world"), 0.0)
        self.assertAlmostEqual(cer("hello world", "hallo world"), 1 / 11)       # S=1, N=11
        self.assertAlmostEqual(cer("hello world", "helo world"), 1 / 11)        # D=1
        self.assertAlmostEqual(cer("hello world", "hello worlds"), 1 / 11)      # I=1
        self.assertAlmostEqual(cer("accuracy", "acurrasy!"), 4 / 8)             # four edits
        self.assertEqual(cer("abc", ""), 1.0)
        self.assertEqual(cer("ab", "abcdef"), 2.0)                              # can exceed 1
        self.assertGreater(cer("Depth", "depth"), 0)                            # case counts

    def test_cer_ignores_only_whitespace_layout(self):
        self.assertEqual(cer("two lines\nof text", "  two   lines of\ttext\n"), 0.0)
        self.assertAlmostEqual(cer("ab cd", "abcd"), 1 / 5)                     # a lost space is an error

    def test_wer(self):
        self.assertEqual(wer("the quick brown fox", "the quick brown fox"), 0.0)
        self.assertEqual(wer("the quick brown fox", "the quack brown"), 0.5)    # S=1, D=1, N=4
        self.assertEqual(wer("the quick brown fox", "the the quick brown fox"), 0.25)
        self.assertEqual(wer("one\ntwo", "one two"), 0.0)
        self.assertEqual(wer("word", "w0rd"), 1.0)

    def test_empty_reference(self):
        for function in (cer, wer):
            for empty in ("", "  \n "):
                with self.assertRaises(ValueError):
                    function(empty, "text")

    def test_compare_documents(self):
        def document(*regions):
            return {"pages": [{"page_number": 1, "regions": [
                {"region_id": f"p001_r{i:03d}", "page_number": 1, "canonical_type": "TEXT", "pdf_bbox": list(bbox),
                 "text": text, "text_source": source, "confidence": confidence}
                for i, (bbox, text, source, confidence) in enumerate(regions, 1)]}]}

        reference = document(((0, 0, 100, 20), "hello world", "native", None),
                             ((0, 30, 100, 90), "the quick brown fox", "native", None),
                             ((0, 200, 100, 220), "never scanned", "native", None))
        hypothesis = document(((0, 31, 100, 91), "the quack brown fox", "ocr", 0.8),
                              ((0, 0, 100, 20), "hello world", "ocr", 0.99),
                              ((0, 400, 100, 420), "no reference here", "ocr", 0.9),
                              ((0, 200, 100, 220), "never scanned", "native", None))     # not OCR: not compared

        result = compare_ocr_documents(reference, hypothesis)

        self.assertEqual([(r["reference_region_id"], r["region_id"]) for r in result["rows"]],
                         [("p001_r001", "p001_r002"), ("p001_r002", "p001_r001")])
        self.assertEqual([r["cer"] for r in result["rows"]], [0.0, 1 / 19])
        self.assertEqual([r["wer"] for r in result["rows"]], [0.0, 0.25])
        self.assertEqual((result["regions"], result["unmatched"], result["characters"]), (2, 1, 30))
        self.assertAlmostEqual(result["cer"], 1 / 30)
        self.assertAlmostEqual(result["wer"], 1 / 6)

        nothing = compare_ocr_documents(reference, document())
        self.assertEqual((nothing["rows"], nothing["cer"], nothing["wer"]), ([], None, None))


class InspectOcrToolErrorTests(TempDirTestCase):
    """Failures that are reported before any model is needed."""
    TOOL = os.path.join(ENGINE, "tools", "inspect_ocr.py")

    def test_bad_input_exits_with_code_2_and_no_traceback(self):
        with open(self.path("bad.pdf"), "wb") as fh:
            fh.write(b"nope")
        os.makedirs(self.path("empty-dir"))
        cases = ([self.path("missing.pdf")], [self.path("bad.pdf"), "--output", self.path("out")],
                 [self.path("empty-dir")], [DIGITAL, "--output", self.path("out"), "--ocr-padding", "-1"],
                 [DIGITAL, "--output", self.path("out"), "--layout-threshold", "1.5"])
        for args in cases:
            with self.subTest(args=args):
                done = subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=300)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn("Error:", done.stderr)
                self.assertNotIn("Traceback", done.stderr)


@unittest.skipUnless(HAVE_OCR_MODEL, "paddleocr is not installed or a model test is switched off")
class OcrModelIntegrationTests(unittest.TestCase):
    """The real layout and OCR models on the fixtures. One detector and one engine for all documents."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="airpa-p4-model-")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        detector = PPDocLayoutV3Detector(device="cpu")
        cls.engine = PaddleOCREngine(device="cpu")
        cls.results = {}
        for name, pdf in (("digital", DIGITAL), ("scanned", SCANNED_TEXT), ("mixed", MIXED)):
            work = os.path.join(cls.tmp, name)
            process_pdf(pdf, work)
            detect_document_layout(work, detector)
            cls.results[name] = run_document_ocr(work, cls.engine, save_crops=(name == "mixed"))

    def regions(self, name):
        return all_regions(self.results[name]["document"])

    def test_engine_loads_once(self):
        self.assertEqual(self.engine.load_count, 1)
        self.assertEqual(self.results["digital"]["timings"]["engine_init_s"], 0.0)      # not needed yet
        self.assertGreater(self.results["scanned"]["timings"]["engine_init_s"], 0)
        self.assertEqual(self.results["mixed"]["timings"]["engine_init_s"], 0.0)        # already loaded

    def test_every_document_gives_a_valid_ocr_json(self):
        for name, result in self.results.items():
            with self.subTest(document=name):
                with open(result["ocr_path"], encoding="utf-8") as fh:
                    data = json.load(fh)
                self.assertEqual(validate_ocr_document(data), [])
                self.assertEqual(data["engine"]["name"], "PaddleOCR")
                self.assertEqual(data["engine"]["device"], "cpu")
                self.assertRegex(data["engine"]["version"], r"^\d+\.\d+")
                self.assertEqual(data["summary"]["ocr_failures"], 0)

    def test_digital_pdf_needs_no_ocr(self):
        regions = self.regions("digital")
        self.assertGreaterEqual(len(regions), 12)
        self.assertEqual({r["text_source"] for r in regions}, {"native"})
        self.assertEqual(self.results["digital"]["document"]["summary"]["ocr_regions"], 0)
        self.assertEqual(self.results["digital"]["timings"]["region_ocr_s"], {})
        texts = [r["text"] for r in regions]
        # Heading and paragraph of one native block end up in their own regions.
        self.assertIn("Abstract", texts)
        self.assertIn("1. Introduction", texts)
        self.assertIn(TABLE_TEXT, texts)
        self.assertFalse(any(t.startswith("Abstract\n") for t in texts))

    def test_scanned_pdf_is_read_by_ocr(self):
        regions = self.regions("scanned")
        text_regions = [r for r in regions if r["canonical_type"] != "TABLE"]
        self.assertGreaterEqual(len(text_regions), 12)
        self.assertEqual({r["text_source"] for r in text_regions}, {"ocr"})
        self.assertEqual({r["routing"]["reason"] for r in text_regions}, {"page_has_no_text_layer"})
        self.assertEqual([r["text_source"] for r in regions if r["canonical_type"] == "TABLE"], ["none"])
        texts = [r["text"] for r in regions]
        self.assertIn("Effect of Network Depth on Accuracy", texts)
        self.assertIn("Abstract", texts)

        for r in text_regions:
            self.assertEqual(r["engine"], "PaddleOCR")
            self.assertTrue(0.5 < r["confidence"] <= 1.0, r)
            self.assertTrue(r["lines"])
            cx0, cy0, cx1, cy1 = r["crop_bbox"]
            for line in r["lines"]:
                self.assertTrue(0.0 < line["confidence"] <= 1.0)
                x0, y0, x1, y1 = line["image_bbox"]
                self.assertTrue(cx0 <= x0 < x1 <= cx1 and cy0 <= y0 < y1 <= cy1, line)
                self.assertEqual(len(line["polygon"]), 4)
        # A paragraph of several lines keeps one entry per line.
        self.assertGreaterEqual(max(len(r["lines"]) for r in text_regions), 4)

    def test_character_and_word_error_rate_against_the_digital_original(self):
        comparison = compare_ocr_documents(self.results["digital"]["document"], self.results["scanned"]["document"])

        self.assertGreaterEqual(comparison["regions"], 12)
        self.assertEqual(comparison["unmatched"], 0)
        self.assertGreater(comparison["characters"], 2000)
        self.assertLess(comparison["cer"], 0.02)
        self.assertLess(comparison["wer"], 0.05)

    def test_mixed_page(self):
        regions = self.regions("mixed")
        by_source = {source: [r for r in regions if r["text_source"] == source] for source in ("native", "ocr")}

        self.assertEqual(len(by_source["ocr"]), 1)
        self.assertGreaterEqual(len(by_source["native"]), 3)
        native_text = " ".join(" ".join(r["text"].split()) for r in by_source["native"])
        for key in ("heading", "native_above", "native_below"):
            self.assertIn(MIXED_TRUTH[key], native_text)
        self.assertNotIn("pasted into the page", native_text)

        ocr = by_source["ocr"][0]
        self.assertEqual(ocr["routing"], {"decision": "ocr", "reason": "no_native_text_in_region"})
        self.assertLess(cer(MIXED_TRUTH["image_text"], ocr["text"]), 0.05)
        self.assertLess(wer(MIXED_TRUTH["image_text"], ocr["text"]), 0.10)
        self.assertGreater(ocr["confidence"], 0.8)
        self.assertEqual([os.path.basename(p) for p in self.results["mixed"]["crops"]],
                         [f"page_001_region_{regions.index(ocr) + 1:03d}.png"])

    def test_scanned_table_as_raw_text_on_request(self):
        work = self.results["scanned"]["work_dir"]
        table = next(r for r in all_regions(run_document_ocr(work, self.engine, ocr_tables=True)["document"])
                     if r["canonical_type"] == "TABLE")

        self.assertEqual(table["text_source"], "ocr")
        self.assertLess(cer(TABLE_TEXT, table["text"]), 0.05)
        self.assertEqual(self.engine.load_count, 1)

    def test_whole_page_recognition(self):
        with Image.open(os.path.join(self.results["scanned"]["work_dir"], "pages", "page_001.png")) as source:
            lines = self.engine.recognize_page(source.convert("RGB"))
        self.assertGreater(len(lines), 15)
        self.assertIn("Abstract", [line.text for line in lines])


@unittest.skipUnless(HAVE_OCR_MODEL, "paddleocr is not installed or a model test is switched off")
class InspectOcrToolIntegrationTests(TempDirTestCase):
    TOOL = os.path.join(ENGINE, "tools", "inspect_ocr.py")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=600)

    def test_pdf_input_then_reuse_of_the_working_directory(self):
        done = self.run_tool(MIXED, "--output", self.path("out"), "--save-ocr-crops")

        self.assertEqual(done.returncode, 0, done.stderr)
        for expected in ("Pages: 1", "Text regions: 4", "Native text used: 3", "OCR regions: 1", "OCR failures: 0",
                         "Average OCR confidence: 0.9", "OCR time: ", "PP-OCRv5_mobile_det", "device CPU, loaded",
                         "Phase 2 pages: created; Phase 3 layout: created", "engine init"):
            self.assertIn(expected, done.stdout)
        self.assertEqual(sorted(os.listdir(self.path("out", "metadata"))), ["layout.json", "ocr.json", "pages.json"])
        self.assertEqual(len(os.listdir(self.path("out", "ocr_crops"))), 1)
        with open(self.path("out", "metadata", "ocr.json"), encoding="utf-8") as fh:
            first = json.load(fh)
        self.assertEqual(validate_ocr_document(first), [])

        # Again, from the PDF and from the directory: Phase 2 and Phase 3 are not repeated.
        earlier = {name: sha256(self.path("out", "metadata", name)) for name in ("pages.json", "layout.json")}
        for args in ([MIXED, "--output", self.path("out")], [self.path("out")]):
            done = self.run_tool(*args)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("Phase 2 pages: reused; Phase 3 layout: reused", done.stdout)
            self.assertEqual(earlier, {name: sha256(self.path("out", "metadata", name)) for name in earlier})
            with open(self.path("out", "metadata", "ocr.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), first)

    def test_digital_pdf_does_not_load_the_ocr_engine(self):
        done = self.run_tool(DIGITAL, "--output", self.path("out"))

        self.assertEqual(done.returncode, 0, done.stderr)
        for expected in ("Pages: 2", "OCR regions: 0", "OCR failures: 0", "Average OCR confidence: n/a",
                         "not loaded: no region needed OCR"):
            self.assertIn(expected, done.stdout)
        self.assertNotIn("Native text used: 0", done.stdout)


if __name__ == "__main__":
    unittest.main()

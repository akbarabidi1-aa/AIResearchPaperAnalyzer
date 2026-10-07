"""Phase 3 tests: region schema, label normalization, coordinate mapping, layout.json, metrics.

    python -m unittest discover -s tests/python -v

The unit tests use a fake detector and need neither PaddleOCR nor a model download. The classes named
*IntegrationTests run the real PP-DocLayoutV3 model (first run downloads about 125 MB); they are skipped
when paddleocr is not installed or when AIRPA_SKIP_LAYOUT_MODEL=1.
"""
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)

from pipeline import (  # noqa: E402
    DEFAULT_LAYOUT_THRESHOLD,
    LayoutDetectionError,
    LayoutDetector,
    LayoutInputError,
    LayoutModelError,
    PPDocLayoutV3Detector,
    RawDetection,
    Region,
    RegionType,
    bbox_iou,
    class_accuracy,
    count_region_types,
    detect_document_layout,
    image_bbox_to_pdf,
    match_regions,
    normalize_detections,
    normalize_label,
    pdf_bbox_to_pixels,
    process_pdf,
    validate_layout_document,
)
from pipeline.common import pymupdf  # noqa: E402
from pipeline.layout import PP_DOCLAYOUT_V3_LABEL_MAP, clip_bbox, find_overlap_issues  # noqa: E402

DIGITAL = os.path.join(FIXTURES, "digital", "synthetic_digital.pdf")          # 2 pages
TWO_COLUMN = os.path.join(FIXTURES, "multicolumn", "synthetic_two_column.pdf")
SCANNED = os.path.join(FIXTURES, "scanned", "synthetic_scanned.pdf")

# label_list of the model's inference.yml.
PP_DOCLAYOUT_V3_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title",
    "figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image",
    "image", "inline_formula", "number", "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]

REGION_KEYS = {"region_id", "page_number", "type", "score", "image_bbox", "pdf_bbox", "source_label", "source_model"}

HAVE_MODEL = (importlib.util.find_spec("paddleocr") is not None
              and os.environ.get("AIRPA_SKIP_LAYOUT_MODEL") != "1")

# A US Letter page at 180 DPI, as Phase 2 stores it.
LETTER_PAGE = {"page_number": 1, "pdf_width": 612.0, "pdf_height": 792.0,
               "image_width": 1530, "image_height": 1980, "scale_x": 2.5, "scale_y": 2.5}


class FakeDetector(LayoutDetector):
    """Returns fixed detections per page image name; counts loads and calls."""

    name = "fake-layout"
    label_map = PP_DOCLAYOUT_V3_LABEL_MAP

    def __init__(self, detections=None, fail_on=None, fail_load=False):
        super().__init__()
        self.detections = detections or {}
        self.fail_on = fail_on
        self.fail_load = fail_load
        self.calls = []

    def _load_model(self):
        if self.fail_load:
            raise RuntimeError("simulated load failure")
        return object()

    def _detect(self, image_path, threshold):
        name = os.path.basename(image_path)
        self.calls.append((name, threshold))
        if name == self.fail_on:
            raise RuntimeError("simulated inference failure")
        return list(self.detections.get(name, self.detections.get("*", [])))


def det(label, score, *bbox):
    return RawDetection(label=label, score=score, bbox=tuple(bbox))


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def assert_boxes_inside_page(test, page):
    """0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, in image and in PDF coordinates."""
    for region in page["regions"]:
        for key, width, height in (("image_bbox", page["image_width"], page["image_height"]),
                                   ("pdf_bbox", page["pdf_width"], page["pdf_height"])):
            x0, y0, x1, y1 = region[key]
            test.assertTrue(0 <= x0 < x1 <= width, f"{region['region_id']} {key} x: {region[key]}")
            test.assertTrue(0 <= y0 < y1 <= height, f"{region['region_id']} {key} y: {region[key]}")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="airpa-p3-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)


class LabelNormalizationTests(unittest.TestCase):
    def test_every_model_label_has_a_documented_mapping(self):
        self.assertEqual(sorted(PP_DOCLAYOUT_V3_LABEL_MAP), sorted(PP_DOCLAYOUT_V3_LABELS))
        for label, region_type in PP_DOCLAYOUT_V3_LABEL_MAP.items():
            self.assertIsInstance(region_type, RegionType, label)

    def test_canonical_classes(self):
        self.assertEqual([t.value for t in RegionType],
                         ["TITLE", "TEXT", "TABLE", "FIGURE", "EQUATION", "CAPTION", "HEADER", "FOOTER",
                          "PAGE_NUMBER", "LIST", "OTHER"])

    def test_mapping_of_the_main_labels(self):
        expected = {
            "doc_title": "TITLE", "paragraph_title": "TITLE", "text": "TEXT", "abstract": "TEXT",
            "reference": "TEXT", "algorithm": "TEXT", "table": "TABLE", "image": "FIGURE", "chart": "FIGURE",
            "display_formula": "EQUATION", "inline_formula": "EQUATION", "figure_title": "CAPTION",
            "header": "HEADER", "footer": "FOOTER", "number": "PAGE_NUMBER", "content": "LIST",
            "footnote": "OTHER", "seal": "OTHER", "formula_number": "OTHER",
        }
        for label, region_type in expected.items():
            self.assertEqual(normalize_label(label, PP_DOCLAYOUT_V3_LABEL_MAP).value, region_type, label)

    def test_unknown_label_becomes_other_and_is_kept(self):
        self.assertIs(normalize_label("hologram", PP_DOCLAYOUT_V3_LABEL_MAP), RegionType.OTHER)

        regions, _ = normalize_detections([det("hologram", 0.9, 10, 10, 50, 50)], LETTER_PAGE, 0.5,
                                          PP_DOCLAYOUT_V3_LABEL_MAP, "fake-layout")
        self.assertEqual((regions[0].type, regions[0].source_label), (RegionType.OTHER, "hologram"))

    def test_source_label_and_model_are_preserved(self):
        detections = [det(label, 0.9, 10, 10 + i, 200, 400 + i) for i, label in enumerate(PP_DOCLAYOUT_V3_LABELS)]
        regions, discarded = normalize_detections(detections, LETTER_PAGE, 0.5, PP_DOCLAYOUT_V3_LABEL_MAP, "m")

        self.assertEqual(discarded, 0)
        self.assertEqual([r.source_label for r in regions], PP_DOCLAYOUT_V3_LABELS)
        self.assertEqual({r.source_model for r in regions}, {"m"})
        for region in regions:
            self.assertIs(region.type, PP_DOCLAYOUT_V3_LABEL_MAP[region.source_label])


class CoordinateConversionTests(TempDirTestCase):
    def test_image_to_pdf_divides_by_the_stored_scale(self):
        self.assertEqual(image_bbox_to_pdf((250, 500, 1000, 1500), 2.5, 2.5), (100.0, 200.0, 400.0, 600.0))
        self.assertEqual(image_bbox_to_pdf((30, 40, 60, 100), 3.0, 2.0), (10.0, 20.0, 20.0, 50.0))

    def test_it_is_the_inverse_of_the_phase_2_mapping(self):
        for scale_x, scale_y in ((2.5, 2.5), (1.3888888888888888, 1.3901515151515151), (0.5, 8.3)):
            box = [72.0, 34.9, 353.8, 56.9]
            there = pdf_bbox_to_pixels(box, scale_x, scale_y)
            back = image_bbox_to_pdf(there, scale_x, scale_y)
            for a, b in zip(box, back):
                self.assertAlmostEqual(a, b, places=9)

    def test_the_whole_image_maps_to_the_whole_page(self):
        regions, _ = normalize_detections([det("image", 0.9, 0, 0, 1530, 1980)], LETTER_PAGE, 0.5,
                                          PP_DOCLAYOUT_V3_LABEL_MAP, "m")
        self.assertEqual(regions[0].image_bbox, (0.0, 0.0, 1530.0, 1980.0))
        self.assertEqual(regions[0].pdf_bbox, (0.0, 0.0, 612.0, 792.0))

    def test_boxes_beyond_the_image_are_clipped(self):
        regions, discarded = normalize_detections([det("text", 0.9, -20, -5.5, 1600, 2100.25)], LETTER_PAGE, 0.5,
                                                  PP_DOCLAYOUT_V3_LABEL_MAP, "m")
        self.assertEqual(discarded, 0)
        self.assertEqual(regions[0].image_bbox, (0.0, 0.0, 1530.0, 1980.0))
        self.assertEqual(regions[0].pdf_bbox, (0.0, 0.0, 612.0, 792.0))

    def test_boxes_inside_the_image_are_not_changed(self):
        regions, _ = normalize_detections([det("text", 0.9, 175.5, 103, 326, 137.25)], LETTER_PAGE, 0.5,
                                          PP_DOCLAYOUT_V3_LABEL_MAP, "m")
        self.assertEqual(regions[0].image_bbox, (175.5, 103.0, 326.0, 137.25))
        self.assertEqual(regions[0].pdf_bbox, (70.2, 41.2, 130.4, 54.9))

    def test_bounds_hold_on_real_phase_2_pages(self):
        with pymupdf.open() as doc:
            doc.new_page(width=500.5, height=333.3).insert_text((60, 80), "An odd-sized page with one line of text.")
            doc.save(self.path("odd.pdf"))

        for pdf, dpi in ((DIGITAL, 180), (DIGITAL, 100), (TWO_COLUMN, 72), (self.path("odd.pdf"), 100)):
            with self.subTest(pdf=os.path.basename(pdf), dpi=dpi):
                work = self.path(f"{os.path.basename(pdf)}-{dpi}")
                page0 = process_pdf(pdf, work, dpi=dpi)["document"]["pages"][0]
                w, h = page0["image_width"], page0["image_height"]
                detector = FakeDetector({"*": [
                    det("text", 0.9, 0, 0, w, h),                   # exactly the page
                    det("table", 0.9, w - 1, h - 1, w, h),          # last pixel
                    det("image", 0.9, -50, -50, w + 50, h + 50),    # larger than the page
                    det("text", 0.9, w * 0.25, h * 0.4, w * 0.75, h * 0.6),
                ]})

                document = detect_document_layout(work, detector)["document"]

                self.assertEqual(validate_layout_document(document), [])
                for page in document["pages"]:
                    self.assertEqual(len(page["regions"]), 4)
                    assert_boxes_inside_page(self, page)
                    # The stored Phase 2 scale is used: pdf = image / scale.
                    for region in page["regions"]:
                        for i, scale in enumerate((page0["scale_x"], page0["scale_y"]) * 2):
                            self.assertAlmostEqual(region["pdf_bbox"][i], region["image_bbox"][i] / scale, places=6)

    def test_phase_2_block_boxes_survive_the_round_trip_through_image_space(self):
        work = self.path("doc")
        pages = process_pdf(TWO_COLUMN, work, dpi=100)["document"]["pages"]    # non-integer scale
        page = pages[0]
        detector = FakeDetector({"*": [
            det("text", 0.9, *pdf_bbox_to_pixels(block["bbox"], page["scale_x"], page["scale_y"]))
            for block in page["blocks"]]})

        regions = detect_document_layout(work, detector)["document"]["pages"][0]["regions"]

        self.assertEqual(len(regions), len(page["blocks"]))
        for region, block in zip(regions, page["blocks"]):
            for a, b in zip(region["pdf_bbox"], block["bbox"]):
                self.assertAlmostEqual(a, b, places=6)


class InvalidBoxTests(unittest.TestCase):
    def test_clip_bbox(self):
        self.assertEqual(clip_bbox([10, 20, 30, 40], 100, 100), (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(clip_bbox((-5, -5, 50, 50), 40, 30), (0.0, 0.0, 40.0, 30.0))
        for bad in ([30, 20, 10, 40],               # inverted x
                    [10, 40, 30, 20],               # inverted y
                    [10, 20, 10, 40],               # no width
                    [10, 20, 30, 20],               # no height
                    [120, 10, 150, 40],             # right of the page
                    [10, -40, 30, -5],              # above the page
                    [10, 20, 30],                   # three values
                    [10, 20, 30, 40, 50],
                    [10, 20, float("nan"), 40],
                    [10, 20, float("inf"), 40],
                    ["10", 20, 30, 40],
                    [True, 20, 30, 40],
                    None):
            with self.subTest(bbox=bad):
                self.assertIsNone(clip_bbox(bad, 100, 100))

    def test_invalid_detections_are_discarded_and_counted(self):
        detections = [
            det("text", 0.9, 100, 100, 400, 300),
            det("text", 0.9, 400, 100, 100, 300),           # inverted
            det("text", 0.9, 100, 100, 100, 300),           # empty
            det("text", 0.9, 2000, 100, 2400, 300),         # outside the image
            det("text", 0.9, 100, float("nan"), 400, 300),
            RawDetection("text", 0.9, (100, 100, 400)),
            det("text", float("nan"), 100, 100, 400, 300),  # no usable score
            det("table", 0.8, 100, 400, 400, 600),
        ]

        regions, discarded = normalize_detections(detections, LETTER_PAGE, 0.5, PP_DOCLAYOUT_V3_LABEL_MAP, "m")

        self.assertEqual(discarded, 6)
        self.assertEqual([r.region_id for r in regions], ["p001_r001", "p001_r002"])    # ids stay consecutive
        self.assertEqual([r.type for r in regions], [RegionType.TEXT, RegionType.TABLE])


class ThresholdTests(TempDirTestCase):
    DETECTIONS = [det("text", 0.95, 10, 10, 200, 100), det("table", 0.5, 10, 200, 200, 300),
                  det("image", 0.49, 10, 400, 200, 500), det("text", 0.1, 10, 600, 200, 700)]

    def scores(self, threshold):
        regions, discarded = normalize_detections(self.DETECTIONS, LETTER_PAGE, threshold,
                                                  PP_DOCLAYOUT_V3_LABEL_MAP, "m")
        self.assertEqual(discarded, 0)      # below the threshold is not "invalid"
        return [r.score for r in regions]

    def test_default_is_the_model_default(self):
        self.assertEqual(DEFAULT_LAYOUT_THRESHOLD, 0.5)

    def test_detections_below_the_threshold_are_left_out(self):
        self.assertEqual(self.scores(0.5), [0.95, 0.5])         # a score equal to the threshold is kept
        self.assertEqual(self.scores(0.0), [0.95, 0.5, 0.49, 0.1])
        self.assertEqual(self.scores(0.3), [0.95, 0.5, 0.49])
        self.assertEqual(self.scores(0.96), [])
        self.assertEqual(self.scores(1), [])

    def test_raw_scores_are_not_rounded(self):
        regions, _ = normalize_detections([det("text", 0.8493318557739258, 10, 10, 200, 100)], LETTER_PAGE, 0.5,
                                          PP_DOCLAYOUT_V3_LABEL_MAP, "m")
        self.assertEqual(regions[0].score, 0.8493318557739258)

    def test_invalid_thresholds(self):
        for bad in (-0.1, 1.5, "0.5", None, True, float("nan")):
            with self.subTest(threshold=bad):
                with self.assertRaises(ValueError):
                    normalize_detections([], LETTER_PAGE, bad, PP_DOCLAYOUT_V3_LABEL_MAP, "m")
                with self.assertRaises(ValueError):
                    FakeDetector().detect("page.png", bad)

    def test_threshold_reaches_the_detector_and_layout_json(self):
        process_pdf(TWO_COLUMN, self.path("doc"))
        detector = FakeDetector({"*": self.DETECTIONS})

        document = detect_document_layout(self.path("doc"), detector, threshold=0.3)["document"]

        self.assertEqual(detector.calls, [("page_001.png", 0.3)])
        self.assertEqual(document["model"]["threshold"], 0.3)
        self.assertEqual([r["score"] for r in document["pages"][0]["regions"]], [0.95, 0.5, 0.49])


class LayoutDocumentTests(TempDirTestCase):
    DETECTIONS = {
        "page_001.png": [det("doc_title", 0.91, 175, 87, 889, 141), det("text", 0.93, 171, 196, 1283, 336)],
        "page_002.png": [det("table", 0.929018497467041, 174, 277, 1071, 434),
                         det("vision_footnote", 0.537, 176, 461, 685, 494)],
    }

    def run_layout(self, pdf=DIGITAL, **kwargs):
        process_pdf(pdf, self.path("doc"))
        self.detector = FakeDetector(dict(self.DETECTIONS))
        return detect_document_layout(self.path("doc"), self.detector, **kwargs)

    def test_region_round_trip(self):
        region = Region("p003_r002", 3, RegionType.EQUATION, 0.75, (10.0, 20.0, 30.0, 40.0),
                        (4.0, 8.0, 12.0, 16.0), "display_formula", "PP-DocLayoutV3")
        data = region.to_dict()

        self.assertEqual(data, {
            "region_id": "p003_r002", "page_number": 3, "type": "EQUATION", "score": 0.75,
            "image_bbox": [10.0, 20.0, 30.0, 40.0], "pdf_bbox": [4.0, 8.0, 12.0, 16.0],
            "source_label": "display_formula", "source_model": "PP-DocLayoutV3"})
        self.assertEqual(Region.from_dict(json.loads(json.dumps(data))), region)
        with self.assertRaises(ValueError):
            Region.from_dict({**data, "type": "PARAGRAPH"})

    def test_layout_json_structure(self):
        result = self.run_layout()

        self.assertEqual(result["layout_path"], self.path("doc", "metadata", "layout.json"))
        with open(result["layout_path"], "rb") as fh:
            raw = fh.read()
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data, result["document"])
        self.assertEqual(validate_layout_document(data), [])

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["model"], {"name": "fake-layout", "threshold": 0.5})
        self.assertEqual(data["source"]["source_pdf"], "synthetic_digital.pdf")
        self.assertEqual(data["source"]["pages_json"], "metadata/pages.json")
        self.assertEqual(data["coordinate_system"]["bbox_format"], "x0,y0,x1,y1")
        self.assertEqual([p["page_number"] for p in data["pages"]], [1, 2])
        self.assertEqual([p["image_path"] for p in data["pages"]], ["pages/page_001.png", "pages/page_002.png"])

        regions = [r for p in data["pages"] for r in p["regions"]]
        self.assertEqual([r["region_id"] for r in regions], ["p001_r001", "p001_r002", "p002_r001", "p002_r002"])
        self.assertEqual([r["type"] for r in regions], ["TITLE", "TEXT", "TABLE", "CAPTION"])
        self.assertEqual([r["source_label"] for r in regions], ["doc_title", "text", "table", "vision_footnote"])
        self.assertEqual(regions[2]["score"], 0.929018497467041)
        self.assertEqual(regions[2]["image_bbox"], [174.0, 277.0, 1071.0, 434.0])
        self.assertEqual(regions[2]["pdf_bbox"], [69.6, 110.8, 428.4, 173.6])
        self.assertEqual(count_region_types(data), {"TITLE": 1, "TEXT": 1, "TABLE": 1, "CAPTION": 1})
        # Small file: locations only, no image data.
        self.assertLess(len(raw), 5000)

    def test_regions_carry_only_location_and_class(self):
        # Phase 3 scope: no text, cells, LaTeX or descriptions on a region.
        for page in self.run_layout()["document"]["pages"]:
            for region in page["regions"]:
                self.assertEqual(set(region), REGION_KEYS)

    def test_phase_2_artifacts_are_not_modified(self):
        process_pdf(DIGITAL, self.path("doc"))
        before = {name: sha256(self.path("doc", *name.split("/")))
                  for name in ("metadata/pages.json", "pages/page_001.png", "pages/page_002.png")}

        detect_document_layout(self.path("doc"), FakeDetector(dict(self.DETECTIONS)), visualize=True)

        after = {name: sha256(self.path("doc", *name.split("/"))) for name in before}
        self.assertEqual(before, after)
        self.assertEqual(sorted(os.listdir(self.path("doc", "metadata"))), ["layout.json", "pages.json"])
        self.assertEqual(sorted(os.listdir(self.path("doc", "pages"))), ["page_001.png", "page_002.png"])

    def test_layout_json_is_identical_across_runs(self):
        first = self.run_layout()["layout_path"]
        with open(first, "rb") as fh:
            a = fh.read()
        with open(self.run_layout()["layout_path"], "rb") as fh:
            self.assertEqual(a, fh.read())

    def test_model_is_loaded_once_for_all_pages(self):
        result = self.run_layout()

        self.assertEqual(self.detector.load_count, 1)
        self.assertEqual([name for name, _ in self.detector.calls], ["page_001.png", "page_002.png"])
        timings = result["timings"]
        self.assertEqual(set(timings), {"model_init_s", "inference_s", "page_inference_s", "total_s"})
        self.assertEqual(len(timings["page_inference_s"]), 2)
        self.assertAlmostEqual(timings["inference_s"], sum(timings["page_inference_s"]))

        # A second document with the same detector does not load again.
        process_pdf(TWO_COLUMN, self.path("other"))
        again = detect_document_layout(self.path("other"), self.detector)
        self.assertEqual(self.detector.load_count, 1)
        self.assertEqual(again["timings"]["model_init_s"], 0.0)

    def test_page_without_detections(self):
        process_pdf(SCANNED, self.path("doc"))
        document = detect_document_layout(self.path("doc"), FakeDetector())["document"]

        self.assertEqual(validate_layout_document(document), [])
        self.assertEqual(document["pages"][0]["regions"], [])
        self.assertEqual(count_region_types(document), {})

    def test_missing_or_broken_phase_2_artifacts(self):
        with self.assertRaises(LayoutInputError):
            detect_document_layout(self.path("nothing-here"), FakeDetector())

        process_pdf(DIGITAL, self.path("doc"))
        os.remove(self.path("doc", "pages", "page_002.png"))
        detector = FakeDetector()
        with self.assertRaises(LayoutInputError):
            detect_document_layout(self.path("doc"), detector)
        self.assertFalse(os.path.exists(self.path("doc", "metadata", "layout.json")))

        with open(self.path("doc", "metadata", "pages.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaises(LayoutInputError):
            detect_document_layout(self.path("doc"), FakeDetector())

    def test_a_failing_page_fails_the_run_and_leaves_no_layout_json(self):
        self.run_layout()
        self.assertTrue(os.path.isfile(self.path("doc", "metadata", "layout.json")))

        detector = FakeDetector(dict(self.DETECTIONS), fail_on="page_002.png")
        with self.assertRaises(LayoutDetectionError) as caught:
            detect_document_layout(self.path("doc"), detector)

        self.assertEqual(caught.exception.page_number, 2)
        self.assertFalse(os.path.exists(self.path("doc", "metadata", "layout.json")))

    def test_model_load_failure(self):
        process_pdf(TWO_COLUMN, self.path("doc"))
        with self.assertRaises(LayoutModelError):
            detect_document_layout(self.path("doc"), FakeDetector(fail_load=True))

    def test_validator_reports_problems(self):
        data = self.run_layout()["document"]
        self.assertTrue(validate_layout_document([]))
        self.assertTrue(validate_layout_document({"model": {}, "pages": []}))

        def broken(change):
            copy = json.loads(json.dumps(data))
            change(copy["pages"][0]["regions"][0])
            return validate_layout_document(copy)

        self.assertTrue(any("outside the page" in p for p in broken(lambda r: r.update(pdf_bbox=[10, 10, 5000, 20]))))
        self.assertTrue(any("not a valid box" in p for p in broken(lambda r: r.update(image_bbox=[50, 10, 20, 20]))))
        self.assertTrue(any("canonical class" in p for p in broken(lambda r: r.update(type="paragraph_title"))))
        self.assertTrue(any("score" in p for p in broken(lambda r: r.update(score=1.2))))
        self.assertTrue(any("source_label" in p for p in broken(lambda r: r.pop("source_label"))))
        self.assertTrue(any("region_id" in p for p in broken(lambda r: r.update(region_id="p001_r002"))))


class VisualizationTests(TempDirTestCase):
    DETECTIONS = {"*": [det("paragraph_title", 0.85, 175, 103, 326, 137), det("table", 0.93, 174, 277, 1071, 434),
                        det("text", 0.9, 0, 0, 400, 40)]}     # the last one touches the top edge

    def test_not_created_by_default(self):
        process_pdf(TWO_COLUMN, self.path("doc"))
        result = detect_document_layout(self.path("doc"), FakeDetector(self.DETECTIONS))

        self.assertEqual(result["visualizations"], [])
        self.assertFalse(os.path.exists(self.path("doc", "layout_debug")))

    def test_one_annotated_copy_per_page(self):
        process_pdf(DIGITAL, self.path("doc"))
        result = detect_document_layout(self.path("doc"), FakeDetector(self.DETECTIONS), visualize=True)

        self.assertEqual(sorted(os.listdir(self.path("doc", "layout_debug"))),
                         ["page_001_layout.png", "page_002_layout.png"])
        self.assertEqual(result["visualizations"],
                         [self.path("doc", "layout_debug", f"page_00{n}_layout.png") for n in (1, 2)])
        for number, drawn in enumerate(result["visualizations"], 1):
            original = pymupdf.Pixmap(self.path("doc", "pages", f"page_00{number}.png"))
            annotated = pymupdf.Pixmap(drawn)
            self.assertEqual((annotated.width, annotated.height), (original.width, original.height))
            self.assertNotEqual(annotated.samples, original.samples)
            # The table outline is drawn in the TABLE colour on its left edge.
            self.assertEqual(annotated.pixel(174, 350)[:3], (20, 140, 60))
            self.assertEqual(original.pixel(174, 350)[:3], (255, 255, 255))


class OverlapInspectionTests(unittest.TestCase):
    def regions(self, *detections):
        return normalize_detections(list(detections), LETTER_PAGE, 0.0, PP_DOCLAYOUT_V3_LABEL_MAP, "m")[0]

    def issues(self, *detections):
        return find_overlap_issues(self.regions(*detections), 1530, 1980)

    def test_clean_page(self):
        self.assertEqual(self.issues(det("text", 0.9, 100, 100, 700, 400), det("text", 0.9, 800, 100, 1400, 400),
                                     det("table", 0.9, 100, 500, 1400, 900)), [])

    def test_same_class_duplicate(self):
        issues = self.issues(det("text", 0.9, 100, 100, 700, 400), det("text", 0.6, 102, 101, 705, 402))

        self.assertEqual([(i["kind"], i["region_ids"]) for i in issues], [("duplicate", ["p001_r001", "p001_r002"])])
        self.assertGreater(issues[0]["value"], 0.9)

    def test_overlap_of_different_classes_or_small_overlap_is_not_a_duplicate(self):
        self.assertEqual(self.issues(det("image", 0.9, 100, 100, 700, 400), det("table", 0.6, 100, 100, 700, 400)), [])
        self.assertEqual(self.issues(det("text", 0.9, 100, 100, 700, 400), det("text", 0.6, 400, 100, 1000, 400)), [])

    def test_page_sized_box(self):
        issues = self.issues(det("image", 0.9, 10, 10, 1520, 1970), det("text", 0.9, 100, 100, 700, 400))

        self.assertEqual([(i["kind"], i["region_ids"]) for i in issues], [("page_sized", ["p001_r001"])])


class IouTests(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([1.5, 2.5, 7.25, 9.75], (1.5, 2.5, 7.25, 9.75)), 1.0)

    def test_disjoint_and_touching_boxes(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)      # shared edge
        self.assertEqual(bbox_iou([0, 0, 10, 10], [10, 10, 20, 20]), 0.0)     # shared corner
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 30, 10, 40]), 0.0)      # same columns, apart in y

    def test_partial_overlap(self):
        # 10x10 boxes shifted by 5 in x: intersection 50, union 150.
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        # shifted by 5 in both directions: intersection 25, union 175.
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_contained_box(self):
        # 5x5 inside 10x10: intersection 25, union 100.
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [2, 2, 7, 7]), 0.25)
        # a thin strip across a box: intersection 10x2, union 100 + 40 - 20.
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [-5, 4, 15, 6]), 20 / 120)

    def test_symmetric_and_scale_free(self):
        a, b = [3.2, 1.1, 40.5, 22.0], [10.0, 5.0, 60.0, 30.0]
        self.assertAlmostEqual(bbox_iou(a, b), bbox_iou(b, a))
        self.assertAlmostEqual(bbox_iou(a, b), bbox_iou([v * 2.5 for v in a], [v * 2.5 for v in b]))
        self.assertTrue(0 < bbox_iou(a, b) < 1)

    def test_same_value_in_image_and_pdf_space(self):
        a_px, b_px = (175, 103, 326, 137), (150, 100, 300, 150)
        self.assertAlmostEqual(bbox_iou(a_px, b_px),
                               bbox_iou(image_bbox_to_pdf(a_px, 2.5, 2.5), image_bbox_to_pdf(b_px, 2.5, 2.5)))

    def test_invalid_boxes_raise(self):
        for bad in ([10, 0, 0, 10], [0, 0, 0, 10], [0, 0, 10], [0, 0, 10, float("nan")], None, "0,0,1,1"):
            with self.subTest(bbox=bad):
                with self.assertRaises(ValueError):
                    bbox_iou([0, 0, 10, 10], bad)
                with self.assertRaises(ValueError):
                    bbox_iou(bad, [0, 0, 10, 10])


class MatchingAndAccuracyTests(unittest.TestCase):
    @staticmethod
    def region(page, region_type, *bbox):
        return {"page_number": page, "type": region_type, "pdf_bbox": list(bbox)}

    def test_rows_and_class_accuracy(self):
        truth = [self.region(1, "TITLE", 0, 0, 100, 20),
                 self.region(1, "TEXT", 0, 30, 100, 90),
                 self.region(1, "TABLE", 0, 100, 100, 160),
                 self.region(2, "FIGURE", 0, 0, 100, 100)]
        predicted = [self.region(1, "TITLE", 0, 0, 100, 20),            # exact
                     self.region(1, "TEXT", 0, 32, 100, 92),            # slightly shifted
                     self.region(1, "FIGURE", 0, 100, 100, 160),        # right place, wrong class
                     self.region(1, "FIGURE", 0, 0, 100, 100)]          # page 1, not page 2

        rows = match_regions(truth, predicted)

        self.assertEqual([r["truth_type"] for r in rows], ["TITLE", "TEXT", "TABLE", "FIGURE"])
        self.assertEqual([r["predicted_type"] for r in rows], ["TITLE", "TEXT", "FIGURE", None])
        self.assertEqual([r["correct_class"] for r in rows], [True, True, False, False])
        self.assertEqual(rows[0]["iou"], 1.0)
        self.assertAlmostEqual(rows[1]["iou"], 58 / 62)
        self.assertEqual(rows[3]["iou"], 0.0)
        self.assertIsNone(rows[3]["predicted_bbox"])
        self.assertEqual(class_accuracy(rows), 0.5)

    def test_a_prediction_is_used_once_and_goes_to_the_best_match(self):
        truth = [self.region(1, "TEXT", 0, 0, 100, 100), self.region(1, "TEXT", 0, 20, 100, 120)]
        predicted = [self.region(1, "TEXT", 0, 20, 100, 120)]

        rows = match_regions(truth, predicted)

        self.assertEqual([r["predicted_type"] for r in rows], [None, "TEXT"])
        self.assertEqual(class_accuracy(rows), 0.5)

    def test_iou_threshold_and_bbox_key(self):
        truth = [{"page_number": 1, "type": "TEXT", "image_bbox": [0, 0, 100, 100]}]
        predicted = [{"page_number": 1, "type": "TEXT", "image_bbox": [0, 0, 100, 60]}]    # IoU 0.6

        self.assertEqual(class_accuracy(match_regions(truth, predicted, bbox_key="image_bbox")), 1.0)
        self.assertEqual(class_accuracy(match_regions(truth, predicted, iou_threshold=0.7, bbox_key="image_bbox")), 0.0)

    def test_no_ground_truth(self):
        self.assertEqual(match_regions([], [self.region(1, "TEXT", 0, 0, 1, 1)]), [])
        self.assertIsNone(class_accuracy([]))


class InspectLayoutToolErrorTests(TempDirTestCase):
    """Failures that are reported before the model is needed."""
    TOOL = os.path.join(ENGINE, "tools", "inspect_layout.py")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=300)

    def test_bad_input_exits_with_code_2_and_no_traceback(self):
        with open(self.path("bad.pdf"), "wb") as fh:
            fh.write(b"nope")
        os.makedirs(self.path("empty-dir"))
        cases = ([self.path("missing.pdf")], [self.path("bad.pdf"), "--output", self.path("out")],
                 [self.path("empty-dir")], [TWO_COLUMN, "--output", self.path("out"), "--layout-threshold", "1.5"])
        for args in cases:
            with self.subTest(args=args):
                done = self.run_tool(*args)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn("Error:", done.stderr)
                self.assertNotIn("Traceback", done.stderr)


def make_image_only_copy(pdf, target):
    """A 'scan' of `pdf`: every page becomes one grayscale bitmap, without a text layer."""
    with pymupdf.open(pdf) as source, pymupdf.open() as out:
        for page in source:
            pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
            out.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, pixmap=pix)
        out.save(target)


FIGURE_AXES = (140.0, 180.0, 470.0, 400.0)      # PDF points


def make_figure_page(target):
    """One page: heading, paragraph, a bar chart drawn with vector graphics, caption, paragraph."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 80), "2. Experimental Results", fontsize=14, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(72, 95, 540, 150),
                            "The chart below reports the validation accuracy of four networks. Accuracy rises with "
                            "depth and then levels off, which matches the trend reported in earlier sections.",
                            fontsize=11)
        x0, y0, x1, y1 = FIGURE_AXES
        page.draw_line((x0, y0), (x0, y1))
        page.draw_line((x0, y1), (x1, y1))
        for i, height in enumerate((90, 140, 170, 185)):
            x = x0 + 30 + i * 75
            page.draw_rect(pymupdf.Rect(x, y1 - height, x + 45, y1), color=(0, 0, 0), fill=(0.2 + 0.15 * i, 0.4, 0.8))
            page.insert_text((x + 5, y1 + 14), f"Net-{'ABCD'[i]}", fontsize=9)
        for k in range(5):
            y = y1 - k * 45
            page.draw_line((x0 - 4, y), (x0, y))
            page.insert_text((x0 - 28, y + 3), str(60 + 10 * k), fontsize=8)
        page.insert_text((150, 440), "Figure 1. Validation accuracy (%) of four networks of increasing depth.",
                         fontsize=10)
        page.insert_textbox(pymupdf.Rect(72, 470, 540, 540),
                            "After the figure the discussion continues with an ordinary paragraph of body text, so "
                            "that the page contains text both above and below the graphic element.", fontsize=11)
        doc.save(target)


@unittest.skipUnless(HAVE_MODEL, "paddleocr is not installed or AIRPA_SKIP_LAYOUT_MODEL=1")
class LayoutModelIntegrationTests(unittest.TestCase):
    """The real PP-DocLayoutV3 model on the fixtures. One detector, loaded once, for all documents."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="airpa-p3-model-")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        make_image_only_copy(DIGITAL, os.path.join(cls.tmp, "scan_of_digital.pdf"))
        make_figure_page(os.path.join(cls.tmp, "figure.pdf"))

        cls.detector = PPDocLayoutV3Detector(device="cpu")
        cls.pages, cls.results = {}, {}
        sources = {"digital": DIGITAL, "two_column": TWO_COLUMN, "scanned": SCANNED,
                   "scan_of_digital": os.path.join(cls.tmp, "scan_of_digital.pdf"),
                   "figure": os.path.join(cls.tmp, "figure.pdf")}
        for name, pdf in sources.items():
            work = os.path.join(cls.tmp, name)
            cls.pages[name] = process_pdf(pdf, work)["document"]
            cls.results[name] = detect_document_layout(work, cls.detector, visualize=(name == "two_column"))

    def regions(self, name, page_number=None):
        return [r for page in self.results[name]["document"]["pages"]
                if page_number in (None, page["page_number"]) for r in page["regions"]]

    def types(self, name, page_number=None):
        return [r["type"] for r in self.regions(name, page_number)]

    def test_model_loads_once_for_all_documents_and_pages(self):
        self.assertEqual(self.detector.load_count, 1)
        self.assertGreater(self.results["digital"]["timings"]["model_init_s"], 0)
        for name in ("two_column", "scanned", "scan_of_digital", "figure"):
            self.assertEqual(self.results[name]["timings"]["model_init_s"], 0.0)
        self.assertEqual(len(self.results["digital"]["timings"]["page_inference_s"]), 2)

    def test_every_document_gives_a_valid_layout_json(self):
        for name, result in self.results.items():
            with self.subTest(document=name):
                with open(result["layout_path"], encoding="utf-8") as fh:
                    data = json.load(fh)
                self.assertEqual(validate_layout_document(data), [])
                self.assertEqual(len(data["pages"]), self.pages[name]["document"]["page_count"])
                self.assertEqual(data["model"]["name"], "PP-DocLayoutV3")
                self.assertEqual(data["model"]["device"], "cpu")
                self.assertEqual(data["model"]["threshold"], DEFAULT_LAYOUT_THRESHOLD)
                self.assertRegex(data["model"]["version"], r"^\d+\.\d+")

    def test_boxes_scores_and_labels_of_every_region(self):
        for name, result in self.results.items():
            for page in result["document"]["pages"]:
                with self.subTest(document=name, page=page["page_number"]):
                    assert_boxes_inside_page(self, page)
                    for region in page["regions"]:
                        self.assertEqual(set(region), REGION_KEYS)
                        self.assertTrue(DEFAULT_LAYOUT_THRESHOLD <= region["score"] <= 1.0)
                        self.assertTrue(math.isfinite(region["score"]))
                        self.assertEqual(region["source_model"], "PP-DocLayoutV3")
                        self.assertIn(region["source_label"], PP_DOCLAYOUT_V3_LABEL_MAP)
                        self.assertEqual(region["type"], PP_DOCLAYOUT_V3_LABEL_MAP[region["source_label"]].value)

    def test_digital_multi_page_pdf(self):
        for page_number in (1, 2):
            types = self.types("digital", page_number)
            self.assertIn("TITLE", types)
            self.assertIn("TEXT", types)
        self.assertGreaterEqual(self.types("digital").count("TEXT"), 4)
        self.assertEqual(self.results["digital"]["issues"], [])     # no duplicates or page-sized boxes

    def test_table_like_region(self):
        tables = [r for r in self.regions("digital", 2) if r["type"] == "TABLE"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["source_label"], "table")
        self.assertNotIn("TABLE", self.types("digital", 1))

        # The PDF-space box matches the native text block of the table (Phase 2 coordinates).
        block = next(b for b in self.pages["digital"]["pages"][1]["blocks"] if b["text"].startswith("Model"))
        self.assertGreater(bbox_iou(tables[0]["pdf_bbox"], block["bbox"]), 0.8)

    def test_two_column_page(self):
        page = self.pages["two_column"]["pages"][0]
        middle = page["pdf_width"] / 2
        text = [r for r in self.regions("two_column") if r["type"] == "TEXT"]
        left = [r for r in text if r["pdf_bbox"][2] < middle]
        right = [r for r in text if r["pdf_bbox"][0] > middle]

        self.assertEqual(len(text), 2)
        self.assertEqual((len(left), len(right)), (1, 1))       # the columns are not merged
        self.assertIn("TITLE", self.types("two_column"))

        # Image -> PDF mapping, checked against the native blocks of Phase 2.
        blocks = page["blocks"]
        left_block = next(b for b in blocks if b["text"].startswith("LEFT-"))
        right_block = next(b for b in blocks if b["text"].startswith("RIGHT-"))
        self.assertGreater(bbox_iou(left[0]["pdf_bbox"], left_block["bbox"]), 0.8)
        self.assertGreater(bbox_iou(right[0]["pdf_bbox"], right_block["bbox"]), 0.8)

    def test_figure_like_region(self):
        figures = [r for r in self.regions("figure") if r["type"] == "FIGURE"]
        self.assertEqual(len(figures), 1)
        self.assertIn(figures[0]["source_label"], ("chart", "image"))
        self.assertGreater(bbox_iou(figures[0]["pdf_bbox"], FIGURE_AXES), 0.5)
        self.assertIn("CAPTION", self.types("figure"))
        self.assertIn("TEXT", self.types("figure"))

    def test_scanned_fixture_is_processed_without_a_text_layer(self):
        page = self.pages["scanned"]["pages"][0]
        self.assertFalse(page["has_text_layer"])
        # The fixture is a bare checkerboard bitmap: any number of regions is acceptable, the page
        # must only be processed and give a valid, in-bounds result (checked in the tests above).
        self.assertEqual(len(self.results["scanned"]["document"]["pages"]), 1)
        self.assertIsInstance(self.regions("scanned"), list)

    def test_image_only_pages_get_regions_without_ocr(self):
        for page in self.pages["scan_of_digital"]["pages"]:
            self.assertFalse(page["has_text_layer"])
            self.assertEqual(page["blocks"], [])

        self.assertIn("TITLE", self.types("scan_of_digital", 1))
        self.assertIn("TEXT", self.types("scan_of_digital", 1))
        self.assertIn("TABLE", self.types("scan_of_digital", 2))
        # Same structure as the digital original, found from pixels alone.
        self.assertEqual(self.types("scan_of_digital"), self.types("digital"))

    def test_visualization_of_real_detections(self):
        result = self.results["two_column"]
        self.assertEqual([os.path.basename(p) for p in result["visualizations"]], ["page_001_layout.png"])
        original = pymupdf.Pixmap(os.path.join(result["work_dir"], "pages", "page_001.png"))
        annotated = pymupdf.Pixmap(result["visualizations"][0])
        self.assertEqual((annotated.width, annotated.height), (original.width, original.height))
        self.assertNotEqual(annotated.samples, original.samples)
        self.assertEqual(self.results["digital"]["visualizations"], [])


@unittest.skipUnless(HAVE_MODEL, "paddleocr is not installed or AIRPA_SKIP_LAYOUT_MODEL=1")
class InspectLayoutToolIntegrationTests(TempDirTestCase):
    TOOL = os.path.join(ENGINE, "tools", "inspect_layout.py")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=600)

    def test_pdf_input_then_existing_working_directory(self):
        done = self.run_tool(DIGITAL, "--output", self.path("out"), "--visualize-layout")

        self.assertEqual(done.returncode, 0, done.stderr)
        for expected in ("Pages: 2", "Detected regions:", "TITLE: ", "TEXT: ", "TABLE: 1",
                         "Model: PP-DocLayoutV3", "Device: CPU", "Threshold: 0.5", "model init", "s/page"):
            self.assertIn(expected, done.stdout)
        self.assertEqual(sorted(os.listdir(self.path("out", "metadata"))), ["layout.json", "pages.json"])
        self.assertEqual(sorted(os.listdir(self.path("out", "layout_debug"))),
                         ["page_001_layout.png", "page_002_layout.png"])
        with open(self.path("out", "metadata", "layout.json"), encoding="utf-8") as fh:
            first = json.load(fh)
        self.assertEqual(validate_layout_document(first), [])

        # Second run on the Phase 2 directory with a stricter threshold: fewer or equal regions.
        pages_json = sha256(self.path("out", "metadata", "pages.json"))
        done = self.run_tool(self.path("out"), "--layout-threshold", "0.9")

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Threshold: 0.9", done.stdout)
        self.assertEqual(sha256(self.path("out", "metadata", "pages.json")), pages_json)
        with open(self.path("out", "metadata", "layout.json"), encoding="utf-8") as fh:
            second = json.load(fh)
        count = lambda doc: sum(len(p["regions"]) for p in doc["pages"])   # noqa: E731
        self.assertLess(count(second), count(first))
        self.assertTrue(all(r["score"] >= 0.9 for p in second["pages"] for r in p["regions"]))


if __name__ == "__main__":
    unittest.main()

"""Phase 2 tests: page rendering, native extraction, pages.json. Standard library unittest.

    python -m unittest discover -s tests/python -v
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)

from pipeline import (  # noqa: E402
    EmptyPdfError,
    InvalidPdfError,
    OutputDirectoryError,
    PageRenderError,
    PdfNotFoundError,
    PdfPipelineError,
    default_work_dir,
    extract_native_pages,
    page_has_text_layer,
    pdf_bbox_to_pixels,
    process_pdf,
    render_pdf_pages,
    validate_pages_document,
)
from pipeline import render as render_module  # noqa: E402
from pipeline.common import pymupdf  # noqa: E402

DIGITAL = os.path.join(FIXTURES, "digital", "synthetic_digital.pdf")          # 2 pages
TWO_COLUMN = os.path.join(FIXTURES, "multicolumn", "synthetic_two_column.pdf")
SCANNED = os.path.join(FIXTURES, "scanned", "synthetic_scanned.pdf")

# A structurally valid PDF whose page tree is empty.
ZERO_PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n"
)


def png_size(path):
    """(width, height) read from the PNG header, independent of PyMuPDF."""
    with open(path, "rb") as fh:
        header = fh.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    return struct.unpack(">II", header[16:24])


def dark_pixels(path):
    """Set of (x, y) of clearly dark pixels (text ink) in a page image."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.Pixmap(path))
    width, samples = pix.width, pix.samples
    return {(i % width, i // width) for i, value in enumerate(samples) if value < 128}


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="airpa-p2-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def write(self, name, data):
        with open(self.path(name), "wb") as fh:
            fh.write(data)
        return self.path(name)


class RenderTests(TempDirTestCase):
    def test_digital_pdf_renders_every_page_with_stable_names(self):
        pages = render_pdf_pages(DIGITAL, self.path("out"))

        self.assertEqual([p["page_number"] for p in pages], [1, 2])
        self.assertEqual(sorted(os.listdir(self.path("out"))), ["page_001.png", "page_002.png"])
        for page in pages:
            self.assertEqual(page["dpi"], 180)
            self.assertTrue(os.path.isabs(page["image_path"]))
            self.assertTrue(os.path.isfile(page["image_path"]))

    def test_rendered_image_count_equals_pdf_page_count(self):
        for pdf in (DIGITAL, TWO_COLUMN, SCANNED):
            with self.subTest(pdf=os.path.basename(pdf)):
                out = self.path(os.path.basename(pdf))
                pages = render_pdf_pages(pdf, out)
                with pymupdf.open(pdf) as doc:
                    expected = doc.page_count
                self.assertEqual(len(pages), expected)
                self.assertEqual(len([f for f in os.listdir(out) if f.endswith(".png")]), expected)

    def test_image_dimensions_match_metadata(self):
        for dpi in (72, 100, 180):
            with self.subTest(dpi=dpi):
                for page in render_pdf_pages(DIGITAL, self.path(f"d{dpi}"), dpi=dpi):
                    self.assertEqual(png_size(page["image_path"]), (page["width"], page["height"]))
                    self.assertEqual((page["image_width"], page["image_height"]), (page["width"], page["height"]))
                    # US Letter, 612 x 792 pt
                    self.assertAlmostEqual(page["width"], 612 * dpi / 72, delta=1)
                    self.assertAlmostEqual(page["height"], 792 * dpi / 72, delta=1)

    def test_scale_is_image_size_over_page_size(self):
        for dpi in (100, 180):
            with self.subTest(dpi=dpi):
                for page in render_pdf_pages(DIGITAL, self.path(f"d{dpi}"), dpi=dpi):
                    self.assertEqual((page["pdf_width"], page["pdf_height"]), (612.0, 792.0))
                    self.assertAlmostEqual(page["scale_x"], page["image_width"] / page["pdf_width"])
                    self.assertAlmostEqual(page["scale_y"], page["image_height"] / page["pdf_height"])
                    self.assertAlmostEqual(page["scale_x"], dpi / 72, delta=0.01)
                    self.assertAlmostEqual(page["scale_y"], dpi / 72, delta=0.01)
                    # The page corner maps to the image corner.
                    corner = pdf_bbox_to_pixels([0, 0, page["pdf_width"], page["pdf_height"]],
                                                page["scale_x"], page["scale_y"])
                    self.assertEqual([round(v, 6) for v in corner], [0, 0, page["image_width"], page["image_height"]])

    def test_rerun_removes_page_images_of_a_longer_earlier_run(self):
        out = self.path("out")
        render_pdf_pages(DIGITAL, out)
        render_pdf_pages(TWO_COLUMN, out)

        self.assertEqual(os.listdir(out), ["page_001.png"])

    def test_missing_pdf(self):
        with self.assertRaises(PdfNotFoundError):
            render_pdf_pages(self.path("missing.pdf"), self.path("out"))
        self.assertFalse(os.path.exists(self.path("out")))

    def test_zero_byte_file(self):
        with self.assertRaises(InvalidPdfError):
            render_pdf_pages(self.write("empty.pdf", b""), self.path("out"))

    def test_corrupt_pdf(self):
        for name, data in (("text.pdf", b"this is not a pdf at all"),
                           ("header_only.pdf", b"%PDF-1.4\n" + os.urandom(2048))):
            with self.subTest(name=name):
                with self.assertRaises((InvalidPdfError, EmptyPdfError)):
                    render_pdf_pages(self.write(name, data), self.path("out"))

    def test_truncated_pdf_fails_cleanly_or_renders_every_page(self):
        with open(DIGITAL, "rb") as fh:
            data = fh.read()
        broken = self.write("truncated.pdf", data[: len(data) // 3])
        try:
            pages = render_pdf_pages(broken, self.path("out"))
        except PdfPipelineError:
            return
        self.assertEqual(len(pages), len(os.listdir(self.path("out"))))

    def test_zero_page_pdf(self):
        with self.assertRaises((EmptyPdfError, InvalidPdfError)):
            render_pdf_pages(self.write("zero.pdf", ZERO_PAGE_PDF), self.path("out"))

    def test_output_directory_that_is_a_file(self):
        blocker = self.write("blocker", b"x")
        with self.assertRaises(OutputDirectoryError):
            render_pdf_pages(DIGITAL, blocker)

    def test_invalid_dpi(self):
        for dpi in (0, -5, 5000, "180", True):
            with self.subTest(dpi=dpi):
                with self.assertRaises(ValueError):
                    render_pdf_pages(DIGITAL, self.path("out"), dpi=dpi)

    def test_a_failing_page_fails_the_whole_run(self):
        original = pymupdf.Page.get_pixmap
        calls = []

        def failing(page, *args, **kwargs):
            calls.append(page.number)
            if page.number == 1:
                raise RuntimeError("simulated render failure")
            return original(page, *args, **kwargs)

        pymupdf.Page.get_pixmap = failing
        try:
            with self.assertRaises(PageRenderError) as caught:
                render_module.render_pdf_pages(DIGITAL, self.path("out"))
        finally:
            pymupdf.Page.get_pixmap = original

        self.assertEqual(caught.exception.page_number, 2)
        self.assertEqual(calls, [0, 1])


class NativeExtractionTests(TempDirTestCase):
    def test_digital_pdf_pages_text_and_blocks(self):
        pages = extract_native_pages(DIGITAL)

        self.assertEqual([p["page_number"] for p in pages], [1, 2])
        for page in pages:
            self.assertEqual((page["width"], page["height"]), (612.0, 792.0))
            self.assertTrue(page["has_text_layer"])
            self.assertGreater(len(page["blocks"]), 0)
            self.assertEqual([b["block_index"] for b in page["blocks"]], list(range(len(page["blocks"]))))
            for block in page["blocks"]:
                self.assertEqual(block["block_type"], "text")
                self.assertTrue(block["text"].strip())
        self.assertIn("Effect of Network Depth on Accuracy", pages[0]["text"])
        self.assertIn("1. Introduction", pages[0]["text"])
        self.assertIn("Table 1.", pages[1]["text"])
        self.assertNotIn("Table 1.", pages[0]["text"])      # text stays on its own page

    def test_block_boxes_are_valid_and_inside_the_page(self):
        for pdf in (DIGITAL, TWO_COLUMN):
            for page in extract_native_pages(pdf):
                for block in page["blocks"]:
                    with self.subTest(pdf=os.path.basename(pdf), page=page["page_number"], block=block["block_index"]):
                        x0, y0, x1, y1 = block["bbox"]
                        self.assertTrue(all(isinstance(v, float) for v in block["bbox"]))
                        self.assertLess(x0, x1)
                        self.assertLess(y0, y1)
                        self.assertGreaterEqual(x0, 0)
                        self.assertGreaterEqual(y0, 0)
                        self.assertLessEqual(x1, page["width"])
                        self.assertLessEqual(y1, page["height"])

    def test_y_axis_points_down(self):
        blocks = extract_native_pages(DIGITAL)[0]["blocks"]
        title = next(b for b in blocks if "Effect of Network Depth" in b["text"])
        intro = next(b for b in blocks if "1. Introduction" in b["text"])

        self.assertLess(title["bbox"][1], intro["bbox"][1])     # the title is above, so its y is smaller
        self.assertLess(title["bbox"][1], 792 / 4)

    def test_two_column_pdf_has_separate_left_and_right_blocks(self):
        page = extract_native_pages(TWO_COLUMN)[0]
        left = [b for b in page["blocks"] if b["text"].startswith("LEFT-")]
        right = [b for b in page["blocks"] if b["text"].startswith("RIGHT-")]

        self.assertTrue(page["has_text_layer"])
        self.assertTrue(left and right)
        middle = page["width"] / 2
        for block in left:
            self.assertLess(block["bbox"][2], middle)
            self.assertNotIn("RIGHT-", block["text"])
        for block in right:
            self.assertGreater(block["bbox"][0], middle)
            self.assertNotIn("LEFT-", block["text"])

    def test_scanned_pdf_has_no_text_layer(self):
        page = extract_native_pages(SCANNED)[0]

        self.assertFalse(page["has_text_layer"])
        self.assertEqual(page["blocks"], [])
        self.assertEqual(page["text"].strip(), "")
        self.assertEqual((page["width"], page["height"]), (612.0, 792.0))

    def test_text_layer_heuristic(self):
        self.assertFalse(page_has_text_layer(""))
        self.assertFalse(page_has_text_layer("   \n\t  "))
        self.assertFalse(page_has_text_layer("12"))
        self.assertFalse(page_has_text_layer("Page 3 of 10"))
        self.assertFalse(page_has_text_layer("- 7 -\n\n"))
        self.assertFalse(page_has_text_layer("DRAFT"))
        self.assertFalse(page_has_text_layer("1234567890 1234567890 1234567890"))     # digits only
        self.assertFalse(page_has_text_layer("�" * 200))                          # undecodable glyphs
        self.assertFalse(page_has_text_layer(". , ; : ! ?" * 20))
        self.assertTrue(page_has_text_layer("Neural networks are widely used for image classification."))
        self.assertTrue(page_has_text_layer("Нейронные сети широко используются для классификации."))

    def test_page_with_only_a_page_number_has_no_text_layer(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((300, 800), "7")
            doc.new_page().insert_text((72, 100), "This page carries a normal sentence of body text.")
            doc.save(self.path("numbers.pdf"))

        pages = extract_native_pages(self.path("numbers.pdf"))

        self.assertEqual([p["has_text_layer"] for p in pages], [False, True])
        self.assertEqual(len(pages[0]["blocks"]), 1)     # the block is still reported

    def test_missing_corrupt_and_empty_files(self):
        with self.assertRaises(PdfNotFoundError):
            extract_native_pages(self.path("missing.pdf"))
        with self.assertRaises(InvalidPdfError):
            extract_native_pages(self.write("empty.pdf", b""))
        with self.assertRaises((InvalidPdfError, EmptyPdfError)):
            extract_native_pages(self.write("bad.pdf", b"garbage, not a pdf"))


class CoordinateMappingTests(TempDirTestCase):
    """Block boxes (PDF points) scaled by scale_x/scale_y must land on the text in the image."""

    def assert_boxes_cover_the_ink(self, pdf, dpi):
        result = process_pdf(pdf, self.path(f"work-{dpi}"), dpi=dpi)
        for page in result["document"]["pages"]:
            ink = dark_pixels(os.path.join(result["work_dir"], page["image_path"]))
            self.assertTrue(ink, "page image has no dark pixels")
            boxes = [pdf_bbox_to_pixels(b["bbox"], page["scale_x"], page["scale_y"]) for b in page["blocks"]]
            self.assertTrue(boxes)

            for block, (x0, y0, x1, y1) in zip(page["blocks"], boxes):
                self.assertTrue(0 <= x0 < x1 <= page["image_width"] and 0 <= y0 < y1 <= page["image_height"])
                inside = sum(1 for x, y in ink if x0 <= x <= x1 and y0 <= y <= y1)
                self.assertGreater(inside, 0, f"no ink inside block {block['block_index']} on page {page['page_number']}")

            tolerance = 2   # pixels, for anti-aliasing at the box edges
            outside = [(x, y) for x, y in ink
                       if not any(x0 - tolerance <= x <= x1 + tolerance and y0 - tolerance <= y <= y1 + tolerance
                                  for x0, y0, x1, y1 in boxes)]
            self.assertEqual(outside, [], f"{len(outside)} ink pixels outside every block on page {page['page_number']}")

    def test_two_column_fixture(self):
        for dpi in (72, 100):       # 100 dpi gives a non-integer scale
            with self.subTest(dpi=dpi):
                self.assert_boxes_cover_the_ink(TWO_COLUMN, dpi)

    def test_digital_fixture(self):
        self.assert_boxes_cover_the_ink(DIGITAL, 100)

    def test_rotated_and_odd_sized_page(self):
        with pymupdf.open() as doc:
            page = doc.new_page(width=500.5, height=333.3)
            page.insert_text((60, 80), "Rotated page: first line of text for mapping.", fontsize=12)
            page.insert_text((60, 250), "Second block placed far below the first one.", fontsize=12)
            page.set_rotation(90)
            doc.save(self.path("rotated.pdf"))

        self.assert_boxes_cover_the_ink(self.path("rotated.pdf"), 100)

        page = process_pdf(self.path("rotated.pdf"), self.path("rot"), dpi=100)["document"]["pages"][0]
        # Width and height are those of the page as displayed (rotated).
        self.assertAlmostEqual(page["pdf_width"], 333.3, places=1)
        self.assertAlmostEqual(page["pdf_height"], 500.5, places=1)
        self.assertGreater(page["image_height"], page["image_width"])


class PagePipelineTests(TempDirTestCase):
    def test_writes_the_working_directory_layout(self):
        result = process_pdf(DIGITAL, self.path("doc"), copy_original=True)

        self.assertEqual(sorted(os.listdir(self.path("doc"))), ["metadata", "original.pdf", "pages"])
        self.assertEqual(sorted(os.listdir(self.path("doc", "pages"))), ["page_001.png", "page_002.png"])
        self.assertEqual(os.listdir(self.path("doc", "metadata")), ["pages.json"])
        self.assertEqual(result["metadata_path"], self.path("doc", "metadata", "pages.json"))
        self.assertEqual(set(result["timings"]), {"open_s", "render_s", "extract_s", "total_s"})

    def test_pages_json_is_valid_utf8_json_with_the_expected_structure(self):
        result = process_pdf(DIGITAL, self.path("doc"))

        with open(result["metadata_path"], "rb") as fh:
            raw = fh.read()
        data = json.loads(raw.decode("utf-8"))      # raises if not UTF-8 / not JSON

        self.assertEqual(validate_pages_document(data), [])
        self.assertEqual(data, result["document"])
        self.assertEqual(data["document"]["source_pdf"], "synthetic_digital.pdf")
        self.assertEqual(data["document"]["page_count"], 2)
        self.assertEqual(len(data["pages"]), 2)
        self.assertEqual(data["document"]["pages_with_text_layer"], 2)
        self.assertEqual(data["document"]["coordinate_system"]["origin"], "top-left")
        self.assertEqual(data["document"]["coordinate_system"]["y_axis"], "down")
        required = {"page_number", "pdf_width", "pdf_height", "image_width", "image_height",
                    "scale_x", "scale_y", "has_text_layer", "text", "blocks", "image_path"}
        for page in data["pages"]:
            self.assertLessEqual(required, set(page))
            self.assertEqual(page["image_path"], f"pages/page_{page['page_number']:03d}.png")
            image = os.path.join(result["work_dir"], page["image_path"])
            self.assertEqual(png_size(image), (page["image_width"], page["image_height"]))

    def test_pages_json_is_identical_across_runs(self):
        first = process_pdf(DIGITAL, self.path("a"))["metadata_path"]
        second = process_pdf(DIGITAL, self.path("b"))["metadata_path"]

        with open(first, "rb") as a, open(second, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_non_ascii_text_survives_the_round_trip(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 100), "Señor Müller measured 5 µm at 20 °C — naïve café résumé.")
            doc.save(self.path("unicode.pdf"))

        result = process_pdf(self.path("unicode.pdf"), self.path("doc"))
        with open(result["metadata_path"], encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertIn("Müller", data["pages"][0]["text"])
        self.assertIn("µm", data["pages"][0]["blocks"][0]["text"])

    def test_scanned_fixture_is_processed_without_text(self):
        data = process_pdf(SCANNED, self.path("doc"))["document"]

        self.assertEqual(data["document"]["page_count"], 1)
        self.assertEqual(data["document"]["pages_without_text_layer"], 1)
        self.assertFalse(data["pages"][0]["has_text_layer"])
        self.assertEqual(data["pages"][0]["blocks"], [])
        self.assertTrue(os.path.isfile(self.path("doc", "pages", "page_001.png")))

    def test_failed_run_leaves_no_pages_json(self):
        process_pdf(DIGITAL, self.path("doc"))
        bad = self.write("bad.pdf", b"not a pdf")

        with self.assertRaises(PdfPipelineError):
            process_pdf(bad, self.path("doc"))

        self.assertFalse(os.path.exists(self.path("doc", "metadata", "pages.json")))

    def test_validator_reports_problems(self):
        data = process_pdf(TWO_COLUMN, self.path("doc"))["document"]

        self.assertEqual(validate_pages_document(data), [])
        self.assertTrue(validate_pages_document([]))
        self.assertTrue(validate_pages_document({"document": {}, "pages": []}))

        broken = json.loads(json.dumps(data))
        broken["pages"][0]["blocks"][0]["bbox"] = [10, 10, 5000, 20]
        self.assertTrue(any("outside the page" in p for p in validate_pages_document(broken)))

        broken = json.loads(json.dumps(data))
        del broken["pages"][0]["scale_x"]
        broken["document"]["page_count"] = 9
        self.assertEqual(len(validate_pages_document(broken)), 2)

    def test_default_work_dir_is_safe_deterministic_and_collision_free(self):
        base = self.path("output")
        with open(DIGITAL, "rb") as fh:
            nasty = self.write("..my paper (v2)!.pdf", fh.read())
        same_name_other_content = os.path.join(self.tmp, "sub", "..my paper (v2)!.pdf")
        os.makedirs(os.path.dirname(same_name_other_content))
        shutil.copyfile(TWO_COLUMN, same_name_other_content)

        first = default_work_dir(nasty, base)

        self.assertEqual(first, default_work_dir(nasty, base))
        self.assertNotEqual(first, default_work_dir(same_name_other_content, base))
        self.assertEqual(os.path.dirname(first), base)
        self.assertRegex(os.path.basename(first), r"^[A-Za-z0-9_-]+-[0-9a-f]{8}$")
        with self.assertRaises(PdfNotFoundError):
            default_work_dir(self.path("missing.pdf"), base)


class InspectToolTests(TempDirTestCase):
    TOOL = os.path.join(ENGINE, "tools", "inspect_pdf.py")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=120)

    def test_prints_a_summary_and_writes_pages_json(self):
        done = self.run_tool(DIGITAL, "--output", self.path("out"))

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Pages: 2", done.stdout)
        self.assertIn("Rendered: 2", done.stdout)
        self.assertIn("Pages with native text: 2", done.stdout)
        self.assertIn("Pages without native text: 0", done.stdout)
        self.assertTrue(os.path.isfile(self.path("out", "metadata", "pages.json")))

    def test_bad_input_exits_with_code_2_and_no_traceback(self):
        for pdf in (self.path("missing.pdf"), self.write("empty.pdf", b""), self.write("bad.pdf", b"nope")):
            with self.subTest(pdf=os.path.basename(pdf)):
                done = self.run_tool(pdf, "--output", self.path("out"))
                self.assertEqual(done.returncode, 2)
                self.assertTrue(done.stderr.startswith("Error:"), done.stderr)
                self.assertNotIn("Traceback", done.stderr)


class V1RegressionTests(unittest.TestCase):
    def test_v1_cli_output_shape_is_unchanged(self):
        done = subprocess.run([sys.executable, os.path.join(ENGINE, "ai_engine.py"), DIGITAL],
                              capture_output=True, text=True, timeout=180)

        self.assertEqual(done.returncode, 0, done.stderr)
        data = json.loads(done.stdout)
        self.assertEqual(list(data), ["status", "file_name", "summary", "keywords", "important_points", "flow", "tables"])
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["keywords"]), 10)
        self.assertEqual(len(data["important_points"]), 5)


if __name__ == "__main__":
    unittest.main()

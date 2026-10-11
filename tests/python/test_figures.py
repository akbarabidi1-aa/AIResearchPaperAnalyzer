"""Phase 7 tests: figure crops, captions, figure numbers, figure types, text in figures, chart reading,
figures.json, metrics.

    python -m unittest discover -s tests/python -v

The unit tests use a fake layout detector and a fake OCR engine and need neither PaddleOCR nor a model
download; the chart reader has no model (it needs OpenCV and numpy). The classes named
*IntegrationTests run the real PP-DocLayoutV3 and PP-OCRv5 models; they are skipped when paddleocr is
not installed or when AIRPA_SKIP_LAYOUT_MODEL=1 or AIRPA_SKIP_OCR_MODEL=1.
"""
import json
import math
import os
import subprocess
import sys
import unittest

from PIL import Image, ImageDraw

from test_layout import sha256
from test_ocr import HAVE_OCR_MODEL, FakeEngine, OcrTestCase

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)

from pipeline import (  # noqa: E402
    ChartExtractionError,
    ChartExtractor,
    ChartText,
    FigureExtractionError,
    FigureExtractor,
    FigureInputError,
    OcrLine,
    PaddleOCREngine,
    PPDocLayoutV3Detector,
    RuleBasedChartExtractor,
    associate_figure_captions,
    calibrate_axis,
    caption_candidates,
    classify_figure_image,
    detect_document_layout,
    evaluate_chart,
    evaluate_figures,
    figure_crop_boxes,
    find_plot_area,
    parse_figure_number,
    parse_number,
    process_pdf,
    run_document_figures,
    run_document_ocr,
    validate_figures_document,
)
from pipeline.charts import empty_chart_data, merge_corners, to_value  # noqa: E402
from pipeline.figures import colour_flatness, usable_ocr_items  # noqa: E402

FIGURES_PDF = os.path.join(FIXTURES, "figures", "synthetic_figures.pdf")                # 5 pages, native text
FIGURES_SCANNED = os.path.join(FIXTURES, "figures", "synthetic_figures_scanned.pdf")    # the same as bitmaps
TRUTH_PATH = os.path.join(FIXTURES, "figures", "synthetic_figures.json")
with open(TRUTH_PATH, encoding="utf-8") as _fh:
    TRUTH = json.load(_fh)["figures"]
BY_KIND = {figure["kind"]: figure for figure in TRUTH}

# The fixture's figures and caption lines as layout regions: {page: [(model label, PDF box)]}.
FIGURE_REGIONS = {}
for _figure in TRUTH:
    _regions = FIGURE_REGIONS.setdefault(_figure["page_number"], [])
    _regions.append(("chart" if _figure["figure_type"] == "chart" else "image", tuple(_figure["pdf_bbox"])))
    _regions += [("figure_title", tuple(box)) for box in _figure["caption_candidates"]]


def all_figures(document):
    return [figure for page in document["pages"] for figure in page["figures"]]


def region(region_id, region_type, *pdf_bbox, label=None):
    return {"region_id": region_id, "page_number": 1, "type": region_type, "pdf_bbox": list(pdf_bbox),
            "source_label": label or region_type.lower()}


def tick(text, centre_x, centre_y, **options):
    """A text item 20 x 12 pixels around a point."""
    return ChartText(text, (centre_x - 10, centre_y - 6, centre_x + 10, centre_y + 6), **options)


# A drawn chart: plot area (60, 20)-(380, 260); y from 0 at pixel 260 to 100 at pixel 20; x ticks every 60 px.
PLOT = (60, 20, 380, 260)


def y_pixel(value):
    return 260 - value * 2.4


def blank_chart(y_ticks=(0, 25, 50, 75, 100), x_ticks=()):
    """(image, draw, texts): the two axis lines and tick labels as text items (their glyphs are not drawn)."""
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(60, 20), (60, 260)], fill="black", width=2)
    draw.line([(60, 260), (380, 260)], fill="black", width=2)
    texts = [tick(str(value), 40, y_pixel(value)) for value in y_ticks]
    texts += [tick(text, x, 274) for text, x in x_ticks]
    return image, draw, texts


def bar_chart(values=(40, 80, 20), labels=("A", "B", "C"), colour=(40, 90, 200)):
    image, draw, texts = blank_chart(x_ticks=[(label, 110 + 90 * i) for i, label in enumerate(labels)])
    for index, value in enumerate(values):
        draw.rectangle([85 + 90 * index, y_pixel(value), 135 + 90 * index, 258], fill=colour)
    return image, texts


def line_chart(series=(((1, 20), (2, 70), (3, 40), (4, 90)),), colours=((200, 40, 40), (30, 140, 60))):
    image, draw, texts = blank_chart(x_ticks=[(str(n), 60 + 70 * n) for n in (1, 2, 3, 4)])
    for points, colour in zip(series, colours):
        draw.line([(60 + 70 * x, y_pixel(y)) for x, y in points], fill=colour, width=3)
    return image, texts


def scatter_chart(points=((1, 20), (1.5, 60), (2, 35), (3, 80), (3.5, 50), (4, 90))):
    image, draw, texts = blank_chart(x_ticks=[(str(n), 60 + 70 * n) for n in (1, 2, 3, 4)])
    for x, y in points:
        px, py = 60 + 70 * x, y_pixel(y)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(170, 50, 120))
    return image, texts


class RaisingChartExtractor(ChartExtractor):
    name = "raising-chart"

    def _extract(self, image, texts, origin):
        raise RuntimeError("simulated chart failure")


class FiguresTestCase(OcrTestCase):
    def prepare_figures(self, pdf=FIGURES_PDF, regions=None, name="doc", engine=None):
        """Phase 2, a layout.json with the fixture's figure and caption regions, and Phase 4."""
        work = self.prepare(pdf, FIGURE_REGIONS if regions is None else regions, name)
        run_document_ocr(work, engine or FakeEngine())
        return work

    def run_figures(self, pdf=FIGURES_PDF, regions=None, extractor=None, **options):
        self.work = self.prepare_figures(pdf, regions)
        self.extractor = extractor or FigureExtractor(FakeEngine())
        return run_document_figures(self.work, self.extractor, **options)

    def read(self, name):
        with open(os.path.join(self.work, "metadata", name), encoding="utf-8") as fh:
            return json.load(fh)


# 1. Figure crops ---------------------------------------------------------------------------------------

class CropTests(unittest.TestCase):
    def test_raw_box_is_the_region_and_padded_box_adds_the_padding(self):
        raw, padded = figure_crop_boxes((100.4, 200.6, 300.2, 400.1), 1000, 1000, padding=8)
        self.assertEqual(raw, (100, 200, 301, 401))
        self.assertEqual(padded, (92, 192, 309, 409))

    def test_padding_is_clipped_to_the_page(self):
        raw, padded = figure_crop_boxes((2, 3, 995, 990), 1000, 1000, padding=20)
        self.assertEqual(raw, (2, 3, 995, 990))
        self.assertEqual(padded, (0, 0, 1000, 1000))

    def test_zero_padding_gives_the_same_box_twice(self):
        raw, padded = figure_crop_boxes((10, 10, 50, 50), 100, 100, padding=0)
        self.assertEqual(raw, padded)

    def test_padding_stops_at_a_caption_below_and_above(self):
        below = [(100, 404, 300, 420)]
        above = [(100, 180, 300, 195)]
        _, padded = figure_crop_boxes((100, 200, 300, 400), 1000, 1000, 8, avoid=below + above)
        self.assertEqual(padded, (92, 195, 308, 404))

    def test_a_caption_in_another_column_does_not_limit_the_padding(self):
        _, padded = figure_crop_boxes((100, 200, 300, 400), 1000, 1000, 8, avoid=[(500, 402, 700, 420)])
        self.assertEqual(padded, (92, 192, 308, 408))

    def test_region_outside_the_page_or_too_small_is_refused(self):
        for bbox in ((1200, 10, 1300, 50), (10, 10, 12, 12), (5, 5, 5, 50), "box"):
            with self.subTest(bbox=bbox), self.assertRaises(FigureExtractionError):
                figure_crop_boxes(bbox, 1000, 1000)

    def test_bad_padding_is_refused(self):
        for padding in (-1, 2.5, True, None):
            with self.subTest(padding=padding), self.assertRaises(ValueError):
                figure_crop_boxes((10, 10, 50, 50), 100, 100, padding)
        with self.assertRaises(ValueError):
            FigureExtractor(padding=-3)


# 3. - 5. Captions --------------------------------------------------------------------------------------

class CaptionAssociationTests(unittest.TestCase):
    FIGURE = ("f", "FIGURE", 100, 200, 300, 400)

    def pairs(self, *regions, texts=None):
        return associate_figure_captions([region(*r) if isinstance(r, tuple) else r for r in regions], texts)

    def test_caption_below(self):
        pairs = self.pairs(self.FIGURE, ("c", "CAPTION", 100, 408, 300, 420))
        self.assertEqual(pairs["f"]["caption_region_id"], "c")
        self.assertEqual(pairs["f"]["position"], "below")
        self.assertAlmostEqual(pairs["f"]["gap"], 8.0)

    def test_caption_above(self):
        pairs = self.pairs(self.FIGURE, ("c", "CAPTION", 100, 180, 300, 192))
        self.assertEqual((pairs["f"]["caption_region_id"], pairs["f"]["position"]), ("c", "above"))

    def test_caption_in_the_other_column_is_rejected_although_it_is_nearest(self):
        # The caption of the right column is level with the bottom of the figure: nearer than any other.
        self.assertEqual(self.pairs(self.FIGURE, ("other", "CAPTION", 320, 390, 520, 402)), {})
        pairs = self.pairs(self.FIGURE, ("other", "CAPTION", 320, 390, 520, 402), ("own", "CAPTION", 100, 425, 300, 437))
        self.assertEqual(pairs["f"]["caption_region_id"], "own")

    def test_each_column_gets_its_own_caption(self):
        pairs = self.pairs(("left", "FIGURE", 72, 100, 290, 250), ("right", "FIGURE", 318, 100, 540, 250),
                           ("cl", "CAPTION", 72, 258, 290, 270), ("cr", "CAPTION", 318, 258, 540, 270))
        self.assertEqual({key: value["caption_region_id"] for key, value in pairs.items()},
                         {"left": "cl", "right": "cr"})

    def test_short_caption_flush_left_under_a_centred_figure(self):
        pairs = self.pairs(("f", "FIGURE", 170, 200, 440, 400), ("c", "CAPTION", 72, 408, 230, 420))
        self.assertEqual(pairs["f"]["caption_region_id"], "c")

    def test_caption_beside_the_figure_is_not_a_caption_of_it(self):
        self.assertEqual(self.pairs(self.FIGURE, ("c", "CAPTION", 150, 250, 290, 262)), {})

    def test_caption_too_far_away_is_not_assigned(self):
        self.assertEqual(self.pairs(self.FIGURE, ("c", "CAPTION", 100, 440, 300, 452)), {})

    def test_two_captions_at_about_the_same_distance_give_no_caption(self):
        self.assertEqual(self.pairs(self.FIGURE, ("above", "CAPTION", 100, 180, 300, 192),
                                    ("below", "CAPTION", 100, 409, 300, 421)), {})

    def test_clearly_nearer_caption_wins(self):
        pairs = self.pairs(self.FIGURE, ("above", "CAPTION", 100, 160, 300, 172),
                           ("below", "CAPTION", 100, 406, 300, 418))
        self.assertEqual(pairs["f"]["caption_region_id"], "below")

    def test_caption_between_two_figures_goes_to_the_nearer_one(self):
        pairs = self.pairs(("upper", "FIGURE", 100, 100, 300, 200), ("ca", "CAPTION", 100, 206, 300, 218),
                           ("lower", "FIGURE", 100, 236, 300, 336), ("cb", "CAPTION", 100, 342, 300, 354))
        self.assertEqual({key: value["caption_region_id"] for key, value in pairs.items()},
                         {"upper": "ca", "lower": "cb"})

    def test_caption_at_the_same_distance_from_two_figures_goes_to_neither(self):
        pairs = self.pairs(("upper", "FIGURE", 100, 100, 300, 200), ("c", "CAPTION", 100, 210, 300, 222),
                           ("lower", "FIGURE", 100, 232, 300, 332))
        self.assertEqual(pairs, {})

    def test_table_caption_is_left_to_the_table(self):
        regions = (self.FIGURE, ("c", "CAPTION", 100, 408, 300, 420))
        self.assertEqual(self.pairs(*regions, texts={"c": "Table 2: Results on the test set."}), {})
        self.assertIn("f", self.pairs(*regions, texts={"c": "Figure 2: Results on the test set."}))

    def test_figure_caption_is_not_offered_to_a_nearer_table(self):
        pairs = self.pairs(self.FIGURE, ("t", "TABLE", 100, 430, 300, 500), ("c", "CAPTION", 100, 412, 300, 424),
                           texts={"c": "Figure 4: Overview."})
        self.assertEqual(pairs["f"]["caption_region_id"], "c")

    def test_caption_without_known_text_can_belong_to_a_table_instead(self):
        pairs = self.pairs(self.FIGURE, ("t", "TABLE", 100, 428, 300, 500), ("c", "CAPTION", 100, 412, 300, 424))
        self.assertEqual(pairs, {})

    def test_note_regions_are_not_captions(self):
        note = region("n", "CAPTION", 100, 408, 300, 420, label="vision_footnote")
        self.assertEqual(self.pairs(self.FIGURE, note), {})
        self.assertEqual(caption_candidates([note]), [])

    def test_text_region_that_reads_like_a_caption_is_a_candidate(self):
        regions = (self.FIGURE, ("t", "TEXT", 100, 408, 300, 420))
        self.assertEqual(self.pairs(*regions, texts={"t": "The figure above shows the model."}), {})
        self.assertEqual(self.pairs(*regions, texts={"t": "Figure 3 shows the model."}), {})
        pairs = self.pairs(*regions, texts={"t": "Figure 3: The model."})
        self.assertEqual((pairs["f"]["caption_region_id"], pairs["f"]["text"]), ("t", "Figure 3: The model."))

    def test_extractor_method_returns_the_match_of_one_figure(self):
        regions = [region(*self.FIGURE), region("c", "CAPTION", 100, 408, 300, 420)]
        extractor = FigureExtractor()
        self.assertEqual(extractor.associate_caption(regions[0], regions)["caption_region_id"], "c")
        self.assertIsNone(extractor.associate_caption(regions[0], regions[:1]))

    def test_fixture_two_column_page_would_fool_a_nearest_caption_rule(self):
        chart, diagram = BY_KIND["two_column_bar_chart"], BY_KIND["two_column_diagram_caption_above"]
        own_gap = chart["caption_pdf_bbox"][1] - chart["pdf_bbox"][3]
        other = diagram["caption_pdf_bbox"]
        other_gap = math.hypot(max(other[0] - chart["pdf_bbox"][2], 0), max(other[1] - chart["pdf_bbox"][3], 0))
        self.assertLess(other_gap, own_gap)


# 6. Figure numbers -------------------------------------------------------------------------------------

class FigureNumberTests(unittest.TestCase):
    def test_number_and_caption_are_separated(self):
        cases = {
            "Figure 1: Overview of the model.": ("Figure 1", "Overview of the model."),
            "Fig. 2. Results on the test set.": ("Fig. 2", "Results on the test set."),
            "Figure 3(a) Detail of the encoder.": ("Figure 3(a)", "Detail of the encoder."),
            "Figure 3 (a): Detail.": ("Figure 3(a)", "Detail."),
            "Fig. 4b: Ablation.": ("Fig. 4b", "Ablation."),
            "Fig.5 - Loss curves": ("Fig. 5", "Loss curves"),
            "FIGURE 12. Architecture": ("FIGURE 12", "Architecture"),
            "FIG. 7 | Samples": ("FIG. 7", "Samples"),
            "Figure S1: Supplementary results.": ("Figure S1", "Supplementary results."),
            "Figure A.2: Appendix plot.": ("Figure A.2", "Appendix plot."),
            "Figure 2.3: Numbered by chapter.": ("Figure 2.3", "Numbered by chapter."),
            "  Figure   6 :   Spaced   out.  ": ("Figure 6", "Spaced out."),
            "Figure 8 Overview without a separator": ("Figure 8", "Overview without a separator"),
            "Figure 10—Dash separated": ("Figure 10", "Dash separated"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_figure_number(text), expected)

    def test_number_alone_has_no_caption_text(self):
        self.assertEqual(parse_figure_number("Figure 9"), ("Figure 9", None))
        self.assertEqual(parse_figure_number("Fig. 9."), ("Fig. 9", None))

    def test_uncertain_labels_give_no_number_and_keep_the_text(self):
        for text in ("Figure 3 shows that the loss falls.", "Overview of the model.", "Figures 1 and 2: both.",
                     "See Figure 4: details.", "Table 1: Results.", "Figure: no number.", "Figurine 3: a toy."):
            with self.subTest(text=text):
                self.assertEqual(parse_figure_number(text), (None, text))

    def test_empty_text(self):
        self.assertEqual(parse_figure_number(""), (None, None))
        self.assertEqual(parse_figure_number(None), (None, None))


# 12. Axis scale and numbers ---------------------------------------------------------------------------

class AxisTests(unittest.TestCase):
    def test_parse_number(self):
        for text, value in {"92.1": 92.1, "0": 0.0, "-3": -3.0, "−0.5": -0.5, "+2": 2.0, "1,200": 1200.0,
                            "45%": 45.0, ".5": 0.5, " 12 ": 12.0}.items():
            with self.subTest(text=text):
                self.assertEqual(parse_number(text), value)
        for text in ("ResNet", "Q1", "1e3", "10^3", "0,5", "1.2.3", "", None, "12 kg", "-"):
            with self.subTest(text=text):
                self.assertIsNone(parse_number(text))

    def test_linear_scale_from_tick_labels(self):
        axis = calibrate_axis([(260, 0), (200, 25), (140, 50), (80, 75), (20, 100)])
        self.assertEqual((axis["used"], axis["ticks"]), (5, 5))
        self.assertAlmostEqual(axis["scale"] * 170 + axis["offset"], 37.5)
        self.assertEqual(to_value(axis, 170), 37.5)

    def test_one_misread_label_is_left_out_of_the_scale(self):
        axis = calibrate_axis([(260, 0), (200, 25), (140, 80), (80, 75), (20, 100)])
        self.assertEqual((axis["used"], axis["ticks"]), (4, 5))
        self.assertAlmostEqual(axis["scale"] * 140 + axis["offset"], 50.0)

    def test_no_scale_without_two_different_values(self):
        self.assertIsNone(calibrate_axis([]))
        self.assertIsNone(calibrate_axis([(100, 5)]))
        self.assertIsNone(calibrate_axis([(100, 5), (200, 5)]))
        self.assertIsNone(calibrate_axis([(100, 5), (101, 6)]))        # labels on top of each other

    def test_logarithmic_axis_gives_no_scale(self):
        self.assertIsNone(calibrate_axis([(260, 1), (200, 10), (140, 100), (80, 1000), (20, 10000)]))

    def test_values_are_rounded_to_what_a_pixel_resolves(self):
        self.assertEqual(to_value({"scale": -0.41667, "offset": 108.3}, 100), 66.6)
        self.assertEqual(to_value({"scale": -0.004, "offset": 1.2}, 100), 0.8)
        self.assertEqual(to_value({"scale": -25.0, "offset": 5000}, 100), 2500.0)

    def test_plot_area_from_the_axis_lines(self):
        import numpy as np
        image, _, _ = blank_chart()
        for found, expected in zip(find_plot_area(np.asarray(image)), PLOT):
            self.assertAlmostEqual(found, expected, delta=1)        # the lines are two pixels wide

    def test_no_plot_area_without_axis_lines(self):
        import numpy as np
        image = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(image).rectangle([100, 100, 200, 200], fill=(200, 30, 30))
        self.assertIsNone(find_plot_area(np.asarray(image)))
        ImageDraw.Draw(image).line([(60, 260), (380, 260)], fill="black", width=2)     # an x axis alone
        self.assertIsNone(find_plot_area(np.asarray(image)))

    def test_axis_titles_ticks_and_title_get_their_roles(self):
        image, texts = bar_chart()
        texts += [ChartText("Model", (190, 284, 250, 296)), ChartText("Top-1", (180, 2, 260, 14)),
                  ChartText("Accuracy (%)", (4, 100, 16, 180), vertical=True)]
        data = RuleBasedChartExtractor().extract_chart(image, texts)
        self.assertEqual(data["title"], "Top-1")
        self.assertEqual(data["x_axis"], {"title": "Model", "ticks": ["A", "B", "C"], "numeric": False,
                                          "calibrated": False})
        self.assertEqual(data["y_axis"], {"title": "Accuracy (%)", "ticks": ["0", "25", "50", "75", "100"],
                                          "numeric": True, "calibrated": True})
        roles = {item.text: item.role for item in texts}
        self.assertEqual((roles["Model"], roles["Top-1"], roles["Accuracy (%)"], roles["A"], roles["50"]),
                         ("x_axis_title", "title", "y_axis_title", "x_tick", "y_tick"))


# 8. - 11., 13. Chart reading -----------------------------------------------------------------------------

class ChartReadingTests(unittest.TestCase):
    def read(self, image, texts, **options):
        return RuleBasedChartExtractor().extract_chart(image, texts, **options)

    def test_bar_chart(self):
        data = self.read(*bar_chart())
        self.assertEqual((data["status"], data["chart_type"]), ("parsed", "bar"))
        self.assertGreaterEqual(data["chart_type_score"], 0.9)
        bars = data["series"][0]["bars"]
        self.assertEqual([bar["category"] for bar in bars], ["A", "B", "C"])
        for bar, value in zip(bars, (40, 80, 20)):
            self.assertAlmostEqual(bar["value"], value, delta=1.0)
            self.assertEqual(bar["value_source"], "visual_estimate")
            self.assertEqual(bar["value"], bar["visual_estimate"])
        self.assertEqual(data["series"][0]["colour"], "#285ac8")
        for found, expected in zip(data["plot_area_image_bbox"], PLOT):
            self.assertAlmostEqual(found, expected, delta=1)

    def test_bar_value_printed_above_a_bar_is_used_with_its_source(self):
        image, texts = bar_chart()
        texts.append(tick("41.3", 110, y_pixel(40) - 10))                                  # native text
        texts.append(tick("79.6", 200, y_pixel(80) - 10, confidence=0.93, source="ocr"))   # read by OCR
        bars = self.read(image, texts)["series"][0]["bars"]
        self.assertEqual((bars[0]["value"], bars[0]["value_source"]), (41.3, "label"))
        self.assertEqual((bars[1]["value"], bars[1]["value_source"]), (79.6, "ocr"))
        self.assertEqual(bars[2]["value_source"], "visual_estimate")
        self.assertAlmostEqual(bars[0]["visual_estimate"], 40, delta=1.0)       # the measurement is kept beside it
        self.assertEqual([item.role for item in texts[-2:]], ["value_label", "value_label"])

    def test_bars_without_a_y_scale_have_no_value(self):
        image, texts = bar_chart()
        texts = [item for item in texts if item.text in ("A", "B", "C")]
        data = self.read(image, texts)
        self.assertEqual((data["status"], data["chart_type"]), ("partial", "bar"))
        self.assertEqual([(bar["value"], bar["value_source"]) for bar in data["series"][0]["bars"]], [(None, None)] * 3)
        self.assertTrue(any("y axis not calibrated" in note for note in data["notes"]))
        self.assertIn("no data values recovered", data["notes"])

    def test_bars_below_zero(self):
        image, draw, texts = blank_chart(y_ticks=())
        texts += [tick(str(value), 40, 140 - value * 2.4) for value in (-50, -25, 0, 25, 50)]      # zero at pixel 140
        texts += [tick("A", 110, 274), tick("B", 200, 274)]
        draw.rectangle([85, 140 - 30 * 2.4, 135, 140], fill=(40, 90, 200))
        draw.rectangle([175, 140, 225, 140 + 20 * 2.4], fill=(40, 90, 200))
        bars = self.read(image, texts)["series"][0]["bars"]
        self.assertAlmostEqual(bars[0]["value"], 30, delta=1.0)
        self.assertAlmostEqual(bars[1]["value"], -20, delta=1.0)

    def test_grouped_bars_are_one_series_per_colour(self):
        image, draw, texts = blank_chart(x_ticks=[("A", 130), ("B", 270)])
        for centre, (first, second) in ((130, (40, 60)), (270, (80, 20))):
            draw.rectangle([centre - 40, y_pixel(first), centre - 2, 258], fill=(40, 90, 200))
            draw.rectangle([centre + 2, y_pixel(second), centre + 40, 258], fill=(220, 120, 30))
        series = self.read(image, texts)["series"]
        self.assertEqual(len(series), 2)
        values = {s["colour"]: [(bar["category"], round(bar["value"])) for bar in s["bars"]] for s in series}
        self.assertEqual(values["#285ac8"], [("A", 40), ("B", 80)])
        self.assertEqual(values["#dc781e"], [("A", 60), ("B", 20)])

    def test_line_chart(self):
        data = self.read(*line_chart())
        self.assertEqual((data["status"], data["chart_type"]), ("parsed", "line"))
        points = data["series"][0]["points"]
        self.assertEqual(len(points), 4)
        for point, (x, y) in zip(points, ((1, 20), (2, 70), (3, 40), (4, 90))):
            self.assertAlmostEqual(point["x"], x, delta=0.05)
            self.assertAlmostEqual(point["y"], y, delta=1.5)
            self.assertEqual(point["value_source"], "visual_estimate")
        self.assertEqual(data["x_axis"]["calibrated"], True)

    def test_two_line_series_are_separated_by_colour(self):
        data = self.read(*line_chart(series=(((1, 20), (2, 70), (3, 40), (4, 90)), ((1, 60), (2, 30), (3, 80), (4, 10)))))
        self.assertEqual(data["chart_type"], "line")
        self.assertEqual(len(data["series"]), 2)
        by_colour = {s["colour"]: [round(p["y"] / 10) * 10 for p in s["points"]] for s in data["series"]}
        self.assertEqual(by_colour["#c82828"], [20, 70, 40, 90])
        self.assertEqual(by_colour["#1e8c3c"], [60, 30, 80, 10])

    def test_line_over_categories_reports_the_tick_label_as_x(self):
        image, texts = line_chart()
        for item, label in zip([i for i in texts if i.bbox[1] > 260], ("Jan", "Feb", "Mar", "Apr")):
            item.text = label
        points = self.read(image, texts)["series"][0]["points"]
        self.assertEqual([point["x"] for point in points], ["Jan", "Feb", "Mar", "Apr"])

    def test_scatter_plot(self):
        expected = ((1, 20), (1.5, 60), (2, 35), (3, 80), (3.5, 50), (4, 90))
        data = self.read(*scatter_chart(expected))
        self.assertEqual((data["status"], data["chart_type"]), ("parsed", "scatter"))
        points = data["series"][0]["points"]
        self.assertEqual(len(points), len(expected))
        for point, (x, y) in zip(points, expected):
            self.assertAlmostEqual(point["x"], x, delta=0.05)
            self.assertAlmostEqual(point["y"], y, delta=1.5)
            self.assertEqual(point["value_source"], "visual_estimate")

    def test_scatter_points_without_scales_have_no_values(self):
        image, _ = scatter_chart()
        data = self.read(image, [])
        self.assertEqual((data["status"], data["chart_type"]), ("partial", "scatter"))
        self.assertTrue(all(p["y"] is None and p["value_source"] is None for p in data["series"][0]["points"]))
        self.assertEqual(len(data["series"][0]["points"][0]["image_point"]), 2)     # the pixel position is kept

    def test_legend_names_the_series_by_colour(self):
        image, texts = line_chart(series=(((1, 20), (2, 70), (3, 40), (4, 90)), ((1, 60), (2, 30), (3, 80), (4, 10))))
        draw = ImageDraw.Draw(image)
        draw.line([(70, 36), (100, 36)], fill=(200, 40, 40), width=3)       # top left: clear of both curves
        draw.line([(70, 56), (100, 56)], fill=(30, 140, 60), width=3)
        texts += [ChartText("Method A", (106, 30, 170, 42)), ChartText("Method B", (106, 50, 170, 62))]
        data = self.read(image, texts)
        self.assertEqual(data["legend"], ["Method A", "Method B"])
        self.assertEqual({s["colour"]: s["name"] for s in data["series"]}, {"#c82828": "Method A", "#1e8c3c": "Method B"})
        self.assertEqual([item.role for item in texts[-2:]], ["legend", "legend"])
        self.assertTrue(all(len(s["points"]) == 4 for s in data["series"]))      # the handles are not data

    def test_chart_without_legend_has_unnamed_series(self):
        data = self.read(*line_chart())
        self.assertEqual(data["legend"], [])
        self.assertIsNone(data["series"][0]["name"])

    def test_text_beside_a_bar_is_not_a_legend_entry(self):
        image, texts = bar_chart(values=(80, 30, 20))
        texts.append(ChartText("note", (150, 150, 180, 162)))       # right of the tall first bar
        data = self.read(image, texts)
        self.assertEqual(data["legend"], [])
        self.assertEqual(data["annotations"], ["note"])

    def test_type_is_not_forced_when_the_marks_are_unclear(self):
        image, draw, texts = blank_chart()
        draw.ellipse([120, 60, 320, 240], fill=(200, 60, 60))        # one large disc: no bar, line or point
        data = self.read(image, texts)
        self.assertEqual((data["status"], data["chart_type"]), ("partial", "unknown"))
        self.assertEqual(data["series"], [])
        self.assertTrue(any("chart type not determined" in note for note in data["notes"]))

    def test_empty_plot_area_is_unknown(self):
        image, _, texts = blank_chart()
        data = self.read(image, texts)
        self.assertEqual((data["chart_type"], data["chart_type_score"], data["series"]), ("unknown", None, []))

    def test_frame_around_the_plot_area_is_not_a_series(self):
        image, texts = line_chart()
        draw = ImageDraw.Draw(image)
        draw.line([(60, 20), (380, 20)], fill="black", width=2)
        draw.line([(380, 20), (380, 260)], fill="black", width=2)
        data = self.read(image, texts)
        self.assertEqual((data["chart_type"], len(data["series"])), ("line", 1))

    def test_geometry_is_reported_in_page_pixels(self):
        image, texts = bar_chart()
        for item in texts:
            item.bbox = (item.bbox[0] + 500, item.bbox[1] + 700, item.bbox[2] + 500, item.bbox[3] + 700)
        data = self.read(image, texts, origin=(500, 700))
        for found, expected in zip(data["plot_area_image_bbox"], (560, 720, 880, 960)):
            self.assertAlmostEqual(found, expected, delta=1)
        self.assertAlmostEqual(data["series"][0]["bars"][0]["value"], 40, delta=1.0)
        self.assertEqual(data["series"][0]["bars"][0]["image_bbox"][0], 585)

    def test_corner_cleanup(self):
        self.assertEqual(merge_corners([(0, 0), (100, 0), (102, 1), (200, 50)], 6, 12), [(0, 0), (101, 0.5), (200, 50)])
        rounded = merge_corners([(0, 0), (97, 0), (105, 8), (105, 100)], 3, 20)        # one cut-off corner
        self.assertEqual(len(rounded), 3)
        self.assertAlmostEqual(rounded[1][0], 105)
        self.assertAlmostEqual(rounded[1][1], 0)
        far = [(0, 0), (100, 0), (150, 50), (150, 100)]
        self.assertEqual(merge_corners(far, 3, 20), far)

    def test_reader_errors_become_chart_extraction_errors(self):
        with self.assertRaises(ChartExtractionError):
            RaisingChartExtractor().extract_chart(Image.new("RGB", (50, 50), "white"), [])

    def test_no_axes_is_a_failed_reading_not_an_exception(self):
        data = self.read(Image.new("RGB", (300, 200), "white"), [ChartText("label", (10, 10, 60, 22))])
        self.assertEqual(data, empty_chart_data("rule-based-opencv", notes=["no plot area: no pair of axis lines found"]))
        self.assertEqual(data["status"], "failed")


# 13. Figure types -------------------------------------------------------------------------------------

class FigureTypeTests(unittest.TestCase):
    def photo(self):
        import numpy as np
        y, x = np.mgrid[0:120, 0:160]
        pixels = np.stack([(x * 255 // 160), (y * 255 // 120), ((x + y) * 255 // 280)], axis=-1).astype("uint8")
        return Image.fromarray(pixels, "RGB")

    def drawing(self):
        image = Image.new("RGB", (300, 120), "white")
        draw = ImageDraw.Draw(image)
        for x in (20, 120, 220):
            draw.rectangle([x, 30, x + 60, 90], fill=(220, 230, 245), outline="black")
        return image

    def test_flat_colours_are_a_diagram_and_continuous_tones_an_image(self):
        self.assertEqual(classify_figure_image(self.drawing()), "diagram")
        self.assertEqual(classify_figure_image(self.photo()), "image")
        self.assertGreater(colour_flatness(self.drawing()), 0.95)
        self.assertLess(colour_flatness(self.photo()), 0.3)

    def test_something_between_stays_a_plain_figure(self):
        image = self.drawing()
        image.paste(self.photo().resize((120, 100)), (160, 10))
        self.assertEqual(classify_figure_image(image), "figure")

    def test_chart_label_of_the_layout_model_is_kept(self):
        self.assertEqual(FigureExtractor().classify("chart", self.photo()), ("chart", "layout_label"))

    def test_other_labels_are_typed_by_the_visual_rules(self):
        extractor = FigureExtractor()
        self.assertEqual(extractor.classify("image", self.drawing()), ("diagram", "visual_rule"))
        self.assertEqual(extractor.classify("image", self.photo()), ("image", "visual_rule"))
        self.assertEqual(extractor.classify("image", Image.new("RGB", (80, 80), "white")), ("unknown", "visual_rule"))

    def test_axes_with_numeric_ticks_make_a_chart_without_the_label(self):
        image, texts = bar_chart()
        self.assertEqual(FigureExtractor().classify("image", image, texts), ("chart", "visual_rule"))
        self.assertEqual(FigureExtractor().classify("image", image, [])[0], "diagram")      # axes alone are not enough


# 7. Text in figures -----------------------------------------------------------------------------------

class FigureTextTests(FiguresTestCase):
    def test_ocr_items_that_are_not_text_are_dropped(self):
        items = [ChartText("Backbone", (0, 0, 80, 24), 0.99, "ocr"), ChartText(".", (0, 30, 6, 54), 0.9, "ocr"),
                 ChartText("I", (0, 60, 8, 84), 0.3, "ocr"), ChartText("C", (0, 90, 10, 100), 0.8, "ocr"),
                 ChartText("0.5", (0, 110, 30, 132), 0.97, "ocr")]
        self.assertEqual([item.text for item in usable_ocr_items(items)], ["Backbone", "0.5"])
        self.assertEqual(usable_ocr_items([]), [])

    def test_native_labels_of_a_diagram_are_kept_as_they_are(self):
        result = self.run_figures()
        diagram = all_figures(result["document"])[0]
        self.assertEqual(diagram["figure_type"], "diagram")
        self.assertEqual([item["text"] for item in diagram["ocr_text"]], ["Input", "Backbone", "Transformer", "Classifier"])
        self.assertEqual(diagram["text_source"], "native")
        for item in diagram["ocr_text"]:
            self.assertEqual((item["text_source"], item["confidence"], item["role"]), ("native", None, None))
            self.assertTrue(diagram["crop_bbox"][0] <= item["image_bbox"][0] < item["image_bbox"][2] <= diagram["crop_bbox"][2])
        self.assertEqual(self.extractor.ocr_engine.sizes[1:], [])       # only the bitmap figure went to OCR

    def test_scanned_diagram_is_read_by_the_ocr_engine(self):
        lines = [OcrLine("Input", 0.98, (30, 40, 110, 70)), OcrLine(" Backbone ", 0.91, (200, 40, 330, 70)),
                 OcrLine("~", 0.95, (400, 40, 420, 70)), OcrLine("x", 0.2, (500, 40, 520, 70))]
        self.work = self.prepare_figures(FIGURES_SCANNED, {1: FIGURE_REGIONS[1][:1]})
        engine = FakeEngine(lines=lines)
        result = run_document_figures(self.work, FigureExtractor(engine))
        figure = all_figures(result["document"])[0]
        self.assertEqual(figure["text_source"], "ocr")
        self.assertEqual([(item["text"], item["confidence"], item["text_source"]) for item in figure["ocr_text"]],
                         [("Input", 0.98, "ocr"), ("Backbone", 0.91, "ocr")])
        box = figure["crop_bbox"]
        self.assertEqual(figure["ocr_text"][0]["image_bbox"], [box[0] + 30.0, box[1] + 40.0, box[0] + 110.0, box[1] + 70.0])
        page = self.read("pages.json")["pages"][0]
        self.assertAlmostEqual(figure["ocr_text"][0]["pdf_bbox"][0], (box[0] + 30) / page["scale_x"])

    def test_figure_ocr_can_be_switched_off(self):
        self.work = self.prepare_figures(FIGURES_SCANNED, {1: FIGURE_REGIONS[1][:1]})
        engine = FakeEngine()
        result = run_document_figures(self.work, FigureExtractor(engine, use_ocr=False))
        figure = all_figures(result["document"])[0]
        self.assertEqual((figure["ocr_text"], figure["text_source"], figure["status"]), ([], None, "ok"))
        self.assertEqual(engine.load_count, 0)
        self.assertEqual(result["document"]["settings"]["figure_ocr"], False)
        self.assertIsNone(result["document"]["extractor"]["ocr_engine"])

    def test_no_description_is_written_for_a_figure(self):
        # Extraction only: a figure entry holds what was printed and where, and no field of prose.
        for figure in all_figures(self.run_figures()["document"]):
            self.assertFalse({"description", "summary", "interpretation", "explanation"} & set(figure))
            printed = {item["text"] for item in figure["ocr_text"]}
            if figure["chart_data"] is None:
                self.assertTrue(printed <= {"Input", "Backbone", "Transformer", "Classifier", "Encoder", "Decoder",
                                            "Stage 1", "Stage 2", "Stage 3", "recognized text 1"})


# 2., 14. - 17. The document run ---------------------------------------------------------------------------

class DocumentFiguresTests(FiguresTestCase):
    def test_every_figure_region_becomes_a_figure_with_its_geometry(self):
        result = self.run_figures()
        layout = self.read("layout.json")
        regions = [r for page in layout["pages"] for r in page["regions"] if r["type"] == "FIGURE"]
        figures = all_figures(result["document"])
        self.assertEqual(len(figures), len(regions), 9)
        for figure, source in zip(figures, regions):
            self.assertEqual(figure["source_region_id"], source["region_id"])
            self.assertEqual(figure["source_label"], source["source_label"])
            self.assertEqual(figure["image_bbox"], source["image_bbox"])
            self.assertEqual(figure["pdf_bbox"], source["pdf_bbox"])
            self.assertEqual(figure["page_number"], source["page_number"])
            crop, box = figure["crop_bbox"], figure["image_bbox"]
            self.assertTrue(crop[0] <= box[0] and crop[1] <= box[1] and crop[2] >= box[2] and crop[3] >= box[3])
        self.assertEqual([figure["figure_id"] for figure in figures[:3]], ["p001_f001", "p001_f002", "p002_f001"])

    def test_crop_image_of_every_figure_is_saved_with_the_size_of_its_region(self):
        result = self.run_figures()
        self.assertEqual(len(result["images"]), 9)
        self.assertEqual(result["debug"], [])
        self.assertFalse(os.path.exists(os.path.join(self.work, "figure_crops")))
        for figure in all_figures(result["document"]):
            path = os.path.join(self.work, *figure["image_path"].split("/"))
            self.assertRegex(figure["image_path"], r"^figure_images/page_\d{3}_fig_\d{3}\.png$")
            x0, y0, x1, y1 = figure["image_bbox"]
            with Image.open(path) as image:
                self.assertEqual(image.size, (math.ceil(x1) - math.floor(x0), math.ceil(y1) - math.floor(y0)))

    def test_crop_pixels_are_those_of_the_page_image(self):
        result = self.run_figures()
        figure = all_figures(result["document"])[1]
        x0, y0, x1, y1 = figure["image_bbox"]
        with Image.open(os.path.join(self.work, "pages", "page_001.png")) as page_image:
            expected = page_image.convert("RGB").crop((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)))
        with Image.open(os.path.join(self.work, *figure["image_path"].split("/"))) as crop:
            self.assertEqual(crop.convert("RGB").tobytes(), expected.tobytes())

    def test_debug_crops_and_overlays_only_on_request(self):
        result = self.run_figures(save_debug=True)
        names = sorted(os.path.basename(path) for path in result["debug"])
        self.assertEqual(len(names), 9 + 5)                                 # every padded crop, an overlay per chart
        self.assertIn("page_002_fig_001_overlay.png", names)
        self.assertNotIn("page_001_fig_001_overlay.png", names)             # a diagram has no overlay
        figure = all_figures(result["document"])[2]
        with Image.open(os.path.join(self.work, "figure_crops", "page_002_fig_001.png")) as padded:
            box = figure["crop_bbox"]
            self.assertEqual(padded.size, (box[2] - box[0], box[3] - box[1]))
        with Image.open(os.path.join(self.work, "figure_crops", "page_002_fig_001_overlay.png")) as overlay:
            self.assertEqual(overlay.size, (box[2] - box[0], box[3] - box[1]))

    def test_captions_and_numbers_of_the_fixture(self):
        figures = all_figures(self.run_figures()["document"])
        for figure, expected in zip(figures, TRUTH):
            with self.subTest(kind=expected["kind"]):
                self.assertEqual(figure["caption"], expected["caption"])
                self.assertEqual(figure["figure_number"], expected["figure_number"])
                self.assertEqual(figure["caption_position"], expected["caption_position"])
        self.assertEqual([figure["figure_number"] for figure in figures],
                         ["Figure 1", "Fig. 2", "Figure 3", "Figure 4", "Figure 5", "Figure 6", "Figure 7", "Fig. 8(a)",
                          None])
        first = figures[0]
        self.assertEqual(first["caption_raw"], "Figure 1: Overview of the proposed architecture.")
        self.assertEqual((first["caption_text_source"], first["caption_confidence"]), ("native", None))
        self.assertEqual(first["caption_pdf_bbox"], self.read("layout.json")["pages"][0]["regions"][1]["pdf_bbox"])

    def test_ambiguous_captions_give_no_caption_and_no_number(self):
        figure = all_figures(self.run_figures()["document"])[-1]
        self.assertEqual((figure["caption"], figure["figure_number"], figure["caption_region_id"]), (None, None, None))
        self.assertEqual(figure["status"], "ok")

    def test_padding_does_not_take_in_the_caption(self):
        result = self.run_figures(extractor=FigureExtractor(FakeEngine(), padding=40))
        layout = self.read("layout.json")
        for figure in all_figures(result["document"]):
            crop = figure["crop_bbox"]
            for other in layout["pages"][figure["page_number"] - 1]["regions"]:
                if other["type"] == "CAPTION":
                    box = other["image_bbox"]
                    inside = crop[0] < box[2] and crop[2] > box[0] and crop[1] < box[3] - 1 and crop[3] > box[1] + 1
                    self.assertFalse(inside, f"{figure['figure_id']} crop reaches into {other['region_id']}")

    def test_types_of_the_fixture(self):
        figures = all_figures(self.run_figures()["document"])
        self.assertEqual([figure["figure_type"] for figure in figures], [expected["figure_type"] for expected in TRUTH])
        self.assertEqual({figure["figure_type_source"] for figure in figures if figure["figure_type"] == "chart"},
                         {"layout_label"})
        self.assertTrue(all((figure["chart_data"] is not None) == (figure["figure_type"] == "chart")
                            for figure in figures))

    def test_charts_of_the_fixture_from_native_text(self):
        result = self.run_figures()
        evaluation = evaluate_figures(TRUTH, result["document"])
        for key in ("type_accuracy", "caption_accuracy", "number_accuracy", "label_accuracy", "chart_type_accuracy",
                    "axis_title_accuracy", "tick_label_accuracy", "legend_accuracy", "exact_value_accuracy",
                    "estimated_points_found"):
            with self.subTest(metric=key):
                self.assertEqual(evaluation[key], 1.0)
        self.assertLess(evaluation["normalized_rmse"], 0.01)
        summary = result["document"]["summary"]
        self.assertEqual((summary["figure_regions"], summary["generic_figures"], summary["charts"]), (9, 4, 5))
        self.assertEqual(summary["chart_types"], {"bar": 2, "line": 2, "scatter": 1})
        self.assertEqual((summary["charts_parsed"], summary["chart_data_extracted"], summary["chart_failures"]), (5, 5, 0))
        self.assertEqual((summary["captions_matched"], summary["figure_numbers_found"], summary["failures"]), (8, 8, 0))

    def test_every_number_says_where_it_comes_from(self):
        figures = all_figures(self.run_figures()["document"])
        labelled, plain = figures[2]["chart_data"], figures[6]["chart_data"]
        self.assertEqual([(bar["category"], bar["value"], bar["value_source"]) for bar in labelled["series"][0]["bars"]],
                         [("ResNet", 72.1, "label"), ("ViT", 84.7, "label"), ("ConvNeXt", 63.4, "label"),
                          ("Swin", 91.2, "label")])
        self.assertTrue(all(bar["value_source"] == "visual_estimate" for bar in plain["series"][0]["bars"]))
        for bar, value in zip(labelled["series"][0]["bars"], (72.1, 84.7, 63.4, 91.2)):
            self.assertAlmostEqual(bar["visual_estimate"], value, delta=0.6)
            self.assertNotEqual(bar["visual_estimate"], bar["value"])        # the measurement is not passed off as the label
        for figure in figures:
            for series in (figure["chart_data"] or {"series": []})["series"]:
                for point in series.get("points", ()):
                    self.assertEqual(point["value_source"], "visual_estimate")
        roles = [item["role"] for item in figures[2]["ocr_text"]]
        self.assertEqual(roles.count("value_label"), 4)
        self.assertEqual(roles.count("y_tick"), 6)

    def test_legend_and_series_names_of_the_fixture(self):
        chart = all_figures(self.run_figures()["document"])[5]["chart_data"]
        self.assertEqual(chart["legend"], ["Method A", "Method B"])
        self.assertEqual(sorted(series["name"] for series in chart["series"]), ["Method A", "Method B"])
        self.assertEqual(all_figures(self.run_figures()["document"])[3]["chart_data"]["legend"], [])

    def test_chart_reader_failure_keeps_the_figure_and_its_crop(self):
        result = self.run_figures(extractor=FigureExtractor(FakeEngine(), RaisingChartExtractor()))
        figure = all_figures(result["document"])[2]
        self.assertEqual((figure["status"], figure["figure_type"]), ("ok", "chart"))
        self.assertEqual(figure["chart_data"]["status"], "failed")
        self.assertEqual(figure["chart_data"]["chart_type"], "unknown")
        self.assertEqual(figure["chart_data"]["series"], [])
        self.assertIn("simulated chart failure", figure["chart_data"]["notes"][0])
        self.assertTrue(os.path.isfile(os.path.join(self.work, *figure["image_path"].split("/"))))
        self.assertEqual(figure["figure_number"], "Figure 3")
        self.assertEqual(len(figure["ocr_text"]), 17)                        # its text is kept too
        summary = result["document"]["summary"]
        self.assertEqual((summary["chart_failures"], summary["charts_parsed"], summary["failures"]), (5, 0, 0))

    def test_figure_that_fails_is_kept_with_its_crop_and_the_run_goes_on(self):
        class FailingSecond(FigureExtractor):
            calls = 0

            def extract(self, *args, **kwargs):
                type(self).calls += 1
                if type(self).calls == 2:
                    raise RuntimeError("simulated figure failure")
                return super().extract(*args, **kwargs)

        result = self.run_figures(extractor=FailingSecond(FakeEngine()))
        figures = all_figures(result["document"])
        failed = figures[1]
        self.assertEqual((failed["status"], failed["figure_type"], failed["chart_data"]), ("failed", "unknown", None))
        self.assertEqual(failed["error"], "RuntimeError: simulated figure failure")
        self.assertTrue(os.path.isfile(os.path.join(self.work, *failed["image_path"].split("/"))))
        self.assertEqual(failed["figure_number"], "Fig. 2")                 # the caption does not depend on the crop
        self.assertEqual([figure["status"] for figure in figures].count("ok"), 8)
        self.assertEqual(result["document"]["summary"]["failures"], 1)

    def test_region_that_cannot_be_cut_out_fails_alone(self):
        regions = {1: [("image", (100.0, 100.0, 101.0, 101.0))] + FIGURE_REGIONS[1]}
        result = self.run_figures(regions=regions)
        figures = all_figures(result["document"])
        self.assertEqual([figure["status"] for figure in figures], ["failed", "ok", "ok"])
        self.assertIsNone(figures[0]["image_path"])
        self.assertIn("too small", figures[0]["error"])

    def test_figures_json_is_written_and_valid(self):
        result = self.run_figures()
        self.assertEqual(result["figures_path"], os.path.join(self.work, "metadata", "figures.json"))
        document = self.read("figures.json")
        self.assertEqual(document, json.loads(json.dumps(result["document"])))
        self.assertEqual(validate_figures_document(document), [])
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["extractor"]["name"], "figure-rules + rule-based-opencv")
        self.assertEqual((document["extractor"]["version"], document["extractor"]["device"]), ("1.0", "cpu"))
        self.assertEqual(document["source"]["source_sha256"], self.read("pages.json")["document"]["source_sha256"])
        self.assertEqual([page["page_number"] for page in document["pages"]], [1, 2, 3, 4, 5])
        for key in ("figure_id", "page_number", "image_bbox", "pdf_bbox", "figure_type", "source_label", "caption",
                    "figure_number", "image_path", "ocr_text", "chart_data", "source_region_id"):
            self.assertIn(key, document["pages"][0]["figures"][0])
        self.assertNotIn("timings", document)

    def test_chart_data_schema(self):
        chart = all_figures(self.run_figures()["document"])[5]["chart_data"]
        self.assertEqual(set(chart), {"status", "extractor", "chart_type", "chart_type_score", "plot_area_image_bbox",
                                      "title", "x_axis", "y_axis", "legend", "series", "annotations", "notes"})
        self.assertEqual(set(chart["x_axis"]), {"title", "ticks", "numeric", "calibrated"})
        self.assertEqual(set(chart["series"][0]), {"name", "colour", "points"})
        self.assertEqual(set(chart["series"][0]["points"][0]), {"x", "y", "value_source", "image_point"})
        bars = all_figures(self.read("figures.json"))[2]["chart_data"]["series"][0]
        self.assertEqual(set(bars["bars"][0]), {"category", "value", "value_source", "visual_estimate", "image_bbox"})

    def test_validation_finds_broken_documents(self):
        document = self.run_figures()["document"]

        def broken(change):
            copy = json.loads(json.dumps(document))
            change(copy)
            return validate_figures_document(copy)

        self.assertEqual(validate_figures_document([]), ["root is not an object"])
        self.assertIn("pages missing or empty", validate_figures_document({"extractor": {"name": "x"}}))
        self.assertTrue(broken(lambda d: d["pages"][0]["figures"][0].update(figure_type="plot")))
        self.assertTrue(broken(lambda d: d["pages"][0]["figures"][0].update(image_bbox=[5, 5, 1, 1])))
        self.assertTrue(broken(lambda d: d["pages"][0]["figures"][0].update(chart_data=empty_chart_data("x"))))
        self.assertTrue(broken(lambda d: d["pages"][1]["figures"][0].update(chart_data=None)))
        self.assertTrue(broken(lambda d: d["pages"][0]["figures"][0].update(caption_region_id=None)))
        self.assertTrue(broken(lambda d: d["pages"][0]["figures"][0].update(image_path=None)))
        self.assertTrue(broken(lambda d: d["pages"][0]["figures"][0]["ocr_text"][0].update(confidence=0.5)))
        self.assertTrue(broken(lambda d: d["pages"][1]["figures"][0]["chart_data"]["series"][0]["bars"][0].update(
            value_source="guess")))
        self.assertTrue(broken(lambda d: d["pages"][1]["figures"][0]["chart_data"]["series"][0]["bars"][0].update(
            value_source=None)))
        self.assertTrue(broken(lambda d: d["pages"][1]["figures"][0]["chart_data"].update(status="partial")))
        self.assertTrue(broken(lambda d: d["pages"].reverse()))

    def test_other_artifacts_are_not_modified(self):
        self.work = self.prepare_figures()
        metadata = os.path.join(self.work, "metadata")
        for name in ("tables.json", "equations.json"):
            with open(os.path.join(metadata, name), "w", encoding="utf-8") as fh:
                fh.write('{"kept": true}\n')
        names = ("pages.json", "layout.json", "ocr.json", "tables.json", "equations.json")
        before = {name: sha256(os.path.join(metadata, name)) for name in names}
        page_image = sha256(os.path.join(self.work, "pages", "page_001.png"))
        run_document_figures(self.work, FigureExtractor(FakeEngine()))
        self.assertEqual({name: sha256(os.path.join(metadata, name)) for name in names}, before)
        self.assertEqual(sha256(os.path.join(self.work, "pages", "page_001.png")), page_image)
        self.assertEqual(sorted(os.listdir(metadata)), sorted(names + ("figures.json",)))

    def test_ocr_engine_is_loaded_once_and_only_when_needed(self):
        self.work = self.prepare_figures(FIGURES_SCANNED)
        engine = FakeEngine()
        loads_before = engine.load_count
        result = run_document_figures(self.work, FigureExtractor(engine))
        self.assertEqual(engine.load_count - loads_before, 1)
        self.assertGreaterEqual(len(engine.sizes), 9)                    # every figure of the scanned copy
        self.assertEqual(result["document"]["summary"]["with_text"], 9)

        self.work = self.prepare_figures(regions={2: FIGURE_REGIONS[2]}, name="native")
        engine = FakeEngine()
        run_document_figures(self.work, FigureExtractor(engine))
        self.assertEqual((engine.load_count, engine.sizes), (0, []))     # vector charts: native text only

    def test_chart_reader_is_set_up_once(self):
        extractor = RuleBasedChartExtractor()
        self.assertEqual((extractor.loaded, extractor.load_count), (False, 0))
        for _ in range(3):
            extractor.extract_chart(*bar_chart())
        self.assertEqual((extractor.loaded, extractor.load_count), (True, 1))
        self.assertEqual(extractor.describe()["name"], "rule-based-opencv")

    def test_timings_are_returned(self):
        timings = self.run_figures()["timings"]
        self.assertEqual(len(timings["figure_s"]), 9)
        self.assertGreater(timings["total_s"], 0)
        self.assertGreaterEqual(timings["engine_init_s"], 0)

    def test_page_without_figures_and_document_without_figures(self):
        result = self.run_figures(regions={1: [("text", (72.0, 72.0, 500.0, 110.0))]})
        self.assertEqual(all_figures(result["document"]), [])
        self.assertEqual(result["document"]["summary"]["figure_regions"], 0)
        self.assertEqual(result["images"], [])
        self.assertEqual(self.extractor.ocr_engine.load_count, 0)

    def test_stale_results_are_removed(self):
        self.run_figures(save_debug=True)
        stale = [os.path.join(self.work, "figure_images", "page_009_fig_001.png"),
                 os.path.join(self.work, "figure_crops", "page_009_fig_001_overlay.png")]
        for path in stale:
            Image.new("RGB", (4, 4)).save(path)
        run_document_figures(self.work, FigureExtractor(FakeEngine()))
        self.assertFalse(any(os.path.exists(path) for path in stale))
        self.assertEqual(os.listdir(os.path.join(self.work, "figure_crops")), [])
        self.assertEqual(len(os.listdir(os.path.join(self.work, "figure_images"))), 9)

    def test_missing_or_foreign_input_is_refused_and_leaves_no_figures_json(self):
        self.work = self.prepare_figures()
        target = os.path.join(self.work, "metadata", "figures.json")
        run_document_figures(self.work, FigureExtractor(FakeEngine()))
        os.remove(os.path.join(self.work, "metadata", "ocr.json"))
        with self.assertRaises(FigureInputError):
            run_document_figures(self.work, FigureExtractor(FakeEngine()))
        self.assertFalse(os.path.exists(target))
        os.remove(os.path.join(self.work, "metadata", "layout.json"))
        with self.assertRaises(FigureInputError):
            run_document_figures(self.work, FigureExtractor(FakeEngine()))


# Metrics ----------------------------------------------------------------------------------------------

class MetricsTests(unittest.TestCase):
    TRUTH_CHART = {
        "chart_type": "bar", "title": "T", "x_axis": {"title": "X", "ticks": ["A", "B"]},
        "y_axis": {"title": "Y", "ticks": ["0", "50", "100"]}, "legend": [], "value_labels": True,
        "series": [{"name": None, "bars": [{"category": "A", "value": 40.0}, {"category": "B", "value": 80.0}]}],
    }

    def chart(self, **changes):
        chart = {"chart_type": "bar", "title": "T", "x_axis": {"title": "X", "ticks": ["A", "B"]},
                 "y_axis": {"title": "Y", "ticks": ["0", "50", "100"]}, "legend": [],
                 "series": [{"name": None, "bars": [
                     {"category": "A", "value": 40.0, "value_source": "label", "visual_estimate": 41.0},
                     {"category": "B", "value": 80.0, "value_source": "ocr", "visual_estimate": 77.0}]}]}
        chart.update(changes)
        return chart

    def test_perfect_chart(self):
        result = evaluate_chart(self.TRUTH_CHART, self.chart())
        self.assertTrue(result["chart_type_match"])
        self.assertEqual((result["titles_correct"], result["titles_expected"]), (3, 3))
        self.assertEqual((result["ticks_correct"], result["ticks_expected"]), (5, 5))
        self.assertEqual((result["labelled_exact"], result["labelled_expected"]), (2, 2))
        self.assertEqual((result["points_found"], result["points_expected"]), (2, 2))
        self.assertAlmostEqual(result["mae"], 2.0)                      # |41 - 40| and |77 - 80|
        self.assertAlmostEqual(result["rmse"], math.sqrt(5.0))
        self.assertEqual(result["y_range"], 100.0)

    def test_an_estimate_does_not_count_as_an_exact_value(self):
        chart = self.chart()
        chart["series"][0]["bars"][0].update(value=40.0, value_source="visual_estimate")
        chart["series"][0]["bars"][1].update(value=80.5)
        self.assertEqual(evaluate_chart(self.TRUTH_CHART, chart)["labelled_exact"], 0)

    def test_wrong_and_missing_labels_are_counted(self):
        chart = self.chart(title="Other", chart_type="line")
        chart["y_axis"] = {"title": None, "ticks": ["0", "100"]}
        result = evaluate_chart(self.TRUTH_CHART, chart)
        self.assertFalse(result["chart_type_match"])
        self.assertEqual((result["titles_correct"], result["ticks_correct"]), (1, 4))

    def test_chart_that_was_not_read(self):
        result = evaluate_chart(self.TRUTH_CHART, None)
        self.assertEqual((result["chart_type_match"], result["titles_correct"], result["points_found"],
                          result["labelled_exact"], result["mae"]), (False, 0, 0, 0, None))

    def test_line_points_are_matched_by_x_and_legend_by_name(self):
        truth = {"chart_type": "line", "title": None, "x_axis": {"title": None, "ticks": ["1", "2", "3"]},
                 "y_axis": {"title": None, "ticks": ["0", "10"]}, "legend": ["A", "B"],
                 "series": [{"name": "A", "points": [{"x": 1, "y": 2.0}, {"x": 2, "y": 4.0}, {"x": 3, "y": 6.0}]},
                            {"name": "B", "points": [{"x": 1, "y": 1.0}, {"x": 2, "y": 1.0}, {"x": 3, "y": 1.0}]}]}
        chart = {"chart_type": "line", "title": None, "x_axis": {"title": None, "ticks": ["1", "2", "3"]},
                 "y_axis": {"title": None, "ticks": ["0", "10"]}, "legend": ["A"],
                 "series": [{"name": "A", "points": [{"x": 1.02, "y": 2.5, "value_source": "visual_estimate"},
                                                     {"x": 2.9, "y": 6.0, "value_source": "visual_estimate"}]}]}
        result = evaluate_chart(truth, chart)
        self.assertEqual((result["legend_correct"], result["legend_expected"], result["legend_exact"]), (1, 2, False))
        self.assertEqual((result["points_found"], result["points_expected"]), (2, 6))
        self.assertAlmostEqual(result["mae"], 0.25)

    def test_document_metrics(self):
        def figure(figure_id, bbox, figure_type, caption=None, number=None, chart=None, texts=()):
            return {"figure_id": figure_id, "page_number": 1, "pdf_bbox": bbox, "figure_type": figure_type,
                    "caption": caption, "figure_number": number, "chart_data": chart,
                    "ocr_text": [{"text": text} for text in texts]}

        truth = [
            {"page_number": 1, "pdf_bbox": [0, 0, 100, 100], "figure_type": "diagram", "caption": "A box.",
             "figure_number": "Figure 1", "labels": ["In", "Out"]},
            {"page_number": 1, "pdf_bbox": [0, 200, 100, 300], "figure_type": "chart", "caption": None,
             "figure_number": None, "chart": self.TRUTH_CHART},
            {"page_number": 1, "pdf_bbox": [0, 400, 100, 500], "figure_type": "image", "caption": "Lost.",
             "figure_number": "Figure 3", "labels": []},
        ]
        document = {"pages": [{"page_number": 1, "figures": [
            figure("p001_f001", [0, 0, 100, 100], "diagram", "A box.", "Figure 1", texts=["In", "x"]),
            figure("p001_f002", [0, 200, 100, 300], "chart", chart=self.chart()),
        ]}]}
        result = evaluate_figures(truth, document)
        self.assertEqual((result["figures"], result["found"], result["charts"]), (3, 2, 1))
        for key in ("type_accuracy", "caption_accuracy", "number_accuracy"):
            self.assertAlmostEqual(result[key], 2 / 3)                 # the figure that was not found counts as wrong
        self.assertEqual(result["label_accuracy"], 0.5)
        self.assertEqual((result["chart_type_accuracy"], result["exact_value_accuracy"]), (1.0, 1.0))
        self.assertAlmostEqual(result["normalized_mae"], 0.02)          # 2.0 on an axis that spans 100
        self.assertIsNone(result["legend_accuracy"])                   # nothing to count

    def test_no_figures_gives_no_rates(self):
        result = evaluate_figures([], {"pages": [{"page_number": 1, "figures": []}]})
        self.assertEqual((result["figures"], result["type_accuracy"], result["normalized_mae"]), (0, None, None))


# CLI ----------------------------------------------------------------------------------------------------

class InspectFiguresToolErrorTests(OcrTestCase):
    def run_tool(self, tool, *arguments):
        return subprocess.run([sys.executable, os.path.join(ENGINE, "tools", tool), *arguments],
                              capture_output=True, text=True, timeout=600)

    def test_missing_pdf_exits_with_2(self):
        completed = self.run_tool("inspect_figures.py", self.path("missing.pdf"), "--output", self.path("out"))
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Error:", completed.stderr)
        self.assertFalse(os.path.exists(self.path("out", "metadata", "figures.json")))

    def test_bad_padding_exits_with_2(self):
        completed = self.run_tool("inspect_figures.py", FIGURES_PDF, "--output", self.path("out"),
                                  "--figure-padding", "-4")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("padding", completed.stderr)

    def test_evaluation_tool_reports_the_metrics(self):
        work = self.prepare(FIGURES_PDF, FIGURE_REGIONS)
        run_document_ocr(work, FakeEngine())
        run_document_figures(work, FigureExtractor(FakeEngine()))
        completed = self.run_tool("evaluate_figures.py", work, "--truth", TRUTH_PATH)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for line in ("Figures: 9 of 9 found", "Caption match accuracy: 1.000", "Figure number accuracy: 1.000",
                     "Figure-type accuracy: 1.000", "Chart type accuracy: 1.000", "Legend text accuracy: 1.000",
                     "Exact value accuracy (values printed in the chart): 1.000 (4 values)"):
            self.assertIn(line, completed.stdout)
        self.assertEqual(self.run_tool("evaluate_figures.py", self.path("nowhere"), "--truth", TRUTH_PATH).returncode, 2)


# Real models ---------------------------------------------------------------------------------------------

@unittest.skipUnless(HAVE_OCR_MODEL, "paddleocr is not installed or the model tests are switched off")
class FiguresModelIntegrationTests(OcrTestCase):
    """The whole pipeline with PP-DocLayoutV3 and PP-OCRv5 on the two copies of the fixture."""

    @classmethod
    def setUpClass(cls):
        cls.detector = PPDocLayoutV3Detector()
        cls.engine = PaddleOCREngine()

    def pipeline(self, pdf):
        work = self.path("doc")
        process_pdf(pdf, work)
        detect_document_layout(work, self.detector)
        run_document_ocr(work, self.engine)
        return run_document_figures(work, FigureExtractor(self.engine))

    def test_layout_model_separates_charts_from_other_figures(self):
        result = self.pipeline(FIGURES_PDF)
        figures = all_figures(result["document"])
        self.assertEqual([figure["source_label"] for figure in figures],
                         ["image", "image", "chart", "chart", "chart", "chart", "chart", "image", "image"])
        self.assertEqual([figure["figure_type"] for figure in figures],
                         ["diagram", "image", "chart", "chart", "chart", "chart", "chart", "diagram", "diagram"])

    def test_digital_fixture(self):
        result = self.pipeline(FIGURES_PDF)
        evaluation = evaluate_figures(TRUTH, result["document"])
        self.assertEqual(evaluation["found"], 9)
        for key in ("type_accuracy", "chart_type_accuracy", "axis_title_accuracy", "tick_label_accuracy",
                    "legend_accuracy", "exact_value_accuracy", "estimated_points_found", "label_accuracy"):
            with self.subTest(metric=key):
                self.assertEqual(evaluation[key], 1.0)
        self.assertLess(evaluation["normalized_rmse"], 0.02)
        # Eight of nine: the layout model reports only one of the two caption lines of the ambiguous
        # figure, so that figure gets a caption although the truth is "undecidable".
        self.assertGreaterEqual(evaluation["caption_accuracy"], 8 / 9)
        figures = all_figures(result["document"])
        self.assertEqual((figures[6]["figure_number"], figures[7]["figure_number"]), ("Figure 7", "Fig. 8(a)"))
        self.assertEqual(figures[7]["caption_position"], "above")
        bars = figures[2]["chart_data"]["series"][0]["bars"]
        self.assertEqual([(bar["value"], bar["value_source"]) for bar in bars],
                         [(72.1, "label"), (84.7, "label"), (63.4, "label"), (91.2, "label")])

    def test_scanned_fixture_goes_through_ocr(self):
        loads = self.engine.load_count
        result = self.pipeline(FIGURES_SCANNED)
        self.assertLessEqual(self.engine.load_count, max(loads, 1))     # loaded once for all tests of the class
        evaluation = evaluate_figures(TRUTH, result["document"])
        self.assertEqual(evaluation["found"], 9)
        for key in ("type_accuracy", "chart_type_accuracy", "legend_accuracy", "exact_value_accuracy",
                    "estimated_points_found", "label_accuracy"):
            with self.subTest(metric=key):
                self.assertEqual(evaluation[key], 1.0)
        self.assertGreaterEqual(evaluation["axis_title_accuracy"], 0.9)
        self.assertGreaterEqual(evaluation["tick_label_accuracy"], 0.9)
        self.assertGreaterEqual(evaluation["caption_accuracy"], 8 / 9)
        self.assertLess(evaluation["normalized_rmse"], 0.02)
        figures = all_figures(result["document"])
        self.assertTrue(all(figure["text_source"] in ("ocr", None) for figure in figures))
        bars = figures[2]["chart_data"]["series"][0]["bars"]
        self.assertEqual({bar["value_source"] for bar in bars}, {"ocr"})
        titles = [item for item in figures[2]["ocr_text"] if item["role"] == "y_axis_title"]
        self.assertEqual([(item["text"], item["vertical"]) for item in titles], [("Accuracy (%)", True)])
        self.assertTrue(all(0 <= item["confidence"] <= 1 for figure in figures for item in figure["ocr_text"]))


@unittest.skipUnless(HAVE_OCR_MODEL, "paddleocr is not installed or the model tests are switched off")
class InspectFiguresToolIntegrationTests(OcrTestCase):
    def run_tool(self, *arguments):
        return subprocess.run([sys.executable, os.path.join(ENGINE, "tools", "inspect_figures.py"), *arguments],
                              capture_output=True, text=True, timeout=900)

    def test_tool_runs_all_phases_and_reuses_them(self):
        work = self.path("out")
        completed = self.run_tool(FIGURES_PDF, "--output", work, "--save-figure-debug")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for line in ("Pages: 5", "Figure regions: 9", "Generic figures: 4", "Charts: 5", "Bar: 2", "Line: 2",
                     "Scatter: 1", "Chart data extracted: 5/5", "Total time:",
                     "Phase 2 pages: created; Phase 3 layout: created; Phase 4 OCR: created"):
            self.assertIn(line, completed.stdout)
        with open(os.path.join(work, "metadata", "figures.json"), encoding="utf-8") as fh:
            self.assertEqual(validate_figures_document(json.load(fh)), [])
        self.assertEqual(len(os.listdir(os.path.join(work, "figure_images"))), 9)
        self.assertEqual(len(os.listdir(os.path.join(work, "figure_crops"))), 14)

        again = self.run_tool(work)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("Phase 2 pages: reused; Phase 3 layout: reused; Phase 4 OCR: reused", again.stdout)
        self.assertFalse(os.listdir(os.path.join(work, "figure_crops")))     # debug output only on request


if __name__ == "__main__":
    unittest.main()

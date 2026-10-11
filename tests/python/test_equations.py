"""Phase 6 tests: equation crops, LaTeX normalization and validation, equation numbers, the recognizer
interface, equations.json, metrics.

    python -m unittest discover -s tests/python -v

The unit tests use a fake layout detector, a fake math recognizer and a fake OCR engine and need neither
PaddleOCR nor a model download. The classes named *IntegrationTests run the real PP-DocLayoutV3,
PP-FormulaNet_plus-S and PP-OCRv5 models (first run downloads about 250 MB for PP-FormulaNet_plus-S);
they are skipped when paddleocr, tokenizers or ftfy is not installed or when AIRPA_SKIP_LAYOUT_MODEL=1,
AIRPA_SKIP_OCR_MODEL=1 or AIRPA_SKIP_MATH_MODEL=1.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

from test_layout import FakeDetector, TempDirTestCase, sha256
from test_ocr import DIGITAL, HAVE_OCR_MODEL, FakeEngine, OcrTestCase

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)

from pipeline import (  # noqa: E402
    MathInputError,
    MathModelError,
    MathPrediction,
    MathRecognitionError,
    MathRecognizer,
    OcrLine,
    PaddleOCREngine,
    PPDocLayoutV3Detector,
    PPFormulaNetRecognizer,
    canonical_latex,
    crop_box,
    crop_equation,
    detect_document_layout,
    equation_crop_box,
    evaluate_equation,
    evaluate_equations,
    find_equation_numbers,
    latex_cer,
    normalize_latex,
    parse_equation_number,
    pdf_bbox_to_pixels,
    process_pdf,
    run_document_equations,
    run_document_ocr,
    split_tag,
    validate_equations_document,
    validate_latex,
)
from pipeline.math_ocr import recognize_equation  # noqa: E402

EQUATIONS_PDF = os.path.join(FIXTURES, "equations", "synthetic_equations.pdf")                  # 4 pages, native text
EQUATIONS_SCANNED = os.path.join(FIXTURES, "equations", "synthetic_equations_scanned.pdf")      # the same as bitmaps
TRUTH_PATH = os.path.join(FIXTURES, "equations", "synthetic_equations.json")
with open(TRUTH_PATH, encoding="utf-8") as _fh:
    TRUTH = json.load(_fh)["equations"]
BY_KIND = {equation["kind"]: equation for equation in TRUTH}

HAVE_MATH_MODEL = (HAVE_OCR_MODEL and os.environ.get("AIRPA_SKIP_MATH_MODEL") != "1"
                   and all(importlib.util.find_spec(name) for name in ("tokenizers", "ftfy")))

# The fixture's equations and their numbers as layout regions: {page: [(model label, PDF box)]}.
EQUATION_REGIONS = {}
for _equation in TRUTH:
    _regions = EQUATION_REGIONS.setdefault(_equation["page_number"], [])
    _regions.append(("display_formula", tuple(_equation["pdf_bbox"])))
    if _equation["number_pdf_bbox"]:
        _regions.append(("formula_number", tuple(_equation["number_pdf_bbox"])))

# What PP-FormulaNet_plus-S writes for the fixture (its own spelling), by kind.
MODEL_STYLE = {
    "simple_algebra": r"y=m x+b",
    "summation": r"L=-{\sum_{i}}y_{i}\operatorname{l o g}(p_{i})",
    "fraction": r"z{=}\frac{x+y}{2}",
    "square_root": r"\sqrt{x^{2}+y^{2}}",
    "greek": r"\alpha+\beta=\lambda",
    "subscript_superscript": r"x_{i}^{2}",
    "integral": r"\int_{a}^{b}f(x)d x",
    "numbered": r"E=m c^{2}",
    "matrix": r"A=\left[\begin{matrix}{a}&{b}\\ {c}&{d}\\ \end{matrix}\right]",
    "multiline_aligned": r"\begin{aligned}{f(x)}&{{}=(x+1)^{2}}\\ {}&{{}=x^{2}+2x+1}\\ \end{aligned}",
    "symbol_fidelity_digits": r"x_{0}\times l_{1}=10",
    "symbol_fidelity_greek": r"\lambda A=a-\alpha",
}
ANSWERS = [MODEL_STYLE[equation["kind"]] for equation in TRUTH]


class FakeRecognizer(MathRecognizer):
    """Returns the given answers, one per call (text, MathPrediction, or an Exception to raise), then
    `default`; counts loads and calls."""

    name = "fake-math"

    def __init__(self, answers=(), fail_load=False, default="a+b"):
        super().__init__()
        self.answers = list(answers)
        self.fail_load = fail_load
        self.default = default
        self.sizes = []                 # size of every image received

    def _load_model(self):
        if self.fail_load:
            raise RuntimeError("simulated load failure")
        return object()

    def _recognize(self, image):
        self.sizes.append(image.size)
        answer = self.answers[len(self.sizes) - 1] if len(self.sizes) <= len(self.answers) else self.default
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, MathPrediction) else MathPrediction(answer)


def all_equations(document):
    return [equation for page in document["pages"] for equation in page["equations"]]


def page_with_formula(width=600, height=400):
    """A white page with ink in the box (200, 150)-(400, 200)."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x in range(200, 400, 12):
        draw.line([(x, 155), (x + 6, 195)], fill="black", width=2)
    return image


class EquationsTestCase(OcrTestCase):
    def prepare_equations(self, pdf=EQUATIONS_PDF, name="doc", regions=None):
        """Phase 2 and a layout.json with the fixture's equation and equation-number regions."""
        return self.prepare(pdf, EQUATION_REGIONS if regions is None else regions, name)

    def run_equations(self, answers=ANSWERS, pdf=EQUATIONS_PDF, **options):
        self.work = self.prepare_equations(pdf)
        self.recognizer = FakeRecognizer(answers)
        return run_document_equations(self.work, self.recognizer, **options)

    def read(self, name):
        with open(os.path.join(self.work, "metadata", name), encoding="utf-8") as fh:
            return json.load(fh)


class NormalizationTests(unittest.TestCase):
    def test_whitespace(self):
        self.assertEqual(normalize_latex("  y =  m x\n+ b \t"), "y = m x + b")

    def test_wrappers_around_the_whole_formula_are_removed(self):
        for wrapped in ("$E = mc^2$", "$$E = mc^2$$", r"\[E = mc^2\]", r"\( E = mc^2 \)",
                        r"\begin{equation} E = mc^2 \end{equation}", r"\begin{equation*}E = mc^2\end{equation*}",
                        "<s> E = mc^2 </s>", "E = mc^2 <pad><pad>"):
            self.assertEqual(normalize_latex(wrapped), "E = mc^2", wrapped)

    def test_delimiters_inside_a_formula_are_kept(self):
        for text in ("$a$ + $b$", r"\begin{aligned} a &= b \end{aligned}", r"\left[ x \right]", r"a \$ b"):
            self.assertEqual(normalize_latex(text), text)

    def test_the_formula_itself_is_never_rewritten(self):
        for text in ("a+b", "b+a", "x_i^2", "x^2_i", r"\frac{x+y}{2}", "(x+y)/2", "0O1l", r"x \times x", "a-b=c",
                     r"\alpha a \lambda A", r"{{a}}", r"\mathrm{l o g}"):
            self.assertEqual(normalize_latex(text), text)

    def test_empty(self):
        for text in ("", "   ", "$$", "$ $"):
            self.assertEqual(normalize_latex(text), "")


class TagTests(unittest.TestCase):
    def test_a_trailing_tag_is_the_equation_number(self):
        self.assertEqual(split_tag(r"E = mc^2 \tag{4}"), ("E = mc^2", "(4)"))
        self.assertEqual(split_tag(r"E = mc^2\tag*{A1}"), ("E = mc^2", "A1"))
        self.assertEqual(split_tag(r"a \tag { 3.2 } "), ("a", "(3.2)"))

    def test_anything_else_is_left_alone(self):
        for text in ("E = mc^2", r"E = mc^2 \qquad (4)", r"\tag{4} E = mc^2", r"a \tag{1} \\ b \tag{2}", r"a \tag{}"):
            self.assertEqual(split_tag(text), (text, None))


class ValidationTests(unittest.TestCase):
    def test_sound_formulas(self):
        for latex in list(MODEL_STYLE.values()) + [equation["latex"] for equation in TRUTH] + [
                r"\begin{cases} x & x > 0 \\ -x & \text{otherwise} \end{cases}",
                r"\left\{ \frac{a}{b} \right.", r"a \, b \; c \! d \quad e \\ f", r"\{ a, b \}", r"x^{2}_{i}",
                r"{}_{a}X^{b}", r"\mathrm{sum} + \text{sqrt of } x", r"\operatorname{frac}(x)", r"50\%",
                r"\begin{array}{cc} a & b \\ c & d \end{array}", r"\|x\|_2 \le \varepsilon"]:
            self.assertEqual(validate_latex(latex), [], latex)

    def test_empty(self):
        for latex in ("", "   ", None):
            self.assertEqual(validate_latex(latex), ["the formula is empty"])

    def test_braces(self):
        self.assertTrue(any("not closed" in p for p in validate_latex(r"\frac{x+y}{2")))
        self.assertTrue(any("} without {" in p for p in validate_latex(r"x^{2}}")))
        self.assertEqual(validate_latex(r"\{ x \}"), [])                # escaped braces are symbols
        self.assertTrue(validate_latex(r"\{ x }"))

    def test_environments(self):
        self.assertTrue(any("is not closed" in p for p in validate_latex(r"\begin{matrix} a & b")))
        self.assertTrue(any("does not close" in p for p in validate_latex(r"a & b \end{matrix}")))
        self.assertTrue(validate_latex(r"\begin{matrix} a \end{aligned}"))
        self.assertTrue(validate_latex(r"\begin{aligned} \begin{matrix} a \end{aligned} \end{matrix}"))

    def test_left_and_right(self):
        self.assertTrue(any("\\left" in p for p in validate_latex(r"\left( x + y")))
        self.assertTrue(validate_latex(r"x + y \right)"))
        self.assertEqual(validate_latex(r"\left( \left[ x \right] \right)"), [])

    def test_scripts_without_argument(self):
        for latex in ("x_", "x^", "x_{i}^", "{x_}", "x__i", "a_&b"):
            self.assertTrue(any("no argument" in p for p in validate_latex(latex)), latex)

    def test_control_sequences(self):
        for latex in ("x + \\", r"x \1 y", "x \\é y", "\\" + "a" * 30 + " x"):
            self.assertTrue(any("invalid control sequence" in p for p in validate_latex(latex)), latex)

    def test_command_without_its_backslash(self):
        self.assertTrue(any("sum" in p for p in validate_latex(r"L = - sum_i y_i")))
        self.assertTrue(any("frac" in p for p in validate_latex(r"frac{a}{b}")))
        self.assertEqual(validate_latex(r"s u m"), [])                  # three variables
        self.assertEqual(validate_latex(r"abc + xyz"), [])

    def test_what_the_model_really_wrote_for_the_scanned_summation(self):
        problems = validate_latex(r"L\mathrm{=}-\ \ {sum{{_}i}}y_{i}\operatorname{l o g}(p_{i})")
        self.assertEqual(problems, ["_ has no argument", "command without a backslash: sum"])

    def test_a_decoder_stuck_in_a_loop(self):
        self.assertTrue(any("repeated" in p for p in validate_latex("x = " + r"\quad " * 40)))
        self.assertEqual(validate_latex("1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1"), [])
        self.assertEqual(validate_latex("x" * 15), [])


class EquationNumberTests(unittest.TestCase):
    EQUATION = (250.0, 270.0, 360.0, 290.0)

    @staticmethod
    def line(text, *bbox, source="native"):
        return (bbox, text, source)

    def numbers(self, *lines, equations=None):
        found = find_equation_numbers(equations or [self.EQUATION], list(lines))
        return [entry and entry["number"] for entry in found]

    def test_parse(self):
        for text, number in (("(4)", "(4)"), (" ( 4 ) ", "(4)"), ("(3.2)", "(3.2)"), ("(12a)", "(12a)"),
                             ("(A.1)", "(A.1)"), ("(B3)", "(B3)"), ("(2-1)", "(2-1)")):
            self.assertEqual(parse_equation_number(text), number, text)
        for text in ("4", "[4]", "Eq. (4)", "(4) holds", "f(2)", "(x)", "(a)", "()", "(1234)", "(4", "", None,
                     "(1), (2)"):
            self.assertIsNone(parse_equation_number(text), text)

    def test_number_at_the_right_margin(self):
        found = find_equation_numbers([self.EQUATION], [self.line("(4)", 526, 272, 540, 287)])
        self.assertEqual(found, [{"number": "(4)", "source": "native", "bbox": [526, 272, 540, 287]}])

    def test_equation_without_a_number(self):
        self.assertEqual(self.numbers(), [None])
        self.assertEqual(self.numbers(self.line("some text", 72, 272, 200, 287)), [None])

    def test_a_number_mentioned_in_running_text_is_not_one(self):
        self.assertEqual(self.numbers(self.line("as in Eq. (1), where", 400, 272, 540, 287)), [None])

    def test_a_number_on_another_line_is_not_this_equations(self):
        self.assertEqual(self.numbers(self.line("(4)", 526, 300, 540, 315)), [None])        # below
        self.assertEqual(self.numbers(self.line("(4)", 526, 240, 540, 255)), [None])        # above

    def test_a_number_on_the_left_is_not_taken(self):
        self.assertEqual(self.numbers(self.line("(4)", 72, 272, 86, 287)), [None])

    def test_number_inside_the_right_part_of_the_region(self):
        wide = (250.0, 270.0, 545.0, 290.0)         # the layout box includes the number
        self.assertEqual(self.numbers(self.line("(4)", 526, 272, 540, 287), equations=[wide]), ["(4)"])

    def test_two_numbers_are_ambiguous(self):
        self.assertEqual(self.numbers(self.line("(4)", 526, 272, 540, 287), self.line("(5)", 480, 272, 494, 287)),
                         [None])

    def test_each_number_goes_to_the_nearest_equation(self):
        # Two columns: the left equation has no number; "(5)" belongs to the right one.
        left, right = (90.0, 270.0, 200.0, 290.0), (340.0, 270.0, 450.0, 290.0)
        self.assertEqual(self.numbers(self.line("(5)", 526, 272, 540, 287), equations=[left, right]), [None, "(5)"])
        # Each column with its own number.
        self.assertEqual(self.numbers(self.line("(4)", 276, 272, 290, 287), self.line("(5)", 526, 272, 540, 287),
                                      equations=[left, right]), ["(4)", "(5)"])

    def test_stacked_equations(self):
        upper, lower = (250.0, 270.0, 360.0, 290.0), (250.0, 300.0, 360.0, 320.0)
        self.assertEqual(self.numbers(self.line("(1)", 526, 272, 540, 287), self.line("(2)", 526, 302, 540, 317),
                                      equations=[upper, lower]), ["(1)", "(2)"])

    def test_the_source_of_the_text_is_kept(self):
        found = find_equation_numbers([self.EQUATION], [self.line("(4)", 526, 272, 540, 287, source="ocr")])
        self.assertEqual(found[0]["source"], "ocr")


class CropTests(unittest.TestCase):
    def test_box_is_the_region_with_padding(self):
        self.assertEqual(equation_crop_box((200.4, 150.2, 399.5, 199.9), 600, 400, 8), (192, 142, 408, 208))
        self.assertEqual(equation_crop_box((200, 150, 400, 200), 600, 400, 0), (200, 150, 400, 200))
        self.assertEqual(equation_crop_box((200, 150, 400, 200), 600, 400), crop_box((200, 150, 400, 200), 600, 400, 8))

    def test_padding_stops_at_the_page_edges(self):
        self.assertEqual(equation_crop_box((2, 3, 598, 399), 600, 400, 8), (0, 0, 600, 400))

    def test_the_crop_is_the_page_pixels_unchanged(self):
        page = page_with_formula()
        before = page.tobytes()
        crop, box = crop_equation(page, (200, 150, 400, 200), 8)

        self.assertEqual(box, (192, 142, 408, 208))
        self.assertEqual(crop.size, (216, 66))
        self.assertEqual(crop.mode, "RGB")                          # no grayscale, no resize, no contrast change
        self.assertEqual(crop.tobytes(), page.crop(box).tobytes())
        self.assertEqual(page.tobytes(), before)                    # the page image is not modified

    def test_an_equation_number_is_cut_off(self):
        number = (360, 160, 395, 190)
        self.assertEqual(equation_crop_box((200, 150, 400, 200), 600, 400, 8, [number]), (192, 142, 358, 208))
        # A number beside the region, beyond the padding, changes nothing.
        self.assertEqual(equation_crop_box((200, 150, 400, 200), 600, 400, 8, [(500, 160, 540, 190)]),
                         (192, 142, 408, 208))
        # A number in the padding only is cut off as well.
        self.assertEqual(equation_crop_box((200, 150, 400, 200), 600, 400, 8, [(404, 160, 440, 190)]),
                         (192, 142, 402, 208))
        # Nothing would be left of the formula: no cut.
        self.assertEqual(equation_crop_box((200, 150, 400, 200), 600, 400, 8, [(193, 160, 395, 190)]),
                         (192, 142, 408, 208))

    def test_regions_that_cannot_be_cut_out(self):
        page = page_with_formula()
        for bbox in ((700, 10, 800, 60), (10, 10, 12, 300), (5, 5, 5, 5), "box", (0, 0, float("nan"), 5)):
            with self.assertRaises(MathRecognitionError):
                crop_equation(page, bbox)
        with self.assertRaises(ValueError):
            crop_equation(page, (200, 150, 400, 200), padding=-1)


class RecognizerTests(unittest.TestCase):
    def test_model_loads_once(self):
        recognizer = FakeRecognizer()
        self.assertEqual((recognizer.loaded, recognizer.load_count), (False, 0))
        for _ in range(3):
            recognizer.recognize(page_with_formula())
        recognizer.load()
        self.assertEqual((recognizer.loaded, recognizer.load_count, len(recognizer.sizes)), (True, 1, 3))

    def test_load_failure(self):
        with self.assertRaises(MathModelError):
            FakeRecognizer(fail_load=True).load()

    def test_the_prediction_is_the_models_own_text(self):
        raw = "  $ y =  m x + b $ "
        prediction = FakeRecognizer([raw]).recognize(page_with_formula())
        self.assertEqual(prediction, MathPrediction(raw, None))

    def test_confidence_is_never_invented(self):
        image = page_with_formula()
        self.assertIsNone(FakeRecognizer(["x"]).recognize(image).confidence)
        self.assertEqual(FakeRecognizer([MathPrediction("x", 0.91)]).recognize(image).confidence, 0.91)
        for bad in (1.7, -0.1, float("nan"), "high", True):
            self.assertIsNone(FakeRecognizer([MathPrediction("x", bad)]).recognize(image).confidence)

    def test_model_errors_become_recognition_errors(self):
        with self.assertRaises(MathRecognitionError) as caught:
            FakeRecognizer([RuntimeError("out of memory")]).recognize(page_with_formula())
        self.assertIn("RuntimeError: out of memory", str(caught.exception))
        with self.assertRaises(MathRecognitionError):
            FakeRecognizer([MathPrediction(None)]).recognize(page_with_formula())

    def test_recognize_region_sends_the_padded_crop(self):
        recognizer = FakeRecognizer(["x^{2}"])
        result = recognizer.recognize_region(page_with_formula(), (200, 150, 400, 200), padding=8)

        self.assertEqual(result["prediction"].latex, "x^{2}")
        self.assertEqual(result["crop_bbox"], [192, 142, 408, 208])
        self.assertEqual((result["crop"].size, result["blank"]), ((216, 66), False))
        self.assertEqual(recognizer.sizes, [(216, 66)])

    def test_a_blank_region_is_not_sent_to_the_model(self):
        recognizer = FakeRecognizer()
        result = recognizer.recognize_region(page_with_formula(), (20, 20, 120, 80))
        self.assertEqual((result["prediction"], result["blank"]), (None, True))
        self.assertEqual((recognizer.sizes, recognizer.load_count), ([], 0))

    def test_paddle_recognizer_describes_itself_without_loading(self):
        recognizer = PPFormulaNetRecognizer(device="cpu")
        described = recognizer.describe()
        self.assertEqual((described["name"], described["provider"], described["device"]),
                         ("PP-FormulaNet_plus-S", "PaddleOCR", "cpu"))
        self.assertEqual(set(described), {"name", "provider", "version", "framework", "device"})
        self.assertFalse(recognizer.loaded)
        self.assertEqual(PPFormulaNetRecognizer(model_name="PP-FormulaNet_plus-M").name, "PP-FormulaNet_plus-M")


class RecognizeEquationTests(unittest.TestCase):
    """One crop at a time: model text -> normalized LaTeX -> status. The model's spelling of every
    fixture formula must pass, unchanged except for whitespace."""

    def fields(self, answer):
        return recognize_equation(FakeRecognizer([answer]), page_with_formula())

    def assert_recognized(self, kind):
        raw = "  " + MODEL_STYLE[kind].replace(" ", "  ") + " "
        fields = self.fields(raw)
        self.assertEqual((fields["status"], fields["problems"]), ("success", []))
        self.assertEqual(fields["raw_prediction"], raw)                     # exactly what the model returned
        self.assertEqual(fields["latex"], MODEL_STYLE[kind])
        self.assertEqual(fields["normalized_latex"], fields["latex"])
        self.assertIsNone(fields["confidence"])
        self.assertEqual(canonical_latex(fields["latex"]), canonical_latex(BY_KIND[kind]["latex"]))
        return fields

    def test_simple_equation(self):
        self.assert_recognized("simple_algebra")

    def test_fraction(self):
        self.assertIn(r"\frac{x+y}{2}", self.assert_recognized("fraction")["latex"])

    def test_summation(self):
        self.assertIn(r"\sum_{i}", self.assert_recognized("summation")["latex"])

    def test_square_root_and_integral(self):
        self.assert_recognized("square_root")
        self.assert_recognized("integral")

    def test_greek_symbols(self):
        latex = self.assert_recognized("greek")["latex"]
        for symbol in (r"\alpha", r"\beta", r"\lambda"):
            self.assertIn(symbol, latex)

    def test_subscript_and_superscript(self):
        self.assertEqual(self.assert_recognized("subscript_superscript")["latex"], "x_{i}^{2}")

    def test_matrix_is_one_block(self):
        latex = self.assert_recognized("matrix")["latex"]
        self.assertEqual((latex.count(r"\begin{matrix}"), latex.count(r"\end{matrix}"), latex.count("&")), (1, 1, 2))

    def test_multiline_equation_is_one_block(self):
        latex = self.assert_recognized("multiline_aligned")["latex"]
        self.assertTrue(latex.startswith(r"\begin{aligned}") and latex.endswith(r"\end{aligned}"))
        self.assertEqual(latex.count(r"\\"), 2)                             # the lines are not split apart

    def test_cases(self):
        fields = self.fields(r"|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}")
        self.assertEqual(fields["status"], "success")

    def test_symbols_are_kept_as_the_model_read_them(self):
        for raw in ("x_{0}", "x_{O}", "l_{1}", "1_{l}", r"a\times b", "a x b", "a-b", "a=b", r"\alpha", "a",
                    r"\lambda A"):
            self.assertEqual(self.fields(raw)["latex"], raw)

    def test_a_trailing_tag_becomes_the_number(self):
        fields = self.fields(r"E=m c^{2}\tag{4}")
        self.assertEqual((fields["latex"], fields["tag_number"], fields["raw_prediction"]),
                         ("E=m c^{2}", "(4)", r"E=m c^{2}\tag{4}"))

    def test_invalid_latex_keeps_the_prediction(self):
        raw = r"z = \frac{x+y}{2"
        fields = self.fields(raw)
        self.assertEqual((fields["status"], fields["latex"]), ("invalid_latex", None))
        self.assertEqual((fields["raw_prediction"], fields["normalized_latex"]), (raw, raw))
        self.assertTrue(fields["problems"])

    def test_an_empty_answer_is_a_failure(self):
        for raw in ("", "   ", "$$"):
            with self.assertRaises(MathRecognitionError):
                self.fields(raw)


class DocumentEquationsTests(EquationsTestCase):
    def test_every_equation_region_becomes_one_equation(self):
        result = self.run_equations()
        equations = all_equations(result["document"])

        self.assertEqual([e["equation_id"] for e in equations],
                         [f"p{p:03d}_e{i:03d}" for p in (1, 2, 3, 4) for i in (1, 2, 3)])
        self.assertEqual([e["status"] for e in equations], ["success"] * 12)
        self.assertEqual([e["latex"] for e in equations], ANSWERS)
        self.assertEqual([e["raw_prediction"] for e in equations], ANSWERS)
        self.assertEqual({e["source_model"] for e in equations}, {"fake-math"})
        self.assertEqual({e["confidence"] for e in equations}, {None})
        self.assertEqual(len(self.recognizer.sizes), 12)            # equation numbers are not sent to the model
        self.assertEqual(result["document"]["summary"], {
            "pages": 4, "equation_regions": 12, "recognized": 12, "invalid_latex": 0, "recognition_failed": 0,
            "with_equation_numbers": 6, "mean_confidence": None})

    def test_only_equation_regions_are_processed(self):
        regions = {1: [("text", (72.0, 108.0, 540.0, 154.0)), ("table", (72.0, 400.0, 540.0, 500.0)),
                       ("image", (72.0, 520.0, 300.0, 700.0)), ("formula_number", (526.0, 271.0, 540.0, 286.0)),
                       ("inline_formula", (300.0, 120.0, 340.0, 134.0)), EQUATION_REGIONS[1][0]]}
        work = self.prepare_equations(regions=regions)
        recognizer = FakeRecognizer()
        document = run_document_equations(work, recognizer)["document"]

        self.assertEqual([e["source_label"] for e in all_equations(document)], ["inline_formula", "display_formula"])
        self.assertEqual(len(recognizer.sizes), 2)

    def test_regions_keep_their_identity_and_boxes(self):
        result = self.run_equations()
        regions = [r for page in self.read("layout.json")["pages"] for r in page["regions"] if r["type"] == "EQUATION"]
        pages = self.read("pages.json")["pages"]
        equations = all_equations(result["document"])

        self.assertEqual(len(regions), 12)
        for equation, region in zip(equations, regions):
            page = pages[equation["page_number"] - 1]
            self.assertEqual((equation["source_region_id"], equation["page_number"], equation["source_label"]),
                             (region["region_id"], region["page_number"], "display_formula"))
            self.assertEqual((equation["image_bbox"], equation["pdf_bbox"]), (region["image_bbox"], region["pdf_bbox"]))
            for i, scale in enumerate((page["scale_x"], page["scale_y"]) * 2):
                self.assertAlmostEqual(equation["pdf_bbox"][i], equation["image_bbox"][i] / scale, places=6)
            self.assertEqual(tuple(equation["crop_bbox"]),
                             crop_box(region["image_bbox"], page["image_width"], page["image_height"], 8))
        # The model got exactly those crops.
        self.assertEqual(self.recognizer.sizes,
                         [(e["crop_bbox"][2] - e["crop_bbox"][0], e["crop_bbox"][3] - e["crop_bbox"][1]) for e in equations])
        # Every equation lies where the fixture drew it.
        for equation, truth in zip(equations, TRUTH):
            for found, drawn in zip(equation["pdf_bbox"], truth["pdf_bbox"]):
                self.assertAlmostEqual(found, drawn, delta=0.5)

    def test_equation_numbers_from_native_text(self):
        equations = all_equations(self.run_equations()["document"])

        self.assertEqual([e["equation_number"] for e in equations], [t["equation_number"] for t in TRUTH])
        self.assertEqual([e["equation_number_source"] for e in equations],
                         ["native" if t["equation_number"] else None for t in TRUTH])
        numbered = next(e for e in equations if e["equation_number"] == "(4)")
        self.assertEqual((numbered["latex"], numbered["page_number"]), ("E=m c^{2}", 3))    # the number is not in the LaTeX
        self.assertGreater(numbered["equation_number_pdf_bbox"][0], numbered["pdf_bbox"][2])

    def test_a_number_inside_the_region_is_kept_out_of_the_crop(self):
        truth = BY_KIND["numbered"]
        wide = (truth["pdf_bbox"][0], truth["pdf_bbox"][1], 545.0, truth["pdf_bbox"][3])    # equation and its "(4)"
        work = self.prepare_equations(regions={3: [("display_formula", wide)]})
        recognizer = FakeRecognizer(["E=m c^{2}"])
        equation, = all_equations(run_document_equations(work, recognizer)["document"])
        with open(os.path.join(work, "metadata", "pages.json"), encoding="utf-8") as fh:
            page = json.load(fh)["pages"][2]

        self.assertEqual((equation["equation_number"], equation["equation_number_source"]), ("(4)", "native"))
        number_x0 = equation["equation_number_pdf_bbox"][0] * page["scale_x"]
        self.assertLess(equation["crop_bbox"][2], number_x0)
        self.assertLess(equation["crop_bbox"][2], equation["image_bbox"][2])
        self.assertEqual(recognizer.sizes[0][0], equation["crop_bbox"][2] - equation["crop_bbox"][0])
        self.assertAlmostEqual(equation["pdf_bbox"][2], 545.0, delta=0.5)       # the region itself is not changed

    def test_scanned_pages_get_their_numbers_from_phase_4_ocr_text(self):
        work = self.prepare_equations(EQUATIONS_SCANNED)
        without = all_equations(run_document_equations(work, FakeRecognizer(ANSWERS))["document"])
        self.assertEqual({e["equation_number"] for e in without}, {None})       # no native text, no ocr.json

        # Phase 4 reads every equation-number region as "(7)" (one OCR line in the middle of its crop).
        run_document_ocr(work, FakeEngine(lines=[OcrLine(" (7) ", 0.9, (12, 12, 40, 40))]))
        result = run_document_equations(work, FakeRecognizer(ANSWERS))
        equations = all_equations(result["document"])
        self.assertEqual([e["equation_number"] for e in equations], ["(7)" if t["equation_number"] else None for t in TRUTH])
        self.assertEqual({e["equation_number_source"] for e in equations if e["equation_number"]}, {"ocr"})
        self.assertEqual(result["document"]["settings"]["equation_numbers_from_ocr"], True)
        self.assertEqual(result["document"]["source"]["ocr_json"], "metadata/ocr.json")

        ignored = run_document_equations(work, FakeRecognizer(ANSWERS), use_ocr_text=False)["document"]
        self.assertEqual({e["equation_number"] for e in all_equations(ignored)}, {None})

    def test_a_tag_written_by_the_model_is_used_when_the_page_shows_no_number(self):
        answers = [r"y=m x+b\tag{9}", r"L=-\sum_{i}y_{i}\tag{9}"]
        equations = all_equations(self.run_equations(answers)["document"])
        self.assertEqual((equations[0]["latex"], equations[0]["equation_number"], equations[0]["equation_number_source"]),
                         ("y=m x+b", "(9)", "latex_tag"))
        # The number printed on the page wins; the tag is removed from the LaTeX either way.
        self.assertEqual((equations[1]["latex"], equations[1]["equation_number"], equations[1]["equation_number_source"]),
                         (r"L=-\sum_{i}y_{i}", "(1)", "native"))
        self.assertEqual(equations[1]["raw_prediction"], answers[1])

    def test_failed_recognition_keeps_the_equation_and_its_image(self):
        answers = list(ANSWERS)
        answers[4] = RuntimeError("simulated model failure")
        answers[7] = "   "
        result = self.run_equations(answers)
        equations = all_equations(result["document"])

        self.assertEqual([e["status"] for e in equations],
                         ["success"] * 4 + ["recognition_failed"] + ["success"] * 2 + ["recognition_failed"] + ["success"] * 4)
        failed = equations[4]
        self.assertEqual((failed["latex"], failed["normalized_latex"], failed["raw_prediction"]), (None, None, None))
        self.assertEqual(failed["error"], "MathRecognitionError: RuntimeError: simulated model failure")
        self.assertEqual(failed["image_path"], "equation_crops/page_002_eq_002.png")
        self.assertIn("empty formula", equations[7]["error"])
        self.assertEqual(equations[7]["equation_number"], "(4)")            # what is known is kept
        # The crop is on disk and is the region of the page image.
        with Image.open(os.path.join(self.work, *failed["image_path"].split("/"))) as crop, \
                Image.open(os.path.join(self.work, "pages", "page_002.png")) as page:
            self.assertEqual(crop.convert("RGB").tobytes(), page.convert("RGB").crop(tuple(failed["crop_bbox"])).tobytes())
        self.assertEqual(sorted(os.listdir(os.path.join(self.work, "equation_crops"))),
                         ["page_002_eq_002.png", "page_003_eq_002.png"])
        self.assertEqual(len(result["crops"]), 2)
        for key, value in (("recognized", 10), ("recognition_failed", 2), ("invalid_latex", 0), ("equation_regions", 12)):
            self.assertEqual(result["document"]["summary"][key], value)
        self.assertEqual(validate_equations_document(result["document"]), [])

    def test_invalid_latex_is_flagged_and_keeps_prediction_and_image(self):
        answers = list(ANSWERS)
        answers[1] = r"  L\mathrm{=}-\ \ {sum{{_}i}}y_{i}  "
        result = self.run_equations(answers)
        invalid = all_equations(result["document"])[1]

        self.assertEqual((invalid["status"], invalid["latex"]), ("invalid_latex", None))
        self.assertEqual(invalid["raw_prediction"], answers[1])
        self.assertEqual(invalid["normalized_latex"], r"L\mathrm{=}-\ \ {sum{{_}i}}y_{i}")
        self.assertEqual(invalid["problems"], ["_ has no argument", "command without a backslash: sum"])
        self.assertNotIn("error", invalid)
        self.assertEqual(invalid["image_path"], "equation_crops/page_001_eq_002.png")
        self.assertTrue(os.path.isfile(os.path.join(self.work, "equation_crops", "page_001_eq_002.png")))
        self.assertEqual(invalid["equation_number"], "(1)")
        self.assertEqual((result["document"]["summary"]["recognized"], result["document"]["summary"]["invalid_latex"]),
                         (11, 1))
        self.assertEqual(validate_equations_document(result["document"]), [])

    def test_a_region_that_cannot_be_read_fails_alone(self):
        regions = {1: [EQUATION_REGIONS[1][0], ("display_formula", (100.0, 600.0, 300.0, 640.0)),     # empty paper
                       ("display_formula", (100.0, 700.0, 101.0, 740.0))]}                           # 2.5 px wide
        work = self.prepare_equations(regions=regions)
        recognizer = FakeRecognizer(["y=m x+b"])
        document = run_document_equations(work, recognizer)["document"]
        good, blank, tiny = all_equations(document)

        self.assertEqual((good["status"], good["latex"]), ("success", "y=m x+b"))
        self.assertEqual((blank["status"], blank["error"]), ("recognition_failed",
                                                             "MathRecognitionError: the equation region is blank"))
        self.assertEqual(blank["image_path"], "equation_crops/page_001_eq_002.png")
        self.assertEqual((tiny["status"], tiny["crop_bbox"], tiny["image_path"]), ("recognition_failed", None, None))
        self.assertIn("too small", tiny["error"])
        self.assertEqual(len(recognizer.sizes), 1)                  # neither was sent to the model
        self.assertEqual(validate_equations_document(document), [])

    def test_equations_json(self):
        result = self.run_equations()

        self.assertEqual(result["equations_path"], os.path.join(self.work, "metadata", "equations.json"))
        data = self.read("equations.json")
        self.assertEqual(data, result["document"])
        self.assertEqual(validate_equations_document(data), [])
        self.assertEqual(set(data), {"schema_version", "recognizer", "settings", "source", "coordinate_system", "pages",
                                     "summary"})
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["recognizer"], {"name": "fake-math"})
        self.assertEqual(data["settings"], {"padding_px": 8, "equation_numbers_from_ocr": False})
        self.assertEqual(data["source"], {"pages_json": "metadata/pages.json", "layout_json": "metadata/layout.json",
                                          "ocr_json": None, "source_pdf": "synthetic_equations.pdf",
                                          "source_sha256": sha256(EQUATIONS_PDF)})
        self.assertEqual([(p["page_number"], len(p["equations"])) for p in data["pages"]], [(1, 3), (2, 3), (3, 3), (4, 3)])
        self.assertEqual(set(data["pages"][0]["equations"][0]), {
            "equation_id", "page_number", "source_region_id", "source_label", "source_model", "image_bbox", "pdf_bbox",
            "crop_bbox", "latex", "normalized_latex", "raw_prediction", "confidence", "status", "problems",
            "equation_number", "equation_number_source", "equation_number_pdf_bbox", "image_path"})
        with open(result["equations_path"], encoding="utf-8") as fh:
            self.assertIn('"latex": "L=-{\\\\sum_{i}}y_{i}', fh.read())        # backslashes survive serialization

    def test_equations_json_is_identical_across_runs(self):
        first = self.run_equations()
        digest = sha256(first["equations_path"])
        again = run_document_equations(self.work, FakeRecognizer(ANSWERS))
        self.assertEqual(sha256(again["equations_path"]), digest)

    def test_a_model_confidence_is_stored_and_averaged(self):
        answers = [MathPrediction("y=m x+b", 0.9), MathPrediction("x", 0.7), MathPrediction("x_", 0.1)]
        document = self.run_equations(answers)["document"]
        self.assertEqual([e["confidence"] for e in all_equations(document)[:4]], [0.9, 0.7, 0.1, None])
        self.assertAlmostEqual(document["summary"]["mean_confidence"], 0.8)     # recognized equations only

    def test_earlier_artifacts_are_not_modified(self):
        work = self.prepare_equations()
        run_document_ocr(work, FakeEngine())
        names = ("metadata/pages.json", "metadata/layout.json", "metadata/ocr.json", "pages/page_001.png")
        before = {name: sha256(os.path.join(work, *name.split("/"))) for name in names}

        run_document_equations(work, FakeRecognizer(ANSWERS), save_crops=True)

        self.assertEqual(before, {name: sha256(os.path.join(work, *name.split("/"))) for name in names})
        self.assertEqual(sorted(os.listdir(os.path.join(work, "metadata"))),
                         ["equations.json", "layout.json", "ocr.json", "pages.json"])

    def test_model_is_loaded_once_for_all_equations_and_documents(self):
        result = self.run_equations()
        self.assertEqual((self.recognizer.load_count, len(self.recognizer.sizes)), (1, 12))
        timings = result["timings"]
        self.assertEqual(set(timings), {"model_init_s", "equation_s", "inference_s", "total_s"})
        self.assertEqual(sorted(timings["equation_s"]), [e["equation_id"] for e in all_equations(result["document"])])
        self.assertAlmostEqual(timings["inference_s"], sum(timings["equation_s"].values()))
        self.assertGreaterEqual(timings["total_s"], timings["inference_s"])

        other = self.prepare_equations(EQUATIONS_SCANNED, "other")
        again = run_document_equations(other, self.recognizer)
        self.assertEqual((self.recognizer.load_count, len(self.recognizer.sizes)), (1, 24))
        self.assertEqual(again["timings"]["model_init_s"], 0.0)

    def test_document_without_equations_does_not_load_the_model(self):
        work = self.prepare(DIGITAL, {1: [("text", (68.4, 78.4, 513.2, 134.4))]})
        recognizer = FakeRecognizer()
        result = run_document_equations(work, recognizer)

        self.assertEqual([p["equations"] for p in result["document"]["pages"]], [[], []])
        self.assertEqual((recognizer.load_count, result["timings"]["model_init_s"]), (0, 0.0))
        self.assertEqual(result["document"]["summary"]["equation_regions"], 0)
        self.assertEqual(validate_equations_document(result["document"]), [])
        self.assertFalse(os.path.exists(os.path.join(work, "equation_crops")))

    def test_crops_are_saved_only_on_request(self):
        result = self.run_equations()
        self.assertEqual(result["crops"], [])
        self.assertFalse(os.path.exists(os.path.join(self.work, "equation_crops")))
        self.assertEqual({e["image_path"] for e in all_equations(result["document"])}, {None})

        result = run_document_equations(self.work, FakeRecognizer(ANSWERS), save_crops=True)
        names = sorted(os.listdir(os.path.join(self.work, "equation_crops")))
        self.assertEqual(names, [f"page_{p:03d}_eq_{i:03d}.png" for p in (1, 2, 3, 4) for i in (1, 2, 3)])
        self.assertEqual([e["image_path"] for e in all_equations(result["document"])],
                         ["equation_crops/" + name for name in names])
        first = all_equations(result["document"])[0]
        with Image.open(result["crops"][0]) as crop:
            self.assertEqual(crop.size, (first["crop_bbox"][2] - first["crop_bbox"][0],
                                         first["crop_bbox"][3] - first["crop_bbox"][1]))

        # Crops of an earlier run do not survive a run that saves none.
        run_document_equations(self.work, FakeRecognizer(ANSWERS))
        self.assertEqual(os.listdir(os.path.join(self.work, "equation_crops")), [])

    def test_padding(self):
        result = self.run_equations(padding=0)
        first = all_equations(result["document"])[0]
        self.assertEqual(result["document"]["settings"]["padding_px"], 0)
        self.assertLessEqual(first["crop_bbox"][2] - first["crop_bbox"][0], first["image_bbox"][2] - first["image_bbox"][0] + 2)
        with self.assertRaises(ValueError):
            run_document_equations(self.work, FakeRecognizer(), padding=-3)

    def test_missing_or_mismatched_artifacts_and_no_file_after_a_failed_run(self):
        with self.assertRaises(Exception) as caught:
            run_document_equations(self.path("nothing"), FakeRecognizer())
        self.assertIn("pages.json", str(caught.exception))

        work = self.path("doc")
        process_pdf(EQUATIONS_PDF, work)                # no layout.json yet
        with self.assertRaises(MathInputError) as caught:
            run_document_equations(work, FakeRecognizer())
        self.assertIn("layout.json", str(caught.exception))

        detect_document_layout(work, FakeDetector({}))
        run_document_equations(work, FakeRecognizer())
        path = os.path.join(work, "metadata", "equations.json")
        self.assertTrue(os.path.isfile(path))

        # The page images are replaced by those of another PDF: layout.json no longer belongs to them.
        process_pdf(DIGITAL, work)
        with self.assertRaises(MathInputError):
            run_document_equations(work, FakeRecognizer())
        self.assertFalse(os.path.exists(path))

    def test_a_model_that_does_not_load_fails_the_run_and_leaves_no_file(self):
        work = self.prepare_equations()
        run_document_equations(work, FakeRecognizer(ANSWERS))
        with self.assertRaises(MathModelError):
            run_document_equations(work, FakeRecognizer(fail_load=True))
        self.assertFalse(os.path.exists(os.path.join(work, "metadata", "equations.json")))

    def test_validator_reports_problems(self):
        answers = list(ANSWERS)
        answers[1], answers[2] = "x_", RuntimeError("no")
        data = self.run_equations(answers)["document"]
        self.assertEqual(validate_equations_document(data), [])
        self.assertTrue(validate_equations_document([]))
        self.assertTrue(validate_equations_document({"recognizer": {"name": "x"}, "pages": []}))

        def broken(index, change):
            copy = json.loads(json.dumps(data))
            change(copy["pages"][0]["equations"][index])
            return validate_equations_document(copy)

        self.assertTrue(broken(0, lambda e: e.update(status="ok")))
        self.assertTrue(broken(0, lambda e: e.update(latex=None)))
        self.assertTrue(broken(0, lambda e: e.update(raw_prediction=None)))
        self.assertTrue(broken(0, lambda e: e.update(confidence=1.4)))
        self.assertTrue(broken(0, lambda e: e.pop("pdf_bbox")))
        self.assertTrue(broken(0, lambda e: e.update(equation_number="(1)")))       # a number without a source
        self.assertTrue(broken(0, lambda e: e.update(page_number=2)))
        self.assertTrue(broken(1, lambda e: e.update(latex="x_")))                  # invalid LaTeX offered as LaTeX
        self.assertTrue(broken(1, lambda e: e.update(image_path=None)))             # the fallback image is required
        self.assertTrue(broken(2, lambda e: e.pop("error")))
        self.assertTrue(broken(2, lambda e: e.update(image_path=None)))


class MetricTests(unittest.TestCase):
    def test_canonical_form_ignores_spelling_only(self):
        same = (
            ("y = mx + b", "y=m x+b"),
            ("x_i^2", "x_{i}^{2}"),
            (r"L = -\sum_i y_i \log(p_i)", r"L=-{\sum_{i}}y_{i}\operatorname{l o g}(p_{i})"),
            (r"L = -\sum_i y_i \log(p_i)", r"L=-\sum_{i}y_{i}\mathrm{log}(p_{i})"),
            (r"z = \frac{x+y}{2}", r"z{=}\frac{x+y}{2}"),
            (r"\int_a^b f(x)\,dx", r"\int_{a}^{b}f(x)d x"),
            (r"\int_a^b f(x)\,dx", r"\displaystyle \int_a^b f(x) \; dx"),
            (r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}",
             r"A=\left[\begin{matrix}{a}&{b}\\ {c}&{d}\\ \end{matrix}\right]"),
            (r"\begin{pmatrix} a \\ b \end{pmatrix}", r"\left( \begin{matrix} a \\ b \end{matrix} \right)"),
            (r"\alpha x", r"\alpha  x"),
            ("$$ a+b $$", "a + b"),
        )
        for expected, written in same:
            self.assertEqual(canonical_latex(expected), canonical_latex(written), (expected, written))

    def test_canonical_form(self):
        self.assertEqual(canonical_latex(r"x_i^2 + \alpha x"), r"x_{i}^{2}+\alpha x")
        self.assertEqual(canonical_latex(r"\frac{a}{b}"), r"\frac{a}{b}")           # arguments keep their braces
        self.assertEqual(canonical_latex(r"{a+b}^{2}"), r"{a+b}^{2}")               # so does the base of a script
        self.assertEqual(canonical_latex(r"{\bf x} y"), r"{\bf x}y")
        self.assertEqual(canonical_latex(r"\begin{array}{cc} a & b \end{array}"), r"\begin{array}{cc}a&b\end{array}")
        self.assertEqual(canonical_latex(r"\mathrm{d} x"), r"\mathrm{d}x")          # not an operator name

    def test_symbols_are_never_merged(self):
        different = (("x_0", "x_O"), ("x_1", "x_l"), ("10", "1O"), ("10", "l0"), (r"a \times b", "a x b"),
                     (r"a \times b", "a b"), ("a - b", "a = b"), (r"\alpha", "a"), (r"\lambda", "A"),
                     (r"\lambda A", "A A"), ("x_i^2", "x^i_2"), ("x_i^2", "x_i 2"), ("x_i", "x i"), ("x^2", "x_2"),
                     (r"\frac{x+y}{2}", r"\frac{x}{2}+y"), (r"\frac{a}{b}", r"\frac{b}{a}"), ("a+b", "b+a"),
                     ("{a+b}^2", "a+b^2"), (r"\sqrt{x^2+y^2}", r"\sqrt{x^2}+y^2"), ("a &= b", "a =& b"),
                     (r"\begin{bmatrix} a & b \end{bmatrix}", r"\begin{pmatrix} a & b \end{pmatrix}"))
        for expected, read in different:
            self.assertNotEqual(canonical_latex(expected), canonical_latex(read), (expected, read))
            self.assertGreater(latex_cer(expected, read), 0.0)

    def test_character_error_rate(self):
        self.assertEqual(latex_cer(r"\alpha + \beta = \lambda", r"\alpha+\beta=\lambda"), 0.0)
        self.assertAlmostEqual(latex_cer("x_i^2", "x_j^2"), 1 / 9)         # one of the nine characters of x_{i}^{2}
        self.assertAlmostEqual(latex_cer("y = mx + b", "y = mx - b"), 1 / 6)
        self.assertEqual(latex_cer("a+b", ""), 1.0)
        self.assertGreater(latex_cer("a", "a+b+c+d"), 1.0)
        with self.assertRaises(ValueError):
            latex_cer("  ", "x")

    def test_one_equation(self):
        expected = {"latex": "E = mc^2", "equation_number": "(4)"}
        exact = evaluate_equation(expected, {"status": "success", "latex": "E=m c^{2}", "equation_number": "(4)"})
        self.assertEqual((exact["recognized"], exact["exact_match"], exact["strict_match"], exact["cer"],
                          exact["normalized_latex_accuracy"], exact["number_match"]), (True, True, False, 0.0, 1.0, True))
        self.assertTrue(evaluate_equation(expected, {"status": "success", "latex": "E = m c^2",
                                                     "equation_number": "(4)"})["strict_match"])

        wrong = evaluate_equation(expected, {"status": "success", "latex": "E=m c^{3}", "equation_number": None})
        self.assertEqual((wrong["recognized"], wrong["exact_match"], wrong["edits"], wrong["number_match"]),
                         (True, False, 1, False))
        self.assertAlmostEqual(wrong["normalized_latex_accuracy"], 1 - 1 / 8)      # E=mc^{2}: eight characters

        # Rejected LaTeX is compared but is not a recognition; a failure or a missing equation scores zero.
        invalid = evaluate_equation(expected, {"status": "invalid_latex", "latex": None, "normalized_latex": "E=m c^{2",
                                               "equation_number": "(4)"})
        self.assertEqual((invalid["recognized"], invalid["exact_match"], invalid["edits"]), (False, False, 1))
        for missing in (None, {"status": "recognition_failed", "latex": None, "normalized_latex": None,
                               "equation_number": None}):
            result = evaluate_equation(expected, missing)
            self.assertEqual((result["recognized"], result["exact_match"], result["cer"],
                              result["normalized_latex_accuracy"], result["number_match"]), (False, False, 1.0, 0.0, False))
        self.assertIsNone(evaluate_equation({"latex": "x"}, None)["number_match"])
        self.assertTrue(evaluate_equation({"latex": "x", "equation_number": None},
                                          {"status": "success", "latex": "x", "equation_number": None})["number_match"])

    def test_document_evaluation(self):
        def equation(page, latex, number=None, status="success", box=(100, 100, 200, 120)):
            return {"equation_id": f"p{page:03d}_e001", "page_number": page, "pdf_bbox": list(box), "status": status,
                    "latex": latex if status == "success" else None, "normalized_latex": latex,
                    "equation_number": number}

        truth = [{"page_number": 1, "kind": "a", "pdf_bbox": [100, 100, 200, 120], "latex": "x_i^2", "equation_number": "(1)"},
                 {"page_number": 2, "kind": "b", "pdf_bbox": [100, 100, 200, 120], "latex": "a+b", "equation_number": None},
                 {"page_number": 3, "kind": "c", "pdf_bbox": [100, 100, 200, 120], "latex": "a+b", "equation_number": "(2)"},
                 {"page_number": 4, "kind": "d", "pdf_bbox": [100, 100, 200, 120], "latex": "a+b", "equation_number": None}]
        document = {"pages": [
            {"page_number": 1, "equations": [equation(1, "x_{i}^{2}", "(1)", box=(101, 99, 199, 121))]},
            {"page_number": 2, "equations": [equation(2, "a-b")]},
            {"page_number": 3, "equations": [equation(3, "a+b}", None, status="invalid_latex")]},
            {"page_number": 4, "equations": [equation(4, "a+b", box=(400, 400, 500, 420))]}]}       # elsewhere

        result = evaluate_equations(truth, document)

        self.assertEqual((result["equations"], result["found"], result["recognized"]), (4, 3, 2))
        self.assertEqual(result["recognition_success_rate"], 0.5)
        self.assertEqual((result["exact_matches"], result["exact_match_rate"], result["strict_matches"]), (1, 0.25, 0))
        self.assertEqual([row["edits"] for row in result["rows"]], [0, 1, 1, 3])
        self.assertEqual((result["characters"], result["edits"]), (18, 5))
        self.assertAlmostEqual(result["cer"], 5 / 18)
        self.assertAlmostEqual(result["normalized_latex_accuracy"], (1 + 2 / 3 + 2 / 3 + 0) / 4)
        self.assertEqual((result["numbers_expected"], result["numbers_correct"]), (2, 1))
        self.assertAlmostEqual(result["number_accuracy"], 2 / 4)
        self.assertEqual([row["kind"] for row in result["rows"]], ["a", "b", "c", "d"])

    def test_the_model_style_answers_score_perfectly_on_the_fixture(self):
        # The metric is not the limit: what the model writes for a correctly read formula is an exact match.
        for kind, written in MODEL_STYLE.items():
            self.assertEqual(canonical_latex(written), canonical_latex(BY_KIND[kind]["latex"]), kind)


class FixtureTests(unittest.TestCase):
    def test_fixture_covers_the_required_kinds_in_several_styles(self):
        self.assertEqual({e["kind"] for e in TRUTH}, set(MODEL_STYLE))
        self.assertEqual(len(TRUTH), 12)
        self.assertEqual({e["font"] for e in TRUTH}, {"cm", "stix", "dejavusans"})
        self.assertEqual([e["equation_number"] for e in TRUTH if e["equation_number"]],
                         ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)"])
        for equation in TRUTH:
            self.assertEqual(validate_latex(equation["latex"]), [], equation["kind"])


class EquationToolErrorTests(TempDirTestCase):
    """Failures that are reported before any model is needed."""

    def test_bad_input_exits_with_code_2_and_no_traceback(self):
        tool = os.path.join(ENGINE, "tools", "inspect_equations.py")
        os.makedirs(self.path("empty-dir"))
        for args in ([self.path("missing.pdf")], [self.path("empty-dir")],
                     [EQUATIONS_PDF, "--output", self.path("out"), "--equation-padding", "-1"],
                     [EQUATIONS_PDF, "--output", self.path("out"), "--layout-threshold", "7"]):
            with self.subTest(args=args):
                done = subprocess.run([sys.executable, tool, *args], capture_output=True, text=True, timeout=300)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn("Error:", done.stderr)
                self.assertNotIn("Traceback", done.stderr)

    def test_evaluation_tool_reports_unreadable_input(self):
        tool = os.path.join(ENGINE, "tools", "evaluate_equations.py")
        done = subprocess.run([sys.executable, tool, self.path("missing"), "--truth", TRUTH_PATH],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 2)
        self.assertIn("Error:", done.stderr)


@unittest.skipUnless(HAVE_MATH_MODEL, "paddleocr, tokenizers or ftfy is not installed, or a model test is switched off")
class MathModelIntegrationTests(unittest.TestCase):
    """The real layout, formula and OCR models on the fixtures. Every model is loaded once.

    The thresholds are engineering thresholds for these twelve generated formulas, set below what the
    model reaches on them (digital: 12 recognized, 10 exact; scanned: 11 recognized, 10 exact). They
    are not a statement about accuracy on real papers.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="airpa-p6-model-")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        detector = PPDocLayoutV3Detector(device="cpu")
        cls.recognizer = PPFormulaNetRecognizer(device="cpu")
        cls.results = {}
        for name, pdf in (("digital", EQUATIONS_PDF), ("scanned", EQUATIONS_SCANNED), ("paper", DIGITAL)):
            work = os.path.join(cls.tmp, name)
            process_pdf(pdf, work)
            detect_document_layout(work, detector)
            if name == "scanned":       # equation numbers of scanned pages come from the Phase 4 OCR text
                run_document_ocr(work, PaddleOCREngine(device="cpu"))
            cls.results[name] = run_document_equations(work, cls.recognizer, save_crops=(name == "digital"))
        cls.metrics = {name: evaluate_equations(TRUTH, cls.results[name]["document"]) for name in ("digital", "scanned")}

    def row(self, name, kind):
        return next(row for row in self.metrics[name]["rows"] if row["kind"] == kind)

    def test_model_loads_once(self):
        self.assertEqual(self.recognizer.load_count, 1)
        self.assertGreater(self.recognizer.load_seconds, 0.0)

    def test_every_document_gives_a_valid_equations_json(self):
        for name, result in self.results.items():
            with self.subTest(document=name):
                with open(result["equations_path"], encoding="utf-8") as fh:
                    data = json.load(fh)
                self.assertEqual(validate_equations_document(data), [])
                self.assertEqual((data["recognizer"]["name"], data["recognizer"]["device"]), ("PP-FormulaNet_plus-S", "cpu"))
                for equation in all_equations(data):
                    self.assertIsNone(equation["confidence"])               # the model reports none
                    self.assertEqual(equation["source_model"], "PP-FormulaNet_plus-S")

    def test_layout_finds_every_fixture_equation(self):
        for name in ("digital", "scanned"):
            self.assertEqual((self.metrics[name]["found"], self.metrics[name]["equations"]), (12, 12), name)
            self.assertEqual(self.results[name]["document"]["summary"]["equation_regions"], 12)

    def test_digital_equations(self):
        metrics = self.metrics["digital"]
        self.assertGreaterEqual(metrics["recognized"], 11)
        self.assertGreaterEqual(metrics["exact_matches"], 9)
        self.assertLessEqual(metrics["cer"], 0.10)
        self.assertGreaterEqual(metrics["normalized_latex_accuracy"], 0.93)

    def test_scanned_equations(self):
        metrics = self.metrics["scanned"]
        self.assertGreaterEqual(metrics["recognized"], 10)
        self.assertGreaterEqual(metrics["exact_matches"], 8)
        self.assertLessEqual(metrics["cer"], 0.15)
        self.assertGreaterEqual(metrics["normalized_latex_accuracy"], 0.90)

    def test_formulas_read_exactly_in_both_documents(self):
        for name in ("digital", "scanned"):
            for kind in ("simple_algebra", "fraction", "square_root", "greek", "integral", "numbered", "matrix",
                         "symbol_fidelity_digits", "symbol_fidelity_greek"):
                row = self.row(name, kind)
                self.assertTrue(row["recognized"] and row["exact_match"], (name, kind, row["predicted"]))

    def test_summation(self):
        row = self.row("digital", "summation")
        self.assertTrue(row["exact_match"], row["predicted"])
        self.assertEqual(row["predicted"], r"L=-\sum_{i}y_{i}\log(p_{i})")

    def test_subscript_and_superscript(self):
        for name in ("digital", "scanned"):
            row = self.row(name, "subscript_superscript")
            self.assertRegex(row["predicted"], r"^x_\{.\}\^\{2\}$")         # the structure is right in both
            self.assertLessEqual(row["cer"], 0.12)

    def test_symbol_fidelity(self):
        for name in ("digital", "scanned"):
            self.assertEqual(self.row(name, "symbol_fidelity_digits")["predicted"], r"x_{0}\times l_{1}=10")
            self.assertEqual(self.row(name, "symbol_fidelity_greek")["predicted"], r"\lambda A=a-\alpha")
            self.assertEqual(self.row(name, "greek")["predicted"], r"\alpha+\beta=\lambda")

    def test_matrix_and_multiline_equations_are_one_block_each(self):
        for name in ("digital", "scanned"):
            matrix = self.row(name, "matrix")["predicted"]
            self.assertEqual(matrix, r"A=\begin{bmatrix}a&b\\c&d\end{bmatrix}")
            aligned = self.row(name, "multiline_aligned")
            self.assertTrue(aligned["recognized"])
            self.assertRegex(aligned["predicted"], r"^\\begin\{aligned\}.*\\\\.*\\end\{aligned\}$")
            self.assertEqual(aligned["predicted"].count(r"\\"), 1)
            self.assertLessEqual(aligned["cer"], 0.25)

    def test_equation_numbers(self):
        digital = all_equations(self.results["digital"]["document"])
        self.assertEqual((self.metrics["digital"]["numbers_correct"], self.metrics["digital"]["number_accuracy"]), (6, 1.0))
        self.assertEqual({e["equation_number_source"] for e in digital if e["equation_number"]}, {"native"})
        numbered = next(e for e in digital if e["equation_number"] == "(4)")
        self.assertEqual(canonical_latex(numbered["latex"]), "E=mc^{2}")

        scanned = all_equations(self.results["scanned"]["document"])
        self.assertGreaterEqual(self.metrics["scanned"]["numbers_correct"], 5)
        self.assertEqual({e["equation_number_source"] for e in scanned if e["equation_number"]}, {"ocr"})

    def test_raw_and_normalized_predictions_are_kept(self):
        for name in ("digital", "scanned"):
            for equation in all_equations(self.results[name]["document"]):
                self.assertIsInstance(equation["raw_prediction"], str)
                self.assertEqual(equation["normalized_latex"], normalize_latex(equation["raw_prediction"]))
                if equation["status"] == "success":
                    self.assertEqual(equation["latex"], equation["normalized_latex"])
                else:       # rejected: no LaTeX is offered, the crop is kept
                    self.assertIsNone(equation["latex"])
                    self.assertTrue(os.path.isfile(os.path.join(self.results[name]["work_dir"],
                                                                *equation["image_path"].split("/"))))

    def test_crops(self):
        self.assertEqual(len(self.results["digital"]["crops"]), 12)
        kept = [e for e in all_equations(self.results["scanned"]["document"]) if e["status"] != "success"]
        self.assertEqual(len(self.results["scanned"]["crops"]), len(kept))

    def test_document_without_equations(self):
        summary = self.results["paper"]["document"]["summary"]
        self.assertEqual((summary["equation_regions"], summary["recognized"]), (0, 0))

    def test_timings(self):
        timings = self.results["digital"]["timings"]
        self.assertEqual(len(timings["equation_s"]), 12)
        self.assertGreater(timings["model_init_s"], 0.0)                    # loaded during the first document
        self.assertEqual(self.results["scanned"]["timings"]["model_init_s"], 0.0)


@unittest.skipUnless(HAVE_MATH_MODEL, "paddleocr, tokenizers or ftfy is not installed, or a model test is switched off")
class InspectEquationsToolIntegrationTests(TempDirTestCase):
    TOOL = os.path.join(ENGINE, "tools", "inspect_equations.py")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, self.TOOL, *args], capture_output=True, text=True, timeout=900)

    def test_pdf_input_then_reuse_of_the_working_directory(self):
        done = self.run_tool(EQUATIONS_PDF, "--output", self.path("out"), "--save-equation-crops")

        self.assertEqual(done.returncode, 0, done.stderr)
        for expected in ("Pages: 4", "Equation regions: 12", "Recognized: ", "Failed: ", "With equation numbers: 6",
                         "Average confidence: n/a", "Total time: ", "Recognizer: PP-FormulaNet_plus-S, device CPU, "
                         "loaded 1 time(s)", "Phase 2 pages: created; Phase 3 layout: created; Phase 4 OCR: not available"):
            self.assertIn(expected, done.stdout)
        self.assertRegex(done.stdout, r"Time: model init \d+\.\d\ds, inference \d+\.\d\ds \(\d+\.\d\ds/equation")
        self.assertEqual(sorted(os.listdir(self.path("out", "metadata"))), ["equations.json", "layout.json", "pages.json"])
        self.assertEqual(len(os.listdir(self.path("out", "equation_crops"))), 12)
        with open(self.path("out", "metadata", "equations.json"), encoding="utf-8") as fh:
            first = json.load(fh)
        self.assertEqual(validate_equations_document(first), [])

        earlier = {name: sha256(self.path("out", "metadata", name)) for name in ("pages.json", "layout.json")}
        done = self.run_tool(self.path("out"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Phase 2 pages: reused; Phase 3 layout: reused", done.stdout)
        self.assertEqual(earlier, {name: sha256(self.path("out", "metadata", name)) for name in earlier})
        with open(self.path("out", "metadata", "equations.json"), encoding="utf-8") as fh:
            again = json.load(fh)
        self.assertEqual([e["latex"] for e in all_equations(again)], [e["latex"] for e in all_equations(first)])
        self.assertEqual(os.listdir(self.path("out", "equation_crops")), [])       # not requested this time

        evaluated = subprocess.run([sys.executable, os.path.join(ENGINE, "tools", "evaluate_equations.py"),
                                    self.path("out"), "--truth", TRUTH_PATH], capture_output=True, text=True, timeout=120)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("Equations: 12 of 12 found", evaluated.stdout)
        self.assertRegex(evaluated.stdout, r"Exact LaTeX match: \d\.\d{3} \(\d+/12\)")
        self.assertIn("Equation numbers: 6/6 found, accuracy 1.000", evaluated.stdout)


if __name__ == "__main__":
    unittest.main()

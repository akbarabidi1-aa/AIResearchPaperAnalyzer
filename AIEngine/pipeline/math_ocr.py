"""Phase 6 equation extraction: EQUATION region -> crop -> image-to-LaTeX -> checked LaTeX -> equations.json.

    Phase 3 EQUATION region
        ->  crop of the Phase 2 page image (region + padding, without the equation number)
        ->  MathRecognizer        the image-to-LaTeX model; raw prediction, confidence if the model has one
        ->  normalize_latex       whitespace and wrappers only
        ->  validate_latex        braces, environments, scripts, control sequences
        ->  equation number       the "(4)" beside the equation, from native text or Phase 4 OCR text
        ->  <work_dir>/metadata/equations.json

Only the recognizer class knows the model library. Nothing here rewrites a symbol or simplifies a
formula: the model's answer is kept as `raw_prediction`, and an equation that cannot be recognized
keeps its region and a saved crop.

Equations only: no figure or chart understanding, no reading order, no summarization.
"""
import glob
import math
import os
import re
import time
from dataclasses import dataclass

from .common import OutputDirectoryError, PdfPipelineError, ensure_directory
from .layout import COORDINATE_SYSTEM, LAYOUT_JSON, RegionType, _package_version, load_pages_document
from .layout_eval import is_valid_bbox
from .ocr import (
    DEFAULT_OCR_PADDING,
    MIN_CROP_SIDE,
    OCR_JSON,
    TEXT_SOURCE_OCR,
    _is_confidence,
    _open_page_image,
    check_padding,
    clean_line,
    crop_box,
    is_blank,
    load_layout_document,
)
from .page_pipeline import METADATA_DIR, PAGES_JSON, _write_json
from .render import pdf_bbox_to_pixels
from .tables import _remove_stale, load_ocr_document

EQUATIONS_SCHEMA_VERSION = 1
EQUATIONS_JSON = "equations.json"
EQUATION_CROPS_DIR = "equation_crops"
EQUATION_CROP_PATTERN = "page_{:03d}_eq_{:03d}.png"

DEFAULT_EQUATION_PADDING = DEFAULT_OCR_PADDING      # pixels of page added around an equation region
NUMBER_MARGIN = 2           # pixels kept free between the crop and an equation number cut off from it

STATUS_SUCCESS = "success"
STATUS_INVALID = "invalid_latex"                    # the model answered, the answer is not sound LaTeX
STATUS_FAILED = "recognition_failed"                # no answer: model error, empty answer, unusable crop

NUMBER_SOURCE_NATIVE = "native"
NUMBER_SOURCE_OCR = "ocr"
NUMBER_SOURCE_TAG = "latex_tag"

# A whole text line that is an equation number: "(4)", "(3.2)", "(12a)", "(A.1)", "(B3)".
_EQUATION_NUMBER = re.compile(r"\(\s*((?:[A-Za-z]{1,2}\.?)?\d{1,3}(?:[.\-]\d{1,3}){0,2}[a-z]?)\s*\)")

# Wrappers around a whole formula. They say "this is math", which the EQUATION region already says.
_WRAPPERS = (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)"),
             (r"\begin{equation}", r"\end{equation}"), (r"\begin{equation*}", r"\end{equation*}"),
             (r"\begin{displaymath}", r"\end{displaymath}"), (r"\begin{math}", r"\end{math}"))
_SPECIAL_TOKENS = ("<s>", "</s>", "<pad>", "<unk>", "[BOS]", "[EOS]", "[PAD]")
_TAG = re.compile(r"\\tag(\*?)\s*\{([^{}]*)\}\s*$")

_TOKEN = re.compile(r"\\[A-Za-z]+\*?|\\.|\\$|\S", re.DOTALL)
_ENVIRONMENT = re.compile(r"\\(begin|end)\s*\{([^{}]*)\}")
# Groups whose content is words, not math: letters in them are not checked for missing backslashes.
_TEXT_GROUP = re.compile(r"\\(?:text|textrm|textbf|textit|mathrm|mathit|mathbf|mathsf|mathtt|operatorname\*?|mbox)"
                         r"\s*\{[^{}]*\}")
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+\*?")
_LETTER_RUN = re.compile(r"[A-Za-z]{3,}")
# Command names that are never a product of variables: seen without a backslash, the backslash was lost.
_BARE_COMMANDS = frozenset({
    "frac", "sqrt", "sum", "prod", "int", "lim", "begin", "end", "left", "right", "alpha", "beta", "gamma",
    "delta", "epsilon", "theta", "lambda", "sigma", "omega", "infty", "partial", "nabla", "cdot", "times",
    "mathrm", "mathbf", "mathcal", "mathbb", "operatorname", "quad", "qquad", "array", "matrix", "aligned",
})
_CONTROL_SYMBOLS = frozenset(" ,;:!\\{}|%$#&_>.'\"`^~=-/+()[]@*<")     # what may follow a backslash besides a letter
MAX_CONTROL_WORD = 24       # letters; a longer "command" is run-together garbage
MAX_TOKEN_REPEAT = 16       # the same token this often in a row is a decoder stuck in a loop


class MathError(PdfPipelineError):
    """Base class for equation extraction failures."""


class MathInputError(MathError):
    """The Phase 2 / Phase 3 artifacts are missing, invalid or do not belong together."""


class MathModelError(MathError):
    """The math recognition model could not be loaded."""


class MathRecognitionError(MathError):
    """One equation could not be recognized. The run goes on with the other equations."""


@dataclass(frozen=True)
class MathPrediction:
    """What a model returned for one image: its LaTeX exactly as returned, and its own confidence
    (0-1), or None when the model does not report one."""
    latex: str
    confidence: float = None


class MathRecognizer:
    """Interface of an image-to-LaTeX model. Subclasses set `name` and implement `_load_model` and
    `_recognize`; nothing outside the subclass touches the model library."""

    name = ""

    def __init__(self):
        self._model = None
        self.load_count = 0         # how many times the model was really loaded
        self.load_seconds = 0.0

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        """Load the model once; later calls do nothing. Raises MathModelError."""
        if self.loaded:
            return self
        started = time.perf_counter()
        try:
            self._model = self._load_model()
        except MathError:
            raise
        except Exception as exc:
            raise MathModelError(f"Cannot load math recognizer {self.name}: {exc}") from exc
        self.load_count += 1
        self.load_seconds = time.perf_counter() - started
        return self

    def recognize(self, image):
        """LaTeX of the formula shown by a PIL image, as a MathPrediction. The text is the model's
        own; a confidence that is not a number between 0 and 1 becomes None.
        Raises MathRecognitionError when the model fails or does not return text."""
        self.load()
        try:
            prediction = self._recognize(image)
        except MathError:
            raise
        except Exception as exc:
            raise MathRecognitionError(f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(prediction, MathPrediction) or not isinstance(prediction.latex, str):
            raise MathRecognitionError("the model returned no text")
        confidence = float(prediction.confidence) if _is_confidence(prediction.confidence) else None
        return MathPrediction(prediction.latex, confidence)

    def recognize_region(self, page_image, image_bbox, padding=DEFAULT_EQUATION_PADDING, exclude=()):
        """Recognize one region of a page image (PIL).

        Returns {"prediction": MathPrediction or None, "crop_bbox": [x0, y0, x1, y1], "crop": PIL image,
        "blank": bool}. A crop without contrast is not sent to the model ("blank": True, no prediction).
        `exclude` are boxes of equation numbers to keep out of the crop (see equation_crop_box).
        Raises MathRecognitionError for a region that cannot be cut out or recognized.
        """
        crop, box = crop_equation(page_image, image_bbox, padding, exclude)
        blank = is_blank(crop)
        return {"prediction": None if blank else self.recognize(crop), "crop_bbox": list(box), "crop": crop,
                "blank": blank}

    def describe(self):
        """The "recognizer" object of equations.json."""
        return {"name": self.name}

    def _load_model(self):
        raise NotImplementedError

    def _recognize(self, image):
        """MathPrediction for a PIL RGB image."""
        raise NotImplementedError


class PPFormulaNetRecognizer(MathRecognizer):
    """PP-FormulaNet_plus-S through PaddleOCR's standalone FormulaRecognition module.

    The model resizes and normalizes the image itself; it gets the crop as it is. It reports no
    confidence, so every prediction has confidence None. Pass another `model_name` of the same family
    (for example "PP-FormulaNet_plus-M") for a larger model.
    """

    name = "PP-FormulaNet_plus-S"

    def __init__(self, device="cpu", model_name=None):
        super().__init__()
        self.device = device
        if model_name:
            self.name = model_name

    def describe(self):
        return {
            "name": self.name,
            "provider": "PaddleOCR",
            "version": _package_version("paddleocr"),
            "framework": "paddlepaddle " + _package_version("paddlepaddle"),
            "device": self.device,
        }

    def _load_model(self):
        # Same reasons as in PPDocLayoutV3Detector._load_model.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import FormulaRecognition
        return FormulaRecognition(model_name=self.name, device=self.device)

    def _recognize(self, image):
        import numpy as np
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])     # BGR, as OpenCV
        result = list(self._model.predict(pixels, batch_size=1))[0]
        return MathPrediction(result["rec_formula"], None)


def equation_crop_box(image_bbox, image_width, image_height, padding=DEFAULT_EQUATION_PADDING, exclude=()):
    """Whole-pixel box (x0, y0, x1, y1) to cut out for an equation region: the region grown by
    `padding` pixels (see ocr.crop_box), then cut on the right just before any box of `exclude` that
    reaches into it. `exclude` holds the boxes of equation numbers in page pixels: a number is not part
    of the formula. A cut that would leave less than MIN_CROP_SIDE pixels is not made.

    Raises MathRecognitionError for a region that cannot be cut out, ValueError for a bad `padding`.
    """
    try:
        x0, y0, x1, y1 = crop_box(image_bbox, image_width, image_height, padding)
    except PdfPipelineError as exc:
        raise MathRecognitionError(str(exc)) from exc
    for other in exclude:
        if other[0] < x1 and other[2] > x0 and other[1] < y1 and other[3] > y0:
            cut = math.floor(other[0]) - NUMBER_MARGIN
            if cut - x0 >= MIN_CROP_SIDE:
                x1 = min(x1, cut)
    return (x0, y0, x1, y1)


def crop_equation(page_image, image_bbox, padding=DEFAULT_EQUATION_PADDING, exclude=()):
    """Cut an equation region out of a page image (PIL). Returns (crop, box): the pixels are those of
    the page image, unchanged, and `box` says where they came from. Raises MathRecognitionError."""
    box = equation_crop_box(image_bbox, page_image.width, page_image.height, padding, exclude)
    return page_image.crop(box), box


def normalize_latex(text):
    """Safe cleanup of a predicted formula. Nothing that changes its meaning:

      - outer whitespace removed, inner runs of whitespace (line breaks included) -> one space;
      - special tokens of a model (<s>, </s>, <pad>, ...) removed from both ends;
      - one pair of delimiters around the WHOLE formula removed: $...$, $$...$$, \\[...\\], \\(...\\),
        \\begin{equation}...\\end{equation} (also equation*, displaymath, math).

    Symbols, braces, commands and their order are never touched.
    """
    text = " ".join(text.split())
    changed = True
    while changed:
        changed = False
        for token in _SPECIAL_TOKENS:
            if text.startswith(token) or text.endswith(token):
                text = text[len(token):] if text.startswith(token) else text[:-len(token)]
                text, changed = text.strip(), True
    for opening, closing in _WRAPPERS:
        inner = text[len(opening):len(text) - len(closing)]
        # "$a$ + $b$" is not one wrapped formula: the closing delimiter must not occur inside.
        if (len(text) >= len(opening) + len(closing) and text.startswith(opening) and text.endswith(closing)
                and closing not in inner and opening not in inner):
            return inner.strip()
    return text


def split_tag(latex):
    """Separate an equation number the model wrote as \\tag{...} at the end of a formula.
    Returns (formula without the tag, number or None): "E = mc^2 \\tag{4}" -> ("E = mc^2", "(4)").
    \\tag*{...} gives the number without parentheses. A formula with several tags is left as it is."""
    match = _TAG.search(latex)
    if not match or latex.count(r"\tag") != 1 or not match.group(2).strip():
        return latex, None
    number = match.group(2).strip()
    return latex[:match.start()].rstrip(), number if match.group(1) else f"({number})"


def validate_latex(latex):
    """Lightweight check of a formula. Returns a list of problems (empty = no problem found).

    Checked: not empty; braces balanced; \\begin / \\end paired and nested; \\left / \\right paired;
    no subscript or superscript without an argument; no backslash followed by nothing, by a digit or by
    a character that makes no command; no "command" of more than MAX_CONTROL_WORD letters; no known
    command name without its backslash ("frac", "sum"); no token repeated MAX_TOKEN_REPEAT times in a row.

    This is not a TeX run: a formula that passes can still fail to compile (unknown but well-formed
    commands are accepted), and it says nothing about whether the formula is the one in the image.
    """
    if not isinstance(latex, str) or not latex.strip():
        return ["the formula is empty"]
    problems, tokens = [], _TOKEN.findall(latex)

    depth = 0
    for token in tokens:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                problems.append("unbalanced braces: } without {")
                break
    if depth > 0:
        problems.append(f"unbalanced braces: {depth} {{ not closed")

    stack = []
    for kind, name in _ENVIRONMENT.findall(latex):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            problems.append(f"\\end{{{name}}} does not close a \\begin{{{name}}}")
            stack = []
            break
    if stack:
        problems.append(f"\\begin{{{stack[-1]}}} is not closed")

    sized = 0
    for token in tokens:
        sized += (token == r"\left") - (token == r"\right")
        if sized < 0:
            break
    if sized:
        problems.append("\\left and \\right are not paired")

    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if token in "_^" and (following is None or following in ("}", "_", "^", "&", r"\\", r"\right", r"\end")):
            problems.append(f"{token} has no argument")
            break
    for token in tokens:
        if token == "\\" or (len(token) == 2 and token[0] == "\\" and token[1] not in _CONTROL_SYMBOLS
                             and not (token[1].isascii() and token[1].isalpha())):
            problems.append(f"invalid control sequence: {token!r}")
            break
        if token[0] == "\\" and len(token) > MAX_CONTROL_WORD + 1:
            problems.append(f"invalid control sequence: {token[:MAX_CONTROL_WORD]}...")
            break

    bare = _CONTROL_WORD.sub(" ", _ENVIRONMENT.sub(" ", _TEXT_GROUP.sub(" ", latex)))
    missing = sorted({run for run in _LETTER_RUN.findall(bare) if run in _BARE_COMMANDS})
    if missing:
        problems.append("command without a backslash: " + ", ".join(missing))

    run = 1
    for previous, token in zip(tokens, tokens[1:]):
        run = run + 1 if token == previous else 1
        if run >= MAX_TOKEN_REPEAT:
            problems.append(f"{token!r} is repeated {MAX_TOKEN_REPEAT} times or more in a row")
            break
    return problems


def parse_equation_number(text):
    """The equation number a text line consists of, or None: "(4)" and " ( 4 ) " -> "(4)"; also
    "(3.2)", "(12a)", "(A.1)". A line that holds anything else ("Eq. (4)", "f(2)", "[4]", "4") is None."""
    match = _EQUATION_NUMBER.fullmatch(clean_line(text or ""))
    return f"({match.group(1)})" if match else None


def find_equation_numbers(equation_boxes, text_lines):
    """Pair equations with the equation numbers beside them, by geometry only.

    `equation_boxes` are the boxes of the equations of one page and `text_lines` its text lines as
    (box, text, source), all in one coordinate system. A line is a candidate when it is nothing but a
    parenthesized number (parse_equation_number). It can belong to an equation when its vertical
    centre lies within the equation's height and its horizontal centre is right of the equation's:
    numbers set on the right, beside the equation or in the right part of its box. A candidate goes to
    the nearest such equation (smallest horizontal gap). An equation with exactly one candidate gets
    its number; with none, or with several, it gets nothing.

    Returns one entry per equation: None, or {"number", "source", "bbox"}.
    """
    claims = [[] for _ in equation_boxes]
    for bbox, text, source in text_lines:
        number = parse_equation_number(text)
        if number is None:
            continue
        centre_x, centre_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        nearest = sorted((max(bbox[0] - box[2], 0.0), index) for index, box in enumerate(equation_boxes)
                         if box[1] <= centre_y <= box[3] and centre_x > (box[0] + box[2]) / 2)
        if nearest and (len(nearest) == 1 or nearest[0][0] < nearest[1][0]):
            claims[nearest[0][1]].append({"number": number, "source": source, "bbox": list(bbox)})
    return [found[0] if len(found) == 1 else None for found in claims]


def page_text_lines(page, ocr_page=None):
    """Text lines of a page as (box in PDF points, text, source) for find_equation_numbers: the native
    lines of pages.json, and the lines Phase 4 recognized by OCR when `ocr_page` (the page of ocr.json)
    is given."""
    lines = [(tuple(line["bbox"]), line["text"], NUMBER_SOURCE_NATIVE)
             for block in page["blocks"] for line in block["lines"]]
    for region in (ocr_page or {}).get("regions", []):
        if region["text_source"] == TEXT_SOURCE_OCR:
            lines += [(tuple(line["pdf_bbox"]), line["text"], NUMBER_SOURCE_OCR) for line in region["lines"]]
    return lines


def recognize_equation(recognizer, crop):
    """Recognize one crop and check the answer. Returns the fields of an equation entry that come
    from the model: latex, normalized_latex, raw_prediction, confidence, status, problems, tag_number.

    `latex` is only set for status "success". `raw_prediction` is always the model's own text and
    `normalized_latex` its cleaned form, also when the check fails. Raises MathRecognitionError.
    """
    prediction = recognizer.recognize(crop)
    normalized, tag_number = split_tag(normalize_latex(prediction.latex))
    if not normalized:
        raise MathRecognitionError("the model returned an empty formula")
    problems = validate_latex(normalized)
    return {
        "latex": None if problems else normalized,
        "normalized_latex": normalized,
        "raw_prediction": prediction.latex,
        "confidence": prediction.confidence,
        "status": STATUS_INVALID if problems else STATUS_SUCCESS,
        "problems": problems,
        "tag_number": tag_number,
    }


def run_document_equations(work_dir, recognizer, padding=DEFAULT_EQUATION_PADDING, save_crops=False,
                           use_ocr_text=True):
    """Recognize every EQUATION region of a working directory and write metadata/equations.json.

    Needs the Phase 2 and Phase 3 artifacts in `work_dir`; neither is modified, and neither are
    ocr.json and tables.json. Only EQUATION regions of layout.json are looked at; pages are not
    searched for formulas. The recognizer is loaded once, and only if there is an equation. Each
    region is handled on its own: one that cannot be recognized is kept with status
    "recognition_failed", `latex` null and an "error", and the run goes on.

    Equation numbers come from the native text lines of pages.json and, with use_ocr_text=True, from
    the OCR lines of ocr.json when the directory has one that belongs to this layout.

    The crop sent to the model is saved as <work_dir>/equation_crops/page_NNN_eq_NNN.png for every
    equation whose status is not "success" (so it stays available for review), and for all equations
    with save_crops=True. `image_path` of an equation is that file, relative to `work_dir`, or null.

    Returns {"equations_path", "work_dir", "document", "timings", "crops"}. `timings` has model_init_s
    (0.0 if the recognizer was already loaded or not needed), equation_s ({equation_id: seconds of
    inference}), inference_s and total_s; it is not written to the file.
    Raises a PdfPipelineError for unusable input or a model that does not load (ValueError for a bad
    `padding`); no equations.json is left behind then.
    """
    started = time.perf_counter()
    check_padding(padding)
    work_dir = os.path.abspath(os.fspath(work_dir))
    equations_path = os.path.join(work_dir, METADATA_DIR, EQUATIONS_JSON)
    _remove_stale(equations_path)       # first: an equations.json of other inputs must not survive a failed run
    pages_document = load_pages_document(work_dir)
    try:
        layout_document = load_layout_document(work_dir, pages_document)
    except PdfPipelineError as exc:
        raise MathInputError(str(exc)) from exc
    if any("lines" not in block for page in pages_document["pages"] for block in page["blocks"]):
        raise MathInputError("pages.json has no native text lines (written before Phase 4); run Phase 2 again")
    ocr_pages = None
    if use_ocr_text:
        try:
            ocr_pages = load_ocr_document(work_dir, layout_document)["pages"]
        except PdfPipelineError:
            ocr_pages = None            # no Phase 4 result for this layout: native text only
    _remove_old_crops(work_dir)

    def is_equation(region):
        return region["type"] == RegionType.EQUATION.value

    model_init_s = 0.0
    if any(is_equation(region) for page in layout_document["pages"] for region in page["regions"]):
        was_loaded = recognizer.loaded
        recognizer.load()
        model_init_s = 0.0 if was_loaded else recognizer.load_seconds

    pages, equation_seconds, crops = [], {}, []
    for index, (page, layout_page) in enumerate(zip(pages_document["pages"], layout_document["pages"])):
        regions = [region for region in layout_page["regions"] if is_equation(region)]
        numbers = find_equation_numbers([region["pdf_bbox"] for region in regions],
                                        page_text_lines(page, ocr_pages[index] if ocr_pages else None))
        page_image = _open_page_image(work_dir, page) if regions else None
        equations = []
        for position, (region, number) in enumerate(zip(regions, numbers), 1):
            equation = _equation_entry(region, position, recognizer.name)
            if number:
                equation.update(equation_number=number["number"], equation_number_source=number["source"],
                                equation_number_pdf_bbox=number["bbox"])
            exclude = [pdf_bbox_to_pixels(number["bbox"], page["scale_x"], page["scale_y"])] if number else []
            crop = None
            try:
                crop, box = crop_equation(page_image, region["image_bbox"], padding, exclude)
                equation["crop_bbox"] = list(box)
                if is_blank(crop):
                    raise MathRecognitionError("the equation region is blank")
                t0 = time.perf_counter()
                try:
                    fields = recognize_equation(recognizer, crop)
                finally:
                    equation_seconds[equation["equation_id"]] = time.perf_counter() - t0
                tag_number = fields.pop("tag_number")
                equation.update(fields)
                if tag_number and not number:
                    equation.update(equation_number=tag_number, equation_number_source=NUMBER_SOURCE_TAG)
            except Exception as exc:    # one unreadable equation must not lose the others
                equation.update(status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
            if crop is not None and (save_crops or equation["status"] != STATUS_SUCCESS):
                name = EQUATION_CROP_PATTERN.format(page["page_number"], position)
                target = os.path.join(ensure_directory(os.path.join(work_dir, EQUATION_CROPS_DIR)), name)
                try:
                    crop.save(target, format="PNG")
                except OSError as exc:
                    raise OutputDirectoryError(f"Cannot write equation crop: {target} ({exc})") from exc
                equation["image_path"] = f"{EQUATION_CROPS_DIR}/{name}"
                crops.append(target)
            equations.append(equation)
        pages.append({"page_number": page["page_number"], "equations": equations})

    document = {
        "schema_version": EQUATIONS_SCHEMA_VERSION,
        "recognizer": recognizer.describe(),
        "settings": {"padding_px": padding, "equation_numbers_from_ocr": ocr_pages is not None},
        "source": {
            "pages_json": f"{METADATA_DIR}/{PAGES_JSON}",
            "layout_json": f"{METADATA_DIR}/{LAYOUT_JSON}",
            "ocr_json": f"{METADATA_DIR}/{OCR_JSON}" if ocr_pages is not None else None,
            "source_pdf": pages_document["document"].get("source_pdf"),
            "source_sha256": pages_document["document"].get("source_sha256"),
        },
        "coordinate_system": dict(COORDINATE_SYSTEM),
        "pages": pages,
    }
    document["summary"] = summarize_equations_document(document)
    problems = validate_equations_document(document)
    if problems:    # a bug in this module, not a property of the document
        raise AssertionError("equations.json would be invalid: " + "; ".join(problems))

    _write_json(equations_path, document)
    return {
        "equations_path": equations_path,
        "work_dir": work_dir,
        "document": document,
        "timings": {
            "model_init_s": model_init_s,
            "equation_s": equation_seconds,
            "inference_s": sum(equation_seconds.values()),
            "total_s": time.perf_counter() - started,
        },
        "crops": crops,
    }


def summarize_equations_document(document):
    """Counts over all equations of an equations.json object. equation_regions = recognized +
    invalid_latex + recognition_failed. mean_confidence is the mean over the recognized equations
    that have a confidence, or None (always None for a model that reports none)."""
    equations = [equation for page in document["pages"] for equation in page["equations"]]
    confidences = [e["confidence"] for e in equations if e["status"] == STATUS_SUCCESS and e["confidence"] is not None]
    return {
        "pages": len(document["pages"]),
        "equation_regions": len(equations),
        "recognized": sum(1 for e in equations if e["status"] == STATUS_SUCCESS),
        "invalid_latex": sum(1 for e in equations if e["status"] == STATUS_INVALID),
        "recognition_failed": sum(1 for e in equations if e["status"] == STATUS_FAILED),
        "with_equation_numbers": sum(1 for e in equations if e["equation_number"] is not None),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


def validate_equations_document(document):
    """Structural check of an equations.json object. Returns a list of problems (empty = valid)."""
    if not isinstance(document, dict):
        return ["root is not an object"]
    problems = []
    recognizer = document.get("recognizer")
    if not isinstance(recognizer, dict) or not isinstance(recognizer.get("name"), str) or not recognizer.get("name"):
        problems.append("recognizer.name missing")
    pages = document.get("pages")
    if not isinstance(pages, list) or not pages:
        return problems + ["pages missing or empty"]

    for i, page in enumerate(pages):
        if (not isinstance(page, dict) or page.get("page_number") != i + 1
                or not isinstance(page.get("equations"), list)):
            problems.append(f"pages[{i}] has no page_number {i + 1} or no equations")
            continue
        for j, equation in enumerate(page["equations"]):
            where = f"pages[{i}].equations[{j}]"
            if not isinstance(equation, dict):
                problems.append(f"{where} is not an object")
                continue
            if not isinstance(equation.get("equation_id"), str) or equation.get("page_number") != page["page_number"]:
                problems.append(f"{where} has no equation_id or a wrong page_number")
            if not all(isinstance(equation.get(key), str) for key in ("source_region_id", "source_model")):
                problems.append(f"{where} has no source_region_id or no source_model")
            for key in ("image_bbox", "pdf_bbox"):
                if not is_valid_bbox(equation.get(key)):
                    problems.append(f"{where}.{key} is not a valid box")
            problems.extend(f"{where}: {problem}" for problem in _equation_problems(equation))
    return problems


def _equation_problems(equation):
    status, latex, raw = equation.get("status"), equation.get("latex"), equation.get("raw_prediction")
    normalized, confidence = equation.get("normalized_latex"), equation.get("confidence")
    number = equation.get("equation_number")
    problems = []
    if status == STATUS_SUCCESS:
        if not (isinstance(latex, str) and latex and latex == normalized and isinstance(raw, str)
                and equation.get("problems") == [] and "error" not in equation):
            problems.append("a recognized equation needs latex, normalized_latex and raw_prediction, and no problems")
    elif status == STATUS_INVALID:
        if not (latex is None and isinstance(normalized, str) and normalized and isinstance(raw, str)
                and equation.get("problems") and "error" not in equation):
            problems.append("an invalid equation needs normalized_latex, raw_prediction and problems, and latex null")
    elif status == STATUS_FAILED:
        if not (latex is None and normalized is None and raw is None and isinstance(equation.get("error"), str)):
            problems.append("a failed equation needs an error and no latex")
    else:
        problems.append("status is not success, invalid_latex or recognition_failed")
    if confidence is not None and not _is_confidence(confidence):
        problems.append("confidence is not null or between 0 and 1")
    if equation.get("crop_bbox") is not None and not is_valid_bbox(equation["crop_bbox"]):
        problems.append("crop_bbox is not a valid box")
    if (number is None) != (equation.get("equation_number_source") is None) or (number is not None and not number):
        problems.append("equation_number and equation_number_source do not fit together")
    if equation.get("image_path") is not None and not isinstance(equation["image_path"], str):
        problems.append("image_path is not text")
    if status in (STATUS_INVALID, STATUS_FAILED) and equation.get("crop_bbox") and not equation.get("image_path"):
        problems.append("an equation without latex must keep its crop image")
    return problems


def _equation_entry(region, position, model_name):
    """An equation entry before recognition: identity and geometry of its layout region, unchanged."""
    return {
        "equation_id": f"p{region['page_number']:03d}_e{position:03d}",
        "page_number": region["page_number"],
        "source_region_id": region["region_id"],
        "source_label": region["source_label"],
        "source_model": model_name,
        "image_bbox": list(region["image_bbox"]),
        "pdf_bbox": list(region["pdf_bbox"]),
        "crop_bbox": None,
        "latex": None,
        "normalized_latex": None,
        "raw_prediction": None,
        "confidence": None,
        "status": STATUS_FAILED,
        "problems": [],
        "equation_number": None,
        "equation_number_source": None,
        "equation_number_pdf_bbox": None,
        "image_path": None,
    }


def _remove_old_crops(work_dir):
    # Crops of an earlier run must not be mistaken for crops of this one.
    crops_dir = os.path.join(work_dir, EQUATION_CROPS_DIR)
    for old in glob.glob(os.path.join(glob.escape(crops_dir), "page_*_eq_*.png")):
        try:
            os.remove(old)
        except OSError as exc:
            raise OutputDirectoryError(f"Cannot replace: {old} ({exc})") from exc

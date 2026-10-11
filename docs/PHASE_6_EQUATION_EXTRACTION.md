# Phase 6 — Equation extraction and LaTeX recognition

Phase 6 turns every `EQUATION` region of Phase 3 into LaTeX:

```
Phase 3 EQUATION region
    ->  crop of the Phase 2 page image
    ->  math recognition model (image -> LaTeX)
    ->  normalization (whitespace, wrappers) and validation
    ->  equation number, if one stands beside the equation
    ->  equations.json
```

```json
{ "latex": "E=m c^{2}", "equation_number": "(4)", "status": "success" }
```

Equations only: no figure or chart understanding, no chart-to-data conversion, no reading order, no
section reconstruction, no summarization, no LLM. It runs alongside V1 and Phases 2–5; `ai_engine.py`,
the ASP.NET application and the database are unchanged and do not call it.

## 1. Selected model

**PP-FormulaNet_plus-S**, through PaddleOCR's standalone `FormulaRecognition` module (not the
PP-StructureV3 pipeline). One model; no other math OCR system was installed.

## 2. Why it was selected

The environment was inspected first: Python 3.10.9, `paddleocr` 3.7.0 and `paddlepaddle` 3.3.1 (CPU
build) from Phases 3–5, **no PyTorch**. The `paddleocr` install already contains the formula
recognition module; only its models and two small decoder packages were missing.

| Candidate | Decision | Reason |
|---|---|---|
| **PP-FormulaNet_plus-S** (PaddleOCR) | **Used** | Runs on the installed PaddlePaddle; CPU; no second deep-learning framework; Apache-2.0; about 0.6 s per equation here. Needs `tokenizers` and `ftfy` (pure wheels, a few MB) |
| PP-FormulaNet_plus-M / -L, UniMERNet (PaddleOCR) | Not used | Same module, larger models (about 0.6 GB, 0.7 GB, 1.5 GB) meant for GPU. Not downloaded, not measured. `PPFormulaNetRecognizer(model_name=...)` accepts them |
| LaTeX_OCR_rec (PaddleOCR port of pix2tex) | Not used | Smaller (about 100 MB) but the older architecture of the module's model list. Not downloaded, not measured |
| pix2tex (LaTeX-OCR) | Not installed | Needs PyTorch and further packages on top of it: a second deep-learning framework of several hundred MB next to PaddlePaddle. MIT licence |
| Texify | Not installed | Needs PyTorch and `transformers`; its weights carry commercial-use conditions, where everything else in this pipeline's model stack is Apache-2.0 |

The comparison between candidates is by requirements, size and licence, and what is said about
pix2tex and Texify is from general knowledge of the projects, not re-checked online for this phase.
**Only PP-FormulaNet_plus-S was run**, on the fixtures of section 12, before the module was built.

## 3. Licence

PP-FormulaNet_plus-S weights, PaddleOCR, PaddleX and PaddlePaddle: Apache-2.0. `tokenizers`:
Apache-2.0. `ftfy`: Apache-2.0. `matplotlib` (fixture generation only): its own BSD-style licence.
PyMuPDF from Phase 2 remains AGPL-3.0.

## 4. Dependency versions

| | |
|---|---|
| Python | 3.10.9 (Windows 10) |
| `paddleocr` | 3.7.0 (unchanged) |
| `paddlepaddle` | 3.3.1, CPU build (unchanged) |
| `paddlex` | 3.7.2 (unchanged) |
| `tokenizers` | 0.23.3 — **new**; decodes the model's output tokens |
| `ftfy` | 6.3.1 — **new**; used by the PaddleX formula post-processing |
| `Pillow` | 12.3.0 (crops) |
| `matplotlib` | 3.10.9 — only to regenerate the equation fixtures; not a pipeline or test requirement |
| Model | `PP-FormulaNet_plus-S`, about 250 MB on disk (weights file 256,845,006 bytes), downloaded to `~/.paddlex/official_models/` on first use |

`pip install tokenizers ftfy` changed no installed package (numpy, PaddlePaddle, PaddleOCR and
huggingface_hub keep their versions). `AIEngine/requirements.txt` lists both.

### Code layout

| File | Role |
|---|---|
| `AIEngine/pipeline/math_ocr.py` | `MathRecognizer` interface, `PPFormulaNetRecognizer`, crops, `normalize_latex`, `validate_latex`, equation numbers, `run_document_equations`, `validate_equations_document` |
| `AIEngine/pipeline/math_eval.py` | `canonical_latex`, `latex_cer`, `evaluate_equation`, `evaluate_equations` |
| `AIEngine/tools/inspect_equations.py` | Command-line tool |
| `AIEngine/tools/evaluate_equations.py` | Accuracy against ground truth |
| `tests/python/test_equations.py` | Tests |
| `tests/fixtures/make_equation_fixtures.py`, `tests/fixtures/equations/` | Fixtures and their ground truth |

Only `PPFormulaNetRecognizer` imports PaddleOCR. The rest uses:

```python
class MathRecognizer:
    name: str
    def load(self)                                   # loads once
    def recognize(self, image) -> MathPrediction     # .latex (the model's own text), .confidence (or None)
    def recognize_region(self, page_image, image_bbox, padding=8, exclude=())
                                                     # -> {"prediction", "crop_bbox", "crop", "blank"}
```

```python
import sys; sys.path.insert(0, "AIEngine")
from pipeline import PPFormulaNetRecognizer, run_document_equations

recognizer = PPFormulaNetRecognizer()                # keep one instance
result = run_document_equations("temp/sample", recognizer)     # needs pages.json and layout.json
```

To use another model, subclass `MathRecognizer` and implement `_load_model` and `_recognize`
(return `MathPrediction(latex, confidence_or_None)`).

## 5. CPU / GPU behaviour

Everything was run on **CPU** (12 logical cores). The machine has an NVIDIA GTX 1050 Ti, but the
installed PaddlePaddle is the CPU build; no CUDA driver, no GPU package and no system setting was
changed. `--device gpu:0` is passed through to PaddleOCR and was **not tested**. Unlike the OCR
detection model of Phase 4, this model runs with PaddlePaddle's default oneDNN setting.

The recognizer is loaded once per process and only if the document has an equation region
(`load_count` is checked in the tests).

## 6. Equation cropping

Input is **only** the `EQUATION` regions of `layout.json` (PP-DocLayoutV3 labels `display_formula`
and `inline_formula`), each on its own. Pages are not searched for formulas.

For each region: `page_number`, `image_bbox` and `pdf_bbox` are read from the layout region and copied
to the result unchanged; the crop is cut from the Phase 2 page image (`pages/page_NNN.png`); the PDF
is not rendered again.

Preprocessing is deliberately minimal:

| Step | Done |
|---|---|
| Padding | 8 px of page around the region (`--equation-padding`), never beyond the page |
| Equation number | If a number was found inside the crop (section 9), the crop ends 2 px before it |
| Colour, size, contrast | **Not touched.** The crop is the page pixels in RGB. The model does its own resizing and normalization (PaddleX `UniMERNetImgDecode` / `UniMERNetTestTransform`) |
| Blank check | A crop with less than 16 grey levels of contrast is not sent to the model (`recognition_failed`, "the equation region is blank") |

`crop_bbox` records the pixels that were sent. The exact crop can be saved (section 11); it is never
altered, and no symbol is rewritten before or after recognition.

## 7. LaTeX normalization

`normalize_latex(text)` — syntax only:

- outer whitespace removed; runs of whitespace (line breaks included) become one space;
- model wrapper tokens at the ends removed (`<s>`, `</s>`, `<pad>`, ...);
- one pair of delimiters around the **whole** formula removed: `$...$`, `$$...$$`, `\[...\]`,
  `\(...\)`, `\begin{equation}...\end{equation}` (also `equation*`, `displaymath`, `math`);
- a trailing `\tag{4}` is moved to `equation_number` (`split_tag`).

Nothing else. `a+b` stays `a+b`; braces, commands, spacing commands and the order of symbols are
kept, including the model's own spelling such as `m x` or `\operatorname{l o g}` (in math mode both
render as `mx` and `log`).

Three forms are stored: `raw_prediction` (exactly what the model returned), `normalized_latex`
(after the steps above) and `latex` (the same text, but only when the validation passed).

## 8. Validation

`validate_latex(latex)` returns a list of problems. It checks:

| Check | Example that fails |
|---|---|
| Not empty | `""` |
| Braces balanced (escaped `\{ \}` are symbols) | `\frac{x+y}{2` |
| `\begin{...}` / `\end{...}` paired and nested | `\begin{matrix} a \end{aligned}` |
| `\left` / `\right` paired | `\left( x + y` |
| No `_` or `^` without an argument | `x_`, `{{_}i}` |
| No backslash followed by nothing, a digit or a non-ASCII character; no "command" longer than 24 letters | `x \1 y` |
| No known command name without its backslash (outside `\text`, `\mathrm`, `\operatorname`) | `sum_i`, `frac{a}{b}` |
| No token repeated 16 times in a row (a decoder stuck in a loop) | `\quad \quad \quad ...` |

A formula with a problem gets `status: "invalid_latex"`, `latex: null`, the problems in `problems`,
and keeps `raw_prediction`, `normalized_latex` and its crop image.

This is **not a TeX run**. No TeX installation and no LaTeX parser package is used: neither is
installed, and a full TeX distribution only for this check was judged not worth it. A formula that
passes can still fail to compile (well-formed but unknown commands are accepted), and the check says
nothing about whether the formula matches the image.

## 9. Equation numbering

`find_equation_numbers(equation_boxes, text_lines)`, geometry only, per page, in PDF points:

1. A text line is a candidate only if it is **nothing but** a parenthesized number: `(4)`, `(3.2)`,
   `(12a)`, `(A.1)`. `Eq. (4)`, `f(2)`, `[4]` and `4` are not.
2. It can belong to an equation when its vertical centre lies within the equation's height and its
   horizontal centre is right of the equation's centre (beside the equation, or in the right part of
   a region box that includes the number).
3. A candidate goes to the nearest such equation (smallest horizontal gap), so in a two-column page
   a number is not given to an equation in the other column.
4. An equation with exactly one candidate gets it. With none, or with more than one,
   `equation_number` is `null`.

Text lines come from:

| `equation_number_source` | Where the text is from |
|---|---|
| `"native"` | Native text lines of `pages.json` |
| `"ocr"` | OCR lines of `ocr.json` (Phase 4), when the working directory has one for this layout. PP-DocLayoutV3 finds the number as its own `formula_number` region, which Phase 4 reads |
| `"latex_tag"` | A trailing `\tag{...}` in the model's output, used only when the page shows no number |

The number is kept out of the LaTeX: it is a separate field, and a number lying inside the crop is
cut off before recognition. `equation_number_pdf_bbox` says where it was read.

Phase 6 does not run OCR itself. For a scanned PDF without `ocr.json`, numbers are `null`;
`inspect_equations.py --ocr-numbers` runs Phase 4 first.

## 10. Schema

Written to `<work_dir>/metadata/equations.json`. **`pages.json`, `layout.json`, `ocr.json` and
`tables.json` are not modified.** UTF-8, no image data.

```json
{
  "schema_version": 1,
  "recognizer": { "name": "PP-FormulaNet_plus-S", "provider": "PaddleOCR", "version": "3.7.0",
                  "framework": "paddlepaddle 3.3.1", "device": "cpu" },
  "settings": { "padding_px": 8, "equation_numbers_from_ocr": false },
  "source": { "pages_json": "metadata/pages.json", "layout_json": "metadata/layout.json", "ocr_json": null,
              "source_pdf": "synthetic_equations.pdf", "source_sha256": "..." },
  "coordinate_system": { "bbox_format": "x0,y0,x1,y1", "origin": "top-left", "...": "as in layout.json" },
  "pages": [
    {
      "page_number": 3,
      "equations": [
        {
          "equation_id": "p003_e002",
          "page_number": 3,
          "source_region_id": "p003_r006",
          "source_label": "display_formula",
          "source_model": "PP-FormulaNet_plus-S",
          "image_bbox": [700.8, 677.0, 827.8, 715.0],
          "pdf_bbox": [280.3, 270.8, 331.1, 286.0],
          "crop_bbox": [692, 669, 836, 723],
          "latex": "E=m c^{2}",
          "normalized_latex": "E=m c^{2}",
          "raw_prediction": "E=m c^{2}",
          "confidence": null,
          "status": "success",
          "problems": [],
          "equation_number": "(4)",
          "equation_number_source": "native",
          "equation_number_pdf_bbox": [526.6, 272.7, 540.0, 287.8],
          "image_path": null
        }
      ]
    }
  ],
  "summary": { "pages": 4, "equation_regions": 12, "recognized": 12, "invalid_latex": 0,
               "recognition_failed": 0, "with_equation_numbers": 6, "mean_confidence": null }
}
```

- `equation_id` is `p<page>_e<index>`, in the order of the layout regions. `source_region_id`,
  `page_number`, `image_bbox` and `pdf_bbox` are those of the layout region.
- `source_label` is the layout model's own label: `display_formula` or `inline_formula` (section 15).
- **`confidence` is `null`**: PP-FormulaNet returns the formula only, no score. Nothing is invented;
  the field exists for recognizers that report one, and `summary.mean_confidence` is then their mean.
- `status`: `"success"`, `"invalid_latex"` (section 8) or `"recognition_failed"` (section 14).
- One `EQUATION` region is one LaTeX block. Matrices, `cases` and multi-line derivations come back as
  one environment (`\begin{matrix}`, `\begin{aligned}`, ...); they are never split into several equations.
- A page without equation regions has `"equations": []`.
- Timings are returned by `run_document_equations` and printed by the tool, not stored, so the file is
  identical across runs.
- `validate_equations_document(data)` checks the structure and returns a list of problems.
- Missing or mismatching earlier artifacts raise a `MathInputError` and leave no `equations.json`.

## 11. Command line

```powershell
# PDF: Phase 2 and Phase 3 run first (or are reused), then the equations
python AIEngine/tools/inspect_equations.py tests/fixtures/equations/synthetic_equations.pdf --output temp/equations

# Keep the crop of every equation
python AIEngine/tools/inspect_equations.py sample.pdf --output temp/sample --save-equation-crops

# Scanned PDF: read equation numbers through Phase 4 OCR
python AIEngine/tools/inspect_equations.py scanned.pdf --output temp/scanned --ocr-numbers

# Accuracy against ground truth
python AIEngine/tools/evaluate_equations.py temp/equations --truth tests/fixtures/equations/synthetic_equations.json

# Tests
python -m unittest discover -s tests/python -v
```

Options: `--output`, `--dpi`, `--layout-threshold`, `--equation-padding`, `--save-equation-crops`,
`--ocr-numbers`, `--force`, `--device`.

**Reuse.** Phase 2 and Phase 3 are reused as in `inspect_ocr.py`; `--force` runs them again. An
`ocr.json` that belongs to the current layout is used for equation numbers; it is only created with
`--ocr-numbers`.

**Crops.** `equation_crops/page_003_eq_001.png` is written for every equation with
`--save-equation-crops`, and otherwise only for equations that were not recognized. The directory is
git-ignored; crops of an earlier run are removed at the start of each run.

Exit code 0, or 2 with a one-line `Error: ...` on stderr. Summary on stdout:

```
Pages: 4
Equation regions: 12
Recognized: 12
Failed: 0
With equation numbers: 6
Average confidence: n/a (the model reports none)
Total time: 16.8s

  p001_e001      -  y=m x+b
  p001_e002    (1)  L=-{\sum_{i}}y_{i}\operatorname{l o g}(p_{i})
  p001_e003    (2)  z{=}\frac{x+y}{2}
  ...
Invalid LaTeX: 0; recognition failed: 0
Recognizer: PP-FormulaNet_plus-S, device CPU, loaded 1 time(s)
Phase 2 pages: created; Phase 3 layout: created; Phase 4 OCR: not available
Equations: ...\metadata\equations.json
Time: model init 10.01s, inference 6.69s (0.56s/equation, slowest 0.74s), total 16.77s
```

`Failed` is invalid LaTeX plus failed recognition. `Total time` is the Phase 6 step (model loading and
recognition), without Phases 2 and 3.

## 12. Metrics

`AIEngine/pipeline/math_eval.py`, against ground truth with known LaTeX:

| Metric | Definition |
|---|---|
| Recognition success rate | equations with `status: "success"` / ground-truth equations |
| Exact LaTeX match | canonical forms identical / ground-truth equations |
| Strict match | identical once whitespace is removed and nothing else. Reported for honesty: it mostly measures spelling habits (`x_i` vs `x_{i}`) |
| Character error rate | (substitutions + deletions + insertions) / characters of the reference, on canonical forms, summed over all equations |
| Normalized LaTeX accuracy | mean over the equations of `1 - min(1, CER)` |
| Equation numbers | numbered equations whose number was found; and the share of all equations with the right number or rightly none |

An equation that was not found, or failed, counts with every character wrong.

**Canonical form** (`canonical_latex`, comparison only — never stored): whitespace removed; spacing
and size commands dropped (`\,`, `\quad`, `\displaystyle`, `\left`, `\right`, ...); one-token scripts
braced (`x_i` → `x_{i}`); `\mathrm{log}` / `\operatorname{log}` → `\log`; braces that only group
dropped (`-{\sum_{i}}y` → `-\sum_{i}y`); a row break before `\end{...}` dropped; a `matrix` between
`[ ]` or `( )` written as `bmatrix` / `pmatrix`. Symbols and structure are never merged: `0`/`O`,
`1`/`l`, `x`/`\times`, `-`/`=`, `a`/`\alpha`, `A`/`\lambda`, `x_i^2`/`x^i_2`, `&=`/`=&` all stay
different (tested).

Fixtures (`tests/fixtures/equations/`, 12 equations on 4 pages; ground truth in `synthetic_equations.json`):

| # | Kind | LaTeX | Font | Number |
|---|---|---|---|---|
| 1 | simple algebra | `y = mx + b` | Computer Modern | — |
| 2 | summation | `L = -\sum_i y_i \log(p_i)` | Computer Modern | (1) |
| 3 | fraction | `z = \frac{x+y}{2}` | STIX | (2) |
| 4 | square root | `\sqrt{x^2 + y^2}` | Computer Modern | — |
| 5 | Greek | `\alpha + \beta = \lambda` | STIX | — |
| 6 | sub/superscript | `x_i^2` | DejaVu Sans | — |
| 7 | integral | `\int_a^b f(x)\,dx` | STIX | (3) |
| 8 | numbered | `E = mc^2` | Computer Modern | (4) |
| 9 | matrix | `A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}` | Computer Modern | (5) |
| 10 | multi-line aligned | `\begin{aligned} f(x) &= (x+1)^2 \\ &= x^2 + 2x + 1 \end{aligned}` | STIX | (6) |
| 11 | symbol fidelity | `x_0 \times l_1 = 10` | Computer Modern | — |
| 12 | symbol fidelity | `\lambda A = a - \alpha` | STIX | — |

`synthetic_equations_scanned.pdf` is the same pages as 150 DPI bitmaps.

Results with the real models (this machine, full pipeline: real layout detection, then recognition):

| Input | Regions found | Recognized | Exact match | Strict match | CER | Normalized accuracy | Numbers |
|---|---|---|---|---|---|---|---|
| `synthetic_equations.pdf` | 12/12 | 12/12 | 10/12 (0.833) | 3/12 | 0.0433 (11/254) | 0.976 | 6/6, native text |
| `synthetic_equations_scanned.pdf` (`--ocr-numbers`) | 12/12 | 11/12 | 10/12 (0.833) | 4/12 | 0.0866 (22/254) | 0.948 | 6/6, OCR text |

What went wrong, exactly:

| Input | Equation | Model output (canonical) | Kind of error |
|---|---|---|---|
| digital | `x_i^2` (sans-serif) | `x_{j}^{2}` | **Symbol error**: `i` read as `j`. Valid LaTeX, so it is *not* caught by validation |
| digital, scanned | aligned derivation | `\begin{aligned}f_{x}{}(x)=&(x+1)^{2}\\=&x^{2}+2x+1\end{aligned}` | A spurious subscript on `f`, and `=&` where the source has `&=`. One block, two lines, as intended |
| scanned | summation | `L\mathrm{=}-\ \ {sum{{_}i}}y_{i}\operatorname{l o g}(p_{i})` | Malformed output. **Caught by validation** (`_ has no argument; command without a backslash: sum`), stored as `invalid_latex` with its crop |

Thresholds asserted in the integration tests (set below these results): digital — at least 11
recognized, 9 exact, CER ≤ 0.10, all 6 numbers; scanned — at least 10 recognized, 8 exact,
CER ≤ 0.15, at least 5 numbers; and nine named formulas (algebra, fraction, square root, Greek,
integral, numbered, matrix, both symbol-fidelity formulas) exact in both documents.

**These are engineering checks, not benchmark results.** Twelve short, clean, generated formulas say
that the code paths work and where this model already slips; they say nothing about accuracy on real
papers. The formulas are drawn by matplotlib's mathtext, not by TeX.

## 13. Performance (reference, this machine, CPU)

12 logical cores, 16 GB RAM, 180 DPI page images, model already downloaded, single runs, earlier
phases reused:

| | |
|---|---|
| Model init, new process | 10.0–10.3 s (includes importing PaddlePaddle); 18.6 s in one earlier run with a cold file cache |
| Inference per equation | 0.56–0.58 s on average, slowest 0.74–0.78 s; 0.70 s on average while the machine was busy |
| `synthetic_equations.pdf`: 12 equations | 16.8 s, of which 10.0 s model init and 6.7 s inference |
| `synthetic_equations_scanned.pdf`: 12 equations | 17.1 s, of which 10.1 s model init and 6.9 s inference |
| Document without equation regions | model not loaded |
| First use | downloads about 250 MB (took about 2 minutes here) |

Measured: model initialization time, inference time per equation (`timings["equation_s"]`), and
total Phase 6 time (`timings["total_s"]`), all printed by the tool. Equations are recognized one at a
time (batch size 1).

## 14. Failure fallback

An equation is never dropped. Every `EQUATION` region has an entry with its identity and both boxes,
whatever happens:

| Case | `status` | `latex` | Kept |
|---|---|---|---|
| Model raised an error, or returned no / empty text | `recognition_failed` | `null` | `error`, `crop_bbox`, `image_path` |
| Region is blank (no contrast) | `recognition_failed` | `null` | `error`, `crop_bbox`, `image_path` |
| Region cannot be cut out (outside the page, under 6 px) | `recognition_failed` | `null` | `error`; no crop exists, so `image_path` is `null` — the boxes still locate it on the page image |
| Output is not sound LaTeX | `invalid_latex` | `null` | `raw_prediction`, `normalized_latex`, `problems`, `crop_bbox`, `image_path` |

```json
{ "status": "recognition_failed", "latex": null, "image_path": "equation_crops/page_002_eq_002.png",
  "error": "MathRecognitionError: RuntimeError: ..." }
```

The crop of every such equation is **always** written to `equation_crops/`, with or without
`--save-equation-crops`, so it is available for display or manual review. The other equations of the
document are processed normally. Only a model that cannot be loaded at all fails the run (exit code 2).

## 15. Known limitations

- **Tested on synthetic formulas only**: twelve short formulas drawn by mathtext. No real paper.
- **Wrong but valid LaTeX is not detected.** The `x_j^2` result above has `status: "success"`.
  Validation checks syntax, not content, and there is no confidence to threshold on.
- **No confidence.** PP-FormulaNet returns none, so `confidence` is always `null`.
- **Inline equations.** The layout model has an `inline_formula` label and such regions are processed
  like any other (`source_label` tells them apart), but it rarely emits it: none was detected in any
  fixture, and inline math inside paragraphs is in practice left as part of the `TEXT` region. Phase 6
  adds no inline-formula detector. The target is display equations.
- The equation must be found by Phase 3. A formula the layout model misses, splits or boxes too
  tightly is not repaired here. A derivation the layout model splits into several regions becomes
  several equations.
- **Multi-line equations** are one LaTeX block with the environment the model chose (`aligned`,
  `matrix`, `array`, `cases`). Alignment points (`&`) and the choice of environment are not reliable,
  as the aligned fixture shows. Lines are not numbered individually: a block gets at most one number.
- **Equation numbers**: right-hand numbers in parentheses only. Numbers on the left, numbers in
  square brackets, numbers on a different line than the equation's vertical extent, and several
  numbers beside one multi-line block give `null`. On scanned pages numbers need Phase 4 OCR text.
  A native text line that holds the equation and its number together is not split.
- The crop is the layout box plus padding. Text of a neighbour inside that box or padding (a comma
  after the formula, a word) is sent to the model with it.
- Validation is a syntax check, not TeX: unknown commands are accepted, and a formula that needs a
  package to compile is not flagged. The list of command names checked for a missing backslash is short.
- The canonical form used for the metrics treats some spellings as equal (section 12); it is not a
  proof of mathematical equivalence, and formulas that differ only in an untreated spelling
  (`\begin{array}{cc}` vs `matrix`, `&=` vs `=&`) count as different.
- Small model (`-S`). Long, dense or handwritten formulas, and formulas in unusual fonts, are expected
  to be harder; this was not measured. Larger models of the family were not tried.
- CPU only, one equation at a time; about 10 s to load the model per process.
- Not connected to the web application, V1 or the database. Nothing renders the LaTeX.

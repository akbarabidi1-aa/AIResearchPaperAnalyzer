# Phase 3 — Document layout detection

Phase 3 takes the page images of Phase 2 and finds **typed regions** on each page:

```
Phase 2 page image  ->  layout model  ->  raw detections  ->  normalization  ->  Region  ->  layout.json
```

A region says *where* something is and *what kind of thing* it is (title, text, table, figure, ...). Nothing
more: no OCR, no table rows/columns/cells, no formula-to-LaTeX, no figure or chart understanding, no
section or reading-order reconstruction. It runs alongside V1 and Phase 2; `ai_engine.py`, the ASP.NET
application and the database are unchanged and do not call it.

## 1. Why PP-DocLayoutV3

- It is a current document-layout model trained on papers, reports and scans in several languages, with
  the classes a research paper needs: titles, paragraphs, tables, images, charts, display and inline
  formulas, captions, headers, footers, page numbers, references, footnotes, algorithms.
- It works on the page **image**, so digital and scanned pages are handled the same way.
- PaddleOCR exposes it as a standalone module (`LayoutDetection`), so it can be used without the full
  PP-StructureV3 / PaddleOCR-VL document pipelines.
- It runs on CPU and the weights are permissively licensed.

## 2. Model and provider

| | |
|---|---|
| Model | PP-DocLayoutV3 (DETR-style detector, input resized to 800 x 800) |
| Provider | PaddlePaddle / PaddleOCR — <https://huggingface.co/PaddlePaddle/PP-DocLayoutV3> |
| Called through | `paddleocr.LayoutDetection(model_name="PP-DocLayoutV3", device="cpu")` |
| Weights | `inference.pdiparams`, 130.8 MB (about 126 MB on disk with config files) |
| Cache location | `~/.paddlex/official_models/PP-DocLayoutV3` (downloaded on first use, about 1 minute here) |
| Device | CPU. No GPU build of PaddlePaddle is installed; CUDA and drivers were not touched |

## 3. Licence

PP-DocLayoutV3 weights: Apache-2.0 (model card). PaddleOCR, PaddleX and PaddlePaddle: Apache-2.0.
Pillow (debug images): MIT-CMU. PyMuPDF from Phase 2 remains AGPL-3.0.

## 4. Dependency versions

Added to `AIEngine/requirements.txt`:

| Package | Version verified | Why |
|---|---|---|
| `paddlepaddle` | 3.3.1 (CPU wheel) | Inference framework |
| `paddleocr` | 3.7.0, base install without extras | `LayoutDetection` module; pulls in `paddlex` 3.7.2 |
| `Pillow` | 12.3.0 | Debug images (already required by `paddlex`) |

Python 3.10.9, Windows 10. `pip` installed 40 packages in total, among them `opencv-contrib-python`
4.10.0.84, `shapely`, `pyclipper`, `huggingface_hub` and `modelscope` (all required by the base
`paddlex` install). numpy, pandas, scikit-learn and PyMuPDF were not changed. Disk use: `paddle` 386 MB,
`cv2` 122 MB, `paddlex` 17 MB, plus the 126 MB model.

```powershell
python -m pip install -r AIEngine/requirements.txt
```

GPU: a CUDA build of PaddlePaddle would work with the same code (`--device gpu:0`) but was not installed
or tested.

## Code layout

| File | Role |
|---|---|
| `AIEngine/pipeline/layout.py` | `LayoutDetector` interface, `PPDocLayoutV3Detector`, `RegionType`, `Region`, label map, normalization, coordinate mapping, `detect_document_layout`, `validate_layout_document` |
| `AIEngine/pipeline/layout_visualize.py` | `draw_layout`: debug image with boxes |
| `AIEngine/pipeline/layout_eval.py` | `bbox_iou`, `match_regions`, `class_accuracy` |
| `AIEngine/tools/inspect_layout.py` | Command-line tool |
| `AIEngine/tools/evaluate_layout.py` | Evaluation table against ground truth (`tests/layout_eval/`) |
| `tests/python/test_layout.py` | Tests |

### The layout abstraction (replacing the model)

Only `PPDocLayoutV3Detector` imports PaddleOCR. The rest of the code uses:

```python
class LayoutDetector:
    name: str                      # written to Region.source_model and layout.json
    label_map: dict                # model label -> RegionType
    def load(self)                 # loads once; later calls do nothing
    def detect(self, image_path, threshold) -> list[RawDetection]   # label, score, bbox in image pixels
```

To use another model, subclass `LayoutDetector`, implement `_load_model` and `_detect`, give it a
`label_map`, and pass an instance to `detect_document_layout`. `Region`, `layout.json`, the coordinate
mapping, the visualization and the evaluation helpers do not change.

```python
import sys; sys.path.insert(0, "AIEngine")
from pipeline import PPDocLayoutV3Detector, detect_document_layout, process_pdf

detector = PPDocLayoutV3Detector()                 # keep one instance; it loads the model once
process_pdf("sample.pdf", "temp/sample")           # Phase 2
result = detect_document_layout("temp/sample", detector, threshold=0.5)
```

## 5. Source labels

The 25 labels of the model (`label_list` in its `inference.yml`):

`abstract`, `algorithm`, `aside_text`, `chart`, `content`, `display_formula`, `doc_title`, `figure_title`,
`footer`, `footer_image`, `footnote`, `formula_number`, `header`, `header_image`, `image`,
`inline_formula`, `number`, `paragraph_title`, `reference`, `reference_content`, `seal`, `table`, `text`,
`vertical_text`, `vision_footnote`.

## 6. Canonical label mapping

Canonical classes: `TITLE TEXT TABLE FIGURE EQUATION CAPTION HEADER FOOTER PAGE_NUMBER LIST OTHER`.

The model label is always kept in `source_label`; `type` is the canonical class. The table is
`PP_DOCLAYOUT_V3_LABEL_MAP` in `layout.py`.

| Model label | Canonical | Note |
|---|---|---|
| `doc_title` | TITLE | Title of the document |
| `paragraph_title` | TITLE | Section / subsection heading. Tell the two apart with `source_label` |
| `text` | TEXT | Body paragraph |
| `abstract` | TEXT | |
| `reference` | TEXT | Reference section (see section 14: PaddleX drops this label) |
| `reference_content` | TEXT | One bibliography entry |
| `algorithm` | TEXT | Pseudo-code block |
| `vertical_text` | TEXT | |
| `table` | TABLE | |
| `image` | FIGURE | |
| `chart` | FIGURE | |
| `display_formula` | EQUATION | |
| `inline_formula` | EQUATION | Formula inside a line of text |
| `figure_title` | CAPTION | Caption of a figure, table or chart |
| `vision_footnote` | CAPTION | Note attached to a figure or table |
| `header`, `header_image` | HEADER | Running header |
| `footer`, `footer_image` | FOOTER | Running footer |
| `number` | PAGE_NUMBER | |
| `content` | LIST | Table of contents |
| `formula_number` | OTHER | The `(3)` tag next to an equation; not a formula itself |
| `footnote` | OTHER | |
| `aside_text` | OTHER | Margin text |
| `seal` | OTHER | Stamp |
| anything else | OTHER | A label not in the table (for example from a newer model version) |

The model has no class for bulleted or numbered lists; they are detected as `text`. `LIST` is therefore
only produced for a table of contents. No label is invented: every `type` comes from a label the model
predicted.

## 7. Image coordinate system

`image_bbox` is in pixels of the Phase 2 page image (`pages/page_NNN.png`):

| Property | Value |
|---|---|
| Origin | Top-left corner of the image |
| Axes | x to the right, y **down** |
| Format | `[x0, y0, x1, y1]`, `x0 < x1`, `y0 < y1` |
| Range | `0 <= x <= image_width`, `0 <= y <= image_height` |

The model resizes the image to 800 x 800 internally and returns boxes already scaled back to the original
image; the values it returns are whole pixels.

## 8. PDF coordinate mapping

`pdf_bbox` is the same box in PDF points, in the Phase 2 page coordinate system (origin top-left of the
displayed page, y down — see `PHASE_2_PAGE_PIPELINE.md`). It is computed from the values **stored in
`pages.json`**, never from `dpi / 72`:

```
pdf_x = image_x / scale_x
pdf_y = image_y / scale_y
```

`image_bbox_to_pdf` is the exact inverse of Phase 2's `pdf_bbox_to_pixels`. Boxes are clipped to the image
first and, to absorb floating-point rounding, to the page afterwards. A detection that is malformed
(wrong length, non-finite, inverted) or empty after clipping is discarded and counted in the page's
`discarded_detections`.

Validation (tests):

- `0 <= x0 < x1 <= width` and `0 <= y0 < y1 <= height` for both boxes, on real Phase 2 pages at 72, 100
  and 180 DPI and on an odd-sized page, including boxes on the last pixel and boxes larger than the page.
- Phase 2 block boxes sent to image space and back return unchanged.
- With the real model, the PDF-space box of the detected table and of each text column has IoU > 0.8 with
  the native text block of Phase 2 in the same place.

## 9. Threshold

| | |
|---|---|
| Default | `0.5` — the model's own default (`draw_threshold` in its `inference.yml`); `DEFAULT_LAYOUT_THRESHOLD` |
| Override | `--layout-threshold 0.4`, or `detect_document_layout(..., threshold=0.4)` |
| Range | 0 to 1, otherwise an error |
| Stored | `model.threshold` in `layout.json`; every region keeps its unrounded `score` |

The threshold was not tuned on the fixtures. Lowering it far is not recommended; see "Overlap handling".

### Overlap handling (NMS)

What the model side already does, in this order (read from the PaddleX source, `layout_analysis/processors.py`):

1. Score threshold.
2. Class-aware NMS (`layout_nms`): **off** by default, and left off. The detector is DETR-style and
   predicts a set of boxes without needing NMS.
3. An overlap filter (`filter_boxes`), always on: boxes under 6 px are removed; when two boxes overlap by
   more than 70 % of the smaller one, the smaller is removed (with exceptions for image/table/chart/seal
   pairs); `reference` boxes are removed.

Observed on the fixtures and two generated pages (7 pages, thresholds 0.5, 0.1 and 0.05): no duplicate
boxes, no heavily overlapping same-class boxes, no page-sized boxes. **No second NMS stage was added.**

`find_overlap_issues` in `layout.py` only *reports* same-class pairs with IoU >= 0.8 and boxes covering
>= 90 % of the page; the tool prints them as "Overlap warnings". It removes nothing. If real papers show
duplicates, a filter belongs in a separate function next to it.

One side effect of step 3 was observed: it ignores scores. At threshold 0.1 on the two-column fixture a
`vision_footnote` box with score 0.13 replaced the right-hand `text` box with score 0.87, because the
low-score box was slightly larger. At the default 0.5 this did not happen.

## 10. `layout.json`

Written to `<work_dir>/metadata/layout.json`, next to `pages.json`. **`pages.json` and the page images are
not modified.** UTF-8, no image data; `image_path` is relative to the working directory.

```json
{
  "schema_version": 1,
  "model": {
    "name": "PP-DocLayoutV3",
    "provider": "PaddleOCR",
    "version": "3.7.0",
    "framework": "paddlepaddle 3.3.1",
    "device": "cpu",
    "threshold": 0.5
  },
  "source": {
    "pages_json": "metadata/pages.json",
    "source_pdf": "synthetic_digital.pdf",
    "source_sha256": "542b3ffc..."
  },
  "coordinate_system": {
    "bbox_format": "x0,y0,x1,y1",
    "origin": "top-left",
    "y_axis": "down",
    "image_bbox_unit": "px",
    "pdf_bbox_unit": "pt",
    "pdf_mapping": "pdf_x = image_x / scale_x; pdf_y = image_y / scale_y (scale_x, scale_y from pages.json)"
  },
  "pages": [
    {
      "page_number": 2,
      "image_path": "pages/page_002.png",
      "image_width": 1530,
      "image_height": 1980,
      "pdf_width": 612.0,
      "pdf_height": 792.0,
      "discarded_detections": 0,
      "regions": [
        {
          "region_id": "p002_r003",
          "page_number": 2,
          "type": "TABLE",
          "score": 0.929018497467041,
          "image_bbox": [174.0, 277.0, 1071.0, 434.0],
          "pdf_bbox": [69.6, 110.8, 428.4, 173.6],
          "source_label": "table",
          "source_model": "PP-DocLayoutV3"
        }
      ]
    }
  ]
}
```

- `model.version` is the PaddleOCR package version; the model itself carries no version number.
- `region_id` is `p<page>_r<index>`, unique in the document. Regions are in the order the model returned
  them. That order is **not** a reading order and must not be used as one.
- A region has exactly these eight fields. In code it is the frozen dataclass `Region`
  (`to_dict` / `from_dict`), with `type` a `RegionType`.
- A page with no detections has `"regions": []`.
- `validate_layout_document(data)` checks the structure and returns a list of problems. Timings are
  returned by `detect_document_layout` and printed by the tool, not stored, so the same input and model
  give the same file.
- A failed run (missing Phase 2 files, model load error, a page that fails) raises a `LayoutError` and
  leaves no `layout.json`. Pages are never skipped.

## 11. Command line

```powershell
# PDF: runs Phase 2 first, then layout detection. Output: AIEngine/output/<name>-<hash>/
python AIEngine/tools/inspect_layout.py tests/fixtures/digital/synthetic_digital.pdf

# Choose the directory, the threshold, and save debug images
python AIEngine/tools/inspect_layout.py sample.pdf --output temp/sample --layout-threshold 0.5 --visualize-layout

# An existing Phase 2 working directory: pages.json and the images are reused
python AIEngine/tools/inspect_layout.py temp/sample

# Tests
python -m unittest discover -s tests/python -v
```

Options: `--output`, `--dpi` (PDF input), `--layout-threshold`, `--visualize-layout`, `--device` (default `cpu`).
Exit code 0, or 2 with a one-line `Error: ...` on stderr. PaddlePaddle prints its own start-up messages on
stderr; the summary is on stdout:

```
Pages: 2

Detected regions:
TITLE: 7
TEXT: 6
TABLE: 1
CAPTION: 1

Model: PP-DocLayoutV3
Device: CPU
Threshold: 0.5
Overlap warnings: 0
Layout: ...\metadata\layout.json
Time: model init 7.14s, inference 13.95s (6.97s/page), total 21.10s
```

## 12. Visualization

With `--visualize-layout` (or `visualize=True`) each page is also saved as
`<work_dir>/layout_debug/page_NNN_layout.png`: a copy of the page image with every box, its canonical
class and its score drawn in a colour per class. Off by default. The original `pages/page_NNN.png` is
never changed. `layout_debug/` is git-ignored. For development and manual evaluation only.

## 13. Performance (reference, this machine, CPU)

12 logical cores, 16 GB RAM, 180 DPI page images (1530 x 1980), single run each with the model already
downloaded:

| PDF | Pages | Model init | Inference | Per page |
|---|---|---|---|---|
| `synthetic_digital.pdf` | 2 | 7.1 s | 14.0 s | 7.0 s |
| `synthetic_two_column.pdf` | 1 | 8.0 s | 7.9 s | 7.9 s |
| `synthetic_scanned.pdf` | 1 | 8.4 s | 7.6 s | 7.6 s |
| generated 10-page PDF | 10 | 7.1 s | 64.3 s | 6.4 s |

- **Model init** (5–11 s over all runs) is paid once per process and includes importing PaddlePaddle. The
  model is loaded once and reused for every page (`detector.load_count` is checked in the tests). A
  long-running worker pays it once at start-up.
- **Inference** is about 5–8 s per page on CPU, with little dependence on page content. A 10-page paper
  takes about a minute. The first page of a process is usually the slowest.
- First use additionally downloads the model (about 1 minute here).

## 14. Limitations

- **Tested on synthetic pages only.** The repository has no real papers. The evaluation table in
  `tests/layout_eval/` gives 18/18 correct classes and mean IoU 0.92 on two synthetic fixtures; this shows
  the plumbing works and is not an accuracy figure for real documents.
- The committed scanned fixture is a checkerboard bitmap with nothing on it, and yields 0 regions. That an
  image-only page with real content is handled is shown by a test that turns the digital fixture into
  bitmaps (no text layer) and gets the same regions as for the original.
- CPU inference is slow (about 6 s per page). No GPU, batching or parallel pages yet.
- Regions are in model order; there is no reading order and no link between a caption and its figure/table.
- Canonical classes are coarser than the model's: document title vs. heading, chart vs. image, footnote
  vs. margin text are only distinguishable through `source_label`.
- `reference` boxes are removed inside PaddleX (overlap filter), so that label does not appear in the
  output; bibliography entries come through as `reference_content`.
- Lists are not a model class and come out as `TEXT`.
- The overlap filter inside PaddleX ignores scores (see "Overlap handling"); use thresholds well below 0.5
  with care.
- Boxes are axis-aligned rectangles. The polygon the model also returns for skewed pages is not used.
- The caption of the fixture table is found as `vision_footnote` with a low score (0.54), so a slightly
  higher threshold would lose it.
- `paddleocr`/`paddlepaddle` are pinned; `layout.json` records the package version but not a hash of the
  weights, and a re-download could in principle fetch updated weights.
- The whole dependency set is large (about 0.7 GB with the model) and is installed in the same Python
  environment as V1.
- Not connected to the web application or the database.

# Phase 4 — Selective OCR for scanned / image-based text

Phase 4 gives every layout region of Phase 3 its **text**, and records where that text came from:

```
Phase 2 pages.json (native lines)  +  Phase 3 layout.json (typed regions)
    ->  routing      usable native text inside the region?
                        yes -> keep the native text        text_source = "native"
                        no  -> OCR the region's image crop  text_source = "ocr"
    ->  ocr.json
```

Text only: no table rows/columns/cells, no formula-to-LaTeX, no figure or chart understanding, no
reading order between regions, no summarization. It runs alongside V1, Phase 2 and Phase 3;
`ai_engine.py`, the ASP.NET application and the database are unchanged and do not call it.

## 1. Why OCR is selective

OCR is slow (about 0.1–2 s per region on CPU here) and makes mistakes that the PDF's own text does not.
Most research papers are digital PDFs, where OCR would add cost and errors and nothing else. So OCR is
a fallback: it runs only on regions that have no usable native text. For a clean digital PDF no region
needs it and the OCR models are **not even loaded**.

## 2. Why native PDF text is preferred

Native text is what the author's software wrote into the file: exact characters, including symbols,
accents and digits that OCR confuses (`l`/`1`, `O`/`0`, `rn`/`m`). It is free to read and has no
confidence to worry about. OCR text is an estimate from pixels. Priority is therefore always:

1. native PDF text;
2. OCR, only where native text is absent or unusable.

## 3. OCR engine

PaddleOCR, **PP-OCRv5**, through its two standalone modules: `TextDetection` (finds text lines in an
image) and `TextRecognition` (reads one line). The same `paddleocr` install as Phase 3 is used; no
second PaddlePaddle stack, no new package, no GPU driver or CUDA change.

The bundled `paddleocr.PaddleOCR` pipeline is **not** used, for two measured reasons:

- With `paddlepaddle` 3.3.1 on CPU, the detection model fails inside oneDNN
  (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`).
  It works with `enable_mkldnn=False`.
- The pipeline takes one such switch for all of its models. Recognition works with oneDNN and is about
  4x faster with it (0.16 s instead of 0.62 s per line). With the two modules, detection runs without
  oneDNN and recognition with it: 15 regions took 8.6 s instead of 35 s.

Only `PaddleOCREngine` in `AIEngine/pipeline/ocr.py` imports PaddleOCR. The rest of the code uses:

```python
class OCREngine:
    name: str
    def load(self)                                        # loads once; later calls do nothing
    def recognize_region(self, page_image, image_bbox, padding=8)   # -> lines in page pixels, crop box
    def recognize_page(self, page_image)                  # -> lines of a whole page image
    def describe(self)                                    # the "engine" object of ocr.json
```

To use another engine, subclass `OCREngine` and implement `_load_model` and `_recognize(image)`, which
returns `OcrLine(text, confidence, bbox, polygon)` in pixels of the image it was given. Cropping,
padding, coordinate mapping, cleanup, routing and `ocr.json` do not change.

```python
import sys; sys.path.insert(0, "AIEngine")
from pipeline import PaddleOCREngine, PPDocLayoutV3Detector, detect_document_layout, process_pdf, run_document_ocr

process_pdf("sample.pdf", "temp/sample")                              # Phase 2
detect_document_layout("temp/sample", PPDocLayoutV3Detector())        # Phase 3
engine = PaddleOCREngine()                                            # keep one instance
result = run_document_ocr("temp/sample", engine)                      # Phase 4 -> temp/sample/metadata/ocr.json
```

## 4. Dependency versions

Nothing was installed for Phase 4. Found in the environment left by Phase 3 and reused:

| Package | Version |
|---|---|
| Python | 3.10.9 (Windows 10) |
| `paddleocr` | 3.7.0 |
| `paddlepaddle` | 3.3.1, CPU build (`paddle.is_compiled_with_cuda()` is `False`) |
| `paddlex` | 3.7.2 |
| `Pillow` | 12.3.0 (page image crops) |
| `numpy` | 2.2.6 |
| `PyMuPDF` | 1.28.2 |

`AIEngine/requirements.txt` only gained a comment.

## 5. Model and device

| | |
|---|---|
| Detection model | `PP-OCRv5_mobile_det`, 4.7 MB on disk |
| Recognition model | `en_PP-OCRv5_mobile_rec`, 7.6 MB on disk (English / Latin letters, digits, punctuation) |
| Cache location | `~/.paddlex/official_models/` (downloaded on first use, 12.3 MB in total) |
| Device | CPU. `--device gpu:0` is passed through but no GPU build is installed and it was not tested |

The mobile models were chosen because they are small and fast on CPU and read the fixtures almost
without error. For other scripts pass another recognition model, for example
`PaddleOCREngine(recognition_model="PP-OCRv5_mobile_rec")` (Chinese/English/Japanese) or
`"latin_PP-OCRv5_mobile_rec"`; the server models (`PP-OCRv5_server_det`, `PP-OCRv5_server_rec`) are
larger and slower. None of these alternatives was measured.

## 6. Supported region classes

| Canonical class | Native text in the region | No usable native text |
|---|---|---|
| `TEXT`, `TITLE`, `CAPTION`, `HEADER`, `FOOTER`, `PAGE_NUMBER`, `LIST` | native | **OCR** |
| `OTHER` (footnote, margin text, equation number, stamp) | native | **OCR** |
| `TABLE` | native, as raw text | nothing; raw OCR lines with `--ocr-tables` |
| `EQUATION` | nothing | nothing |
| `FIGURE` | nothing | nothing |

- **Tables** get text lines only. Lines on the same height are joined with a space, so a row reads
  `Net-A 8 81.2 12`; that is line joining, not structure. There are no rows, columns or cells in
  `ocr.json`. Table parsing is Phase 5.
- **Equations** get no text at all, native or OCR: glyph soup from a formula is not a usable result,
  and formula recognition is a later phase.
- **Figures** get no text: labels inside a figure are part of the figure, which is not interpreted.

## 7. Routing decision

`route_region(region, page, ocr_tables=False)` returns a `Routing(decision, reason, native_text,
native_blocks)`; `should_ocr(region, page)` is the same decision as a boolean.

| `decision` | `reason` | When |
|---|---|---|
| `skip` | `region_type_has_no_text` | `EQUATION`, `FIGURE` |
| `native` | `native_text_in_region` | Usable native text lies inside the region |
| `skip` | `table_ocr_not_requested` | `TABLE` without native text, and no `--ocr-tables` |
| `ocr` | `native_text_unusable` | Native text is there but cannot be used (see section 8) |
| `ocr` | `page_has_no_text_layer` | No native text in the region; the page has no text layer |
| `ocr` | `no_native_text_in_region` | No native text in the region; the page does have a text layer |

The decision is made **per region**. Phase 2's page-level `has_text_layer` only chooses between the
last two reasons; it never decides whether OCR runs. A digital page with one pasted picture of text
therefore gets native text for its paragraphs and OCR for that one region.

All regions are routed before anything else happens. The engine is loaded only if at least one region
has `decision: "ocr"`.

## 8. Native-text overlap logic

**Lines, not blocks.** A Phase 2 block is MuPDF's grouping of nearby lines, and it regularly spans two
layout regions: in the digital fixture `Abstract` and the paragraph under it are one block, while
Phase 3 finds a `TITLE` and a `TEXT` region. Comparing block boxes with region boxes would give the
heading region the whole paragraph, or send both regions to OCR. Phase 2 therefore now also stores the
**lines** of each block (`blocks[].lines[]`, box and text; see `PHASE_2_PAGE_PIPELINE.md`). This is the
only change to an earlier phase: the field is added, nothing else in `pages.json` differs. A
`pages.json` written before Phase 4 is rejected with a message to run Phase 2 again.

The rule, `native_lines_in_region`:

```
inside(line, region) = area(line ∩ region) / area(line)        (PDF points, both boxes from the JSON files)

A native line belongs to the region when inside >= 0.5 (LINE_INSIDE_MIN).
```

It is intersection over the **line's** area, not IoU: a line is much smaller than its region, so IoU
would always be near zero. With 0.5 a line goes to the region that holds most of it, and to at most
one of two regions that do not overlap.

The lines found are joined (section "Text cleanup") into the region's native text. That text is used
when `is_usable_text` accepts it:

- it has at least one letter or digit, and
- at most 10 % of its non-space characters are undecodable (U+FFFD, private-use code points, control
  characters) — the sign of a font without a usable character map.

Otherwise the region is OCR'd. The blocks that contributed are recorded in `native_blocks`.

## 9. Crop generation

`crop_box(image_bbox, image_width, image_height, padding)` and `OCREngine.recognize_region`:

1. The crop comes from the **Phase 2 page image** (`pages/page_NNN.png`), using Phase 3's `image_bbox`.
   The PDF is not opened or rendered again. The image must have the size stored in `pages.json`.
2. The box is clipped to the image, then rounded outward to whole pixels.
3. A box that is malformed, entirely outside the image, or smaller than 6 px on a side after clipping
   cannot be read: the region is reported as a failure (section 10) and the run continues.
4. `padding` pixels (default 8, `--ocr-padding`) are added on every side, never beyond the page edge.
   Text detection is unreliable for text that touches the border of its image.
5. A crop without contrast (lightest and darkest pixel less than 16 grey levels apart) is empty paper.
   It is not sent to the engine; the region gets `text: ""`.
6. Lines whose centre lies in the padding, outside the region itself, belong to a neighbouring region
   and are dropped.

Detected line polygons are cut out as upright rectangles (pages are rendered upright).

### Text cleanup

Conservative, and the same for native and OCR text (`clean_line`, `join_lines`):

- per line: outer whitespace removed, inner runs of whitespace replaced by one space;
- lines ordered top to bottom; two lines whose vertical centres lie inside each other are one row, and
  its fragments are joined left to right with a space;
- rows joined with a line break (`\n`).

No spelling correction, no dictionary, no language model, no de-hyphenation, no rewriting. For OCR'd
regions `raw_text` keeps the engine's lines exactly as returned and in the engine's order.

## 10. OCR result schema

Written to `<work_dir>/metadata/ocr.json`, next to `pages.json` and `layout.json`, **neither of which
is modified**. UTF-8, no image data.

```json
{
  "schema_version": 1,
  "engine": {
    "name": "PaddleOCR", "version": "3.7.0", "framework": "paddlepaddle 3.3.1", "device": "cpu",
    "detection_model": "PP-OCRv5_mobile_det", "recognition_model": "en_PP-OCRv5_mobile_rec"
  },
  "settings": { "padding_px": 8, "ocr_tables": false, "line_inside_min": 0.5 },
  "source": {
    "pages_json": "metadata/pages.json", "layout_json": "metadata/layout.json",
    "source_pdf": "synthetic_mixed.pdf", "source_sha256": "..."
  },
  "coordinate_system": { "bbox_format": "x0,y0,x1,y1", "origin": "top-left", "y_axis": "down", "...": "as in layout.json" },
  "pages": [
    {
      "page_number": 1,
      "has_text_layer": true,
      "regions": [
        {
          "region_id": "p001_r002", "page_number": 1, "canonical_type": "TEXT", "source_label": "text",
          "image_bbox": [174.0, 280.0, 1340.0, 357.0], "pdf_bbox": [69.58, 112.0, 535.82, 142.8],
          "text": "This first paragraph is ordinary selectable text ...",
          "text_source": "native",
          "confidence": null,
          "engine": null,
          "routing": { "decision": "native", "reason": "native_text_in_region" },
          "native_blocks": [1]
        },
        {
          "region_id": "p001_r003", "page_number": 1, "canonical_type": "TEXT", "source_label": "text",
          "image_bbox": [178.0, 480.0, 1323.0, 597.0], "pdf_bbox": [71.18, 192.0, 529.02, 238.8],
          "text": "This second paragraph was pasted into the page as a picture. ...\nrendered page.",
          "text_source": "ocr",
          "confidence": 0.9910979657462149,
          "engine": "PaddleOCR",
          "routing": { "decision": "ocr", "reason": "no_native_text_in_region" },
          "raw_text": "rendered page.\nbut absent from the text layer, ...",
          "crop_bbox": [170, 472, 1331, 605],
          "lines": [
            {
              "text": "rendered page.",
              "confidence": 0.9854884743690491,
              "image_bbox": [178.0, 559.0, 370.0, 595.0],
              "pdf_bbox": [71.18, 223.6, 147.95, 238.0],
              "polygon": [[179.0, 559.0], [370.0, 562.0], [369.0, 595.0], [178.0, 592.0]]
            }
          ]
        }
      ]
    }
  ],
  "summary": {
    "pages": 1, "regions": 4, "text_regions": 4, "native_regions": 3, "ocr_regions": 1, "ocr_empty": 0,
    "ocr_failures": 0, "skipped_regions": 0, "mean_ocr_confidence": 0.9910979657462149
  }
}
```

- Every region of `layout.json` appears once, in the same order, with the same `region_id` and boxes.
  `canonical_type` is Phase 3's `type`.
- **OCR regions** additionally have `raw_text`, `crop_bbox` (page pixels of the image sent to the
  engine) and `lines`. Each line keeps its own unrounded `confidence`, its box in page pixels and in
  PDF points, and the detected `polygon` (four corners, page pixels). Word boxes are not produced.
- Region `confidence` is the mean of the line confidences weighted by their number of characters.
  It is `null` for native text and for OCR that found no text.
- **Failures.** A region whose OCR raises, or that is too small or invalid to crop, keeps
  `text: null`, `text_source: "none"`, `routing.decision: "ocr"` and gets `"error": "<type>: <message>"`.
  The other regions are processed normally; `summary.ocr_failures` counts them.
- A skipped region (`EQUATION`, `FIGURE`, table without text) has `text: null`, `text_source: "none"`
  and no `error`.
- `summary.text_regions = native_regions + ocr_regions + ocr_failures`. `ocr_empty` (OCR ran or the
  crop was blank, no text found) is part of `ocr_regions`.
- Timings are returned by `run_document_ocr` (`engine_init_s`, `ocr_s`, `region_ocr_s` per region id,
  `total_s`) and printed by the tool, not stored, so the same input gives the same file.
- `validate_ocr_document(data)` checks the structure and returns a list of problems.
- Unusable input (missing or mismatching `pages.json` / `layout.json`, unreadable page image) or an
  engine that does not load raises a `PdfPipelineError` and leaves no `ocr.json`.

## 11. Provenance

Every region states its source in `text_source`, and the fields that go with it differ, so the two
kinds of text cannot be confused:

| `text_source` | `confidence` | `engine` | Extra fields |
|---|---|---|---|
| `"native"` | `null` | `null` | `native_blocks` (indexes into `pages.json`) |
| `"ocr"` | 0–1, or `null` without text | `"PaddleOCR"` | `raw_text`, `crop_bbox`, `lines` |
| `"none"` | `null` | `null` | `error` if OCR was attempted and failed |

`routing.reason` says why. The validator rejects a native region with a confidence and an OCR region
without an engine, lines or (when it has text) a confidence.

## 12. CER / WER

`AIEngine/pipeline/ocr_eval.py`:

```
CER = (S + D + I) / N        S substitutions, D deletions, I insertions, N characters of the ground truth
WER = the same over whitespace-separated words
```

- `edit_counts(reference, hypothesis)` returns `(S, D, I)` of a minimal edit script (Levenshtein).
- `cer(reference, hypothesis)`: whitespace is normalized on both sides (line breaks count as a space),
  case and punctuation count. The value can exceed 1. An empty reference raises `ValueError`.
- `wer(reference, hypothesis)`.
- `compare_ocr_documents(reference, hypothesis)` pairs the OCR'd regions of one `ocr.json` with the
  regions at the same place in another and sums edits and lengths over all pairs.

```powershell
python AIEngine/tools/evaluate_ocr.py temp/scan --reference temp/digital
```

Results on the fixtures (this machine, the models above, 180 DPI pages):

| Check | Regions | Reference characters | CER | WER | Mean confidence |
|---|---|---|---|---|---|
| `synthetic_scanned_text.pdf` (OCR) against `synthetic_digital.pdf` (native text) | 14 | 2220 | 0.0005 (1 edit) | 0.0032 (1 word of 317) | 0.99 |
| Bitmap paragraph of `synthetic_mixed.pdf` against its known text | 1 | 202 | 0.0000 | 0.0000 | 0.99 |

**These numbers are engineering validation, not an accuracy claim.** The "scans" are clean 150 DPI
renderings of Helvetica text: no skew, noise, blur, bleed-through, stains, small fonts, multi-column
gutters or mathematics. Real scanned papers will be worse, and nothing here measures by how much.

## 13. Command line

```powershell
# PDF: Phase 2 and Phase 3 run first (or are reused), then routing and OCR
python AIEngine/tools/inspect_ocr.py tests/fixtures/mixed/synthetic_mixed.pdf --output temp/mixed

# Save the image crops that were sent to the engine
python AIEngine/tools/inspect_ocr.py sample.pdf --output temp/sample --save-ocr-crops

# An existing working directory; also OCR tables without native text as raw lines
python AIEngine/tools/inspect_ocr.py temp/sample --ocr-tables

# Tests
python -m unittest discover -s tests/python -v
```

Options: `--output`, `--dpi`, `--layout-threshold`, `--ocr-padding` (default 8), `--ocr-tables`,
`--save-ocr-crops`, `--force`, `--device` (default `cpu`).

**Reuse.** For a PDF, Phase 2 is skipped when the working directory already has a valid `pages.json`
for the same file (SHA-256) and DPI, with lines and all page images. Phase 3 is skipped when
`layout.json` belongs to that `pages.json` and was made with the same threshold. `--force` runs both
again. The tool prints which was reused. A working directory as input is taken as it is.

**Debug crops.** With `--save-ocr-crops` each OCR'd region is also saved as
`<work_dir>/ocr_crops/page_NNN_region_NNN.png` (region = its position on the page, as in `region_id`).
Off by default; `ocr_crops/` is git-ignored.

Exit code 0, or 2 with a one-line `Error: ...` on stderr. PaddlePaddle prints its own start-up messages
on stderr; the summary is on stdout:

```
Pages: 2
Text regions: 14
Native text used: 0
OCR regions: 14
OCR failures: 0
Average OCR confidence: 0.99
OCR time: 7.6s

Regions without text (figure, equation, table without native text): 1
OCR regions without any text found: 0
Engine: PaddleOCR (PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec), device CPU, loaded
Phase 2 pages: reused; Phase 3 layout: reused
OCR: ...\metadata\ocr.json
Time: engine init 4.56s, OCR 7.65s (slowest region 1.83s), total 12.28s
```

Behaviour by kind of PDF (fixtures, real models):

| PDF | Text regions | Native | OCR | Failures | Engine loaded |
|---|---|---|---|---|---|
| `digital/synthetic_digital.pdf` (2 pages) | 15 | 15 | 0 | 0 | no |
| `multicolumn/synthetic_two_column.pdf` | 3 | 3 | 0 | 0 | no |
| `scanned/synthetic_scanned_text.pdf` (2 pages, no text layer) | 14 (+1 table skipped) | 0 | 14 | 0 | yes |
| the same with `--ocr-tables` | 15 | 0 | 15 | 0 | yes |
| `mixed/synthetic_mixed.pdf` | 4 | 3 | 1 | 0 | yes |
| `scanned/synthetic_scanned.pdf` (checkerboard bitmap) | 0 | 0 | 0 | 0 | no |

An image-only PDF, for which V1 still answers "No text found in PDF. Make sure it is not a scanned
image.", gets its text here. V1 was deliberately not changed; connecting the two is a later phase.

## 14. Performance (reference, this machine, CPU)

12 logical cores, 16 GB RAM, 180 DPI page images, models already downloaded, single runs:

| | |
|---|---|
| Engine init, new process | 4.6 s (includes importing PaddlePaddle) |
| Engine init, PaddlePaddle already imported by the layout model in the same process | 0.6 s |
| `synthetic_scanned_text.pdf`: 14 regions, 33 lines | 7.6–8.3 s OCR over three runs, about 0.55 s per region |
| Slowest region (8-line paragraph, 1128 x 299 px crop) | 1.8–2.0 s |
| One-line region (heading, caption) | 0.07–0.45 s |
| `synthetic_mixed.pdf`: 1 region, 3 lines | 0.9 s |
| Digital PDFs | 0 s; engine not loaded; routing takes about 10 ms |

- The models are loaded once per process and reused for every region and page
  (`engine.load_count` is checked in the tests). A long-running worker pays the init once.
- Time grows with the number of text lines: roughly 0.25 s per line of a paragraph.
- On a scanned page the layout model of Phase 3 (about 4–7 s per page) costs as much as the OCR.
- First use additionally downloads 12.3 MB of models.

## 15. Limitations

- **Tested on synthetic pages only.** The repository has no real scanned papers. The CER figures show
  that the plumbing works, nothing more (section 12).
- Routing depends on Phase 3. Text the layout model does not find, or finds as a `FIGURE`, is not read.
  A bitmap of text with a frame around it may well be detected as a figure.
- English/Latin recognition model by default. Other scripts need another model (section 5); the language
  is not detected.
- No handling of skewed, rotated or vertical text: line polygons are cut as upright rectangles, and no
  orientation or unwarping model is used.
- Mathematics inside a text region is OCR'd as ordinary characters and will come out wrong; equation
  regions are skipped entirely.
- A scanned table yields nothing unless `--ocr-tables` is given, and then only text lines.
- Native text is trusted when it has letters or digits and few undecodable characters. A text layer
  that is readable but wrong (a bad OCR layer embedded by a scanner, or a font with a wrong character
  map) is kept as "native"; it is not compared with the pixels.
- A region that mixes native text and pictured text is treated as native as soon as one usable native
  line lies in it; the pictured part is then lost.
- A line is assigned by a 0.5 area share. Inline formula regions and overlapping regions can make a
  line count for two regions, or a line that straddles two regions equally count for neither.
- Region confidence is the engine's own estimate. It was not calibrated, and no confidence threshold
  is applied: low-confidence text is kept and marked, not removed.
- Words are not located, only lines.
- Text is per region. There is no reading order across regions, no de-hyphenation and no paragraph
  joining.
- CPU only, one region at a time; no batching across regions, no parallel pages.
- The oneDNN workaround (section 3) is tied to `paddlepaddle` 3.3.1; `ocr.json` records package
  versions but not a hash of the model weights.
- Not connected to the web application, V1 or the database.

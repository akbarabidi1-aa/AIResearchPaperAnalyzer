# Phase 5 — Table structure extraction

Phase 5 turns every `TABLE` region of Phase 3 into a machine-readable table:

```
Phase 3 TABLE region
    ->  structure      rows, columns, cells, merged cells
    ->  cell text      from the PDF where it has text, from OCR otherwise
    ->  headers, caption
    ->  tables.json
```

```json
{ "headers": ["Method", "Accuracy", "F1"],
  "rows": [["ResNet", "92.10", "91.40"], ["ViT", "94.70", "94.10"]] }
```

Tables only: no formula-to-LaTeX, no figure or chart understanding, no section or reading-order
reconstruction, no summarization, no LLM. It runs alongside V1 and Phases 2–4; `ai_engine.py`, the
ASP.NET application and the database are unchanged and do not call it.

## 1. Chosen table extractor

Two parts, chosen per table:

| Part | Used for | What it is |
|---|---|---|
| **Native geometry** | Tables whose native text forms a regular grid | About 80 lines of Python in `tables.py` over the text lines Phase 2 already stores. No model, no new package |
| **SLANet_plus** | Everything else: merged cells, irregular layouts, scans | PaddleOCR's table structure recognition model, through its standalone `TableStructureRecognition` module |

Cell text never comes from the structure model. It comes from the PDF (native) or from the Phase 4
OCR engine.

## 2. Why it was selected

The environment was inspected first: `paddleocr` 3.7.0 and `paddlepaddle` 3.3.1 (CPU) from Phases 3–4
already ship the table modules. SLANet_plus was then **run on the fixtures before anything was built**:
it returned the right rows and columns for all six tables (bordered, borderless, merged header; digital
and scanned), including the `colspan="2"` of the merged header, in about 0.3 s per table on CPU.

| Candidate | Decision | Reason |
|---|---|---|
| **SLANet_plus** (PaddleOCR module) | **Used** | Already installed stack, 7.6 MB, CPU, works on images so scans and digital pages are handled alike, reports `rowspan`/`colspan` |
| Native geometry (own code) | **Used** | Digital tables need no model: one stored text line per cell is enough for a regular grid |
| PP-StructureV3 / `TableRecognitionPipelineV2` | Not used | A whole pipeline (table classification, wired and wireless structure models, cell detection, its own layout and OCR) with models of several hundred MB; it would duplicate Phases 3 and 4. Not downloaded, not measured |
| pdfplumber, Camelot | Not installed | Both read the PDF file itself, and only digital PDFs. The later phases work from `pages.json`, not from the PDF, and the geometry they would provide is the small part already written here. Camelot also needs extra system dependencies |
| Table Transformer | Not installed | Needs PyTorch and `transformers`: a second deep-learning framework next to PaddlePaddle, for a job the installed stack already does |

No conflicting Paddle version and no new Python package was installed.

## 3. Licence

SLANet_plus weights, PaddleOCR, PaddleX and PaddlePaddle: Apache-2.0. The native geometry code is part
of this repository. PyMuPDF from Phase 2 remains AGPL-3.0.

## 4. Dependency versions

Nothing was installed for Phase 5. `AIEngine/requirements.txt` only gained a comment.

| | |
|---|---|
| Python | 3.10.9 (Windows 10) |
| `paddleocr` | 3.7.0 |
| `paddlepaddle` | 3.3.1, CPU build |
| `paddlex` | 3.7.2 |
| `Pillow` | 12.3.0 (crops, debug images) |
| Structure model | `SLANet_plus`, 7.6 MB on disk, downloaded to `~/.paddlex/official_models/` on first use |
| OCR models | those of Phase 4 (`PP-OCRv5_mobile_det`, `en_PP-OCRv5_mobile_rec`), the same engine instance |
| Device | CPU. `--device gpu:0` is passed through but was not tested |

Unlike the OCR detection model (Phase 4), SLANet_plus runs with oneDNN on this PaddlePaddle build.

### Code layout

| File | Role |
|---|---|
| `AIEngine/pipeline/tables.py` | `TableExtractor`, `TableStructureModel` interface, `SLANetPlusStructureModel`, native geometry, cell filling, headers, captions, `run_document_tables`, `validate_tables_document` |
| `AIEngine/pipeline/tables_visualize.py` | `draw_table`: debug image |
| `AIEngine/pipeline/tables_eval.py` | `evaluate_table`, `evaluate_tables` |
| `AIEngine/tools/inspect_tables.py` | Command-line tool |
| `AIEngine/tools/evaluate_tables.py` | Accuracy against ground truth |
| `tests/python/test_tables.py` | Tests |
| `tests/fixtures/make_table_fixtures.py`, `tests/fixtures/tables/` | Fixtures and their ground truth |

Only `SLANetPlusStructureModel` imports PaddleOCR. The rest uses:

```python
class TableStructureModel:
    name: str
    def load(self)                       # loads once
    def recognize(self, image) -> TableStructure      # cells: row, column, row_span, col_span, bbox (pixels)

class TableExtractor:
    def __init__(self, structure_model, ocr_engine)   # a Phase 4 OCREngine
    def load(self)                                    # optional: both models are loaded lazily otherwise
    def extract(self, region, page, open_page_image, mode="auto", ocr_lines=None)
    def extract_native(self, region, page)            # geometry + native text
    def extract_image(self, region, page, page_image, native_lines=None, ocr_lines=None)   # hybrid / vision
```

```python
import sys; sys.path.insert(0, "AIEngine")
from pipeline import PaddleOCREngine, SLANetPlusStructureModel, TableExtractor, run_document_tables

extractor = TableExtractor(SLANetPlusStructureModel(), PaddleOCREngine())    # keep one instance
result = run_document_tables("temp/sample", extractor)      # needs pages.json, layout.json, ocr.json
```

To use another structure model, subclass `TableStructureModel` and implement `_load_model` and
`_recognize`. `parse_structure_tokens` converts HTML structure tokens with one box per `<td>` (the
output format of the SLANet family) into a `TableStructure`.

## Path selection

Only `TABLE` regions of `layout.json` are processed, each on its own; pages are not searched for tables.
For each region, the native text lines inside it are collected with the Phase 4 rule (a line belongs to
the region when at least half of its area is inside).

```
usable native text in the region?
 ├─ no  ............................................  vision   structure model + OCR
 └─ yes: do the lines form a regular grid?
      ├─ yes  ......................................  native   geometry + native text      (no model)
      └─ no   ......................................  hybrid   structure model + native text (no OCR)
```

`--table-mode native|hybrid|vision` forces a path (for comparison and debugging).

## 5. Digital / native path

`parse_native_table(lines)`, in PDF points, on the lines of `pages.json`:

1. **Rows**: lines on the same height (each one's vertical centre lies inside the other), top to bottom.
2. **Columns**: gaps of the horizontal projection. Texts closer than 2 pt are the same column.
3. Each line goes to its row and column; texts sharing a cell are joined with a space.
4. Every grid position becomes a cell (empty text where nothing is), with the box
   `column width x row height`.

Ruling lines play no part, so a bordered and a borderless table are handled the same way. The result
is only **trusted for a regular grid**. It is rejected, and the hybrid path used instead, when:

| Check | Typical cause |
|---|---|
| Fewer than 2 rows or 2 columns | Not a table, or one line per row |
| Two texts in the same row and column | A wide merged cell bridges two columns |
| A text does not line up with its column (left edge, right edge and centre all more than 3 pt off the column's median; checked when the column has at least 3 texts) | A header centred over several columns |
| Less than 50 % of the grid positions hold text | Cell text wrapped over several lines |

`structure_confidence` of a native table is `filled positions / all positions x (1 - misaligned texts / texts)`.
The native path never produces a merged cell: it cannot tell one from a wide or shifted cell, so it
hands such tables to the model instead of guessing. Nothing is OCR'd and no model is loaded.

## 6. Scanned / vision path

```
TABLE region  ->  crop of the Phase 2 page image (region + 8 px, as in Phase 4)
              ->  SLANet_plus: HTML structure tokens + one box per cell
              ->  parse_structure_tokens: grid positions, rowspan / colspan
              ->  text lines of the same crop from the Phase 4 OCREngine
              ->  fill_structure: each line into the cell that holds most of it
```

- The PDF is not rendered again; the crop comes from `pages/page_NNN.png`.
- **OCR reuse.** The OCR engine is the Phase 4 `OCREngine`, the same instance the tool used for
  Phase 4, so the OCR models are loaded once. If `ocr.json` already holds lines for the table region
  (Phase 4 run with `--ocr-tables`), those lines are used and the table is not OCR'd again.
- A line goes to the cell containing the largest share of the line's area, if that is at least 30 %.
  Lines that fit no cell are **not dropped silently**: their text is listed in `unassigned_text`.
- `structure_confidence` is the model's structure score times the share of lines that found a cell.
- Each cell keeps the character-weighted mean confidence of its OCR lines.

## 7. Hybrid path

Vision structure + native text: the structure model reads the grid (and the merged cells) from the
image, and the cells are filled with the **native lines of the PDF**, mapped to page pixels with the
scale stored in `pages.json`. Nothing is OCR'd, so the text is exact.

It is used automatically for a table that has native text but is not a regular grid, which is exactly
where geometry alone would have to guess. In the fixtures that is the table with a merged header cell:
the native path rejects it (the centred `Accuracy` does not line up with a column), the model reports
`colspan="2"`, and all 20 positions get their native text.

## 8. Table and cell schema

Written to `<work_dir>/metadata/tables.json`. **`pages.json`, `layout.json` and `ocr.json` are not
modified.** UTF-8, no image data.

```json
{
  "schema_version": 1,
  "extractor": {
    "name": "native-geometry + SLANet_plus",
    "structure_model": { "name": "SLANet_plus", "provider": "PaddleOCR", "version": "3.7.0",
                         "framework": "paddlepaddle 3.3.1", "device": "cpu" },
    "ocr_engine": { "name": "PaddleOCR", "version": "3.7.0", "device": "cpu", "...": "as in ocr.json" }
  },
  "settings": { "mode": "auto", "padding_px": 8 },
  "source": { "pages_json": "metadata/pages.json", "layout_json": "metadata/layout.json",
              "ocr_json": "metadata/ocr.json", "source_pdf": "synthetic_tables.pdf", "source_sha256": "..." },
  "coordinate_system": { "bbox_format": "x0,y0,x1,y1", "origin": "top-left", "...": "as in layout.json" },
  "pages": [
    {
      "page_number": 3,
      "tables": [
        {
          "table_id": "p003_t001",
          "region_id": "p003_r004",
          "page_number": 3,
          "image_bbox": [222.0, 535.0, 1201.0, 812.0],
          "pdf_bbox": [88.8, 214.0, 480.2, 324.8],
          "caption": "Table 3. Accuracy and size of the ablated models.",
          "caption_region_id": "p003_r003",
          "n_rows": 5,
          "n_columns": 4,
          "header_rows": [0, 1],
          "headers": ["Model", "Accuracy Top-1", "Accuracy Top-5", "Params"],
          "rows": [["Small", "81.20", "95.10", "5.3"], ["Base", "86.50", "97.42", "21.8"], ["Large", "88.10", "98.00", "86.0"]],
          "cells": [
            { "row": 0, "column": 1, "row_span": 1, "col_span": 2, "text": "Accuracy", "text_source": "native",
              "confidence": null, "is_header": true,
              "image_bbox": [561.0, 530.0, 931.0, 594.0], "pdf_bbox": [224.3, 212.0, 372.3, 237.6] }
          ],
          "source": "hybrid",
          "structure_method": "model",
          "structure_confidence": 0.9999,
          "notes": [],
          "unassigned_text": []
        }
      ]
    }
  ],
  "summary": { "pages": 3, "table_regions": 3, "tables_extracted": 3, "native": 2, "vision": 0, "hybrid": 1,
               "cells": 63, "with_headers": 3, "with_caption": 3, "extraction_failures": 0 }
}
```

- `table_id` is `p<page>_t<index>`. `region_id`, `page_number`, `image_bbox` and `pdf_bbox` are copied
  from the layout region unchanged.
- `cells` holds every cell once, ordered by row and column. A merged cell is one entry at its top-left
  position with its spans. For the native path a cell box is the column's width by the row's height;
  for the model paths it is the model's cell box, which may reach a few pixels beyond the table region
  (the crop is padded). Both coordinate systems are given, as in Phase 3.
- `rows` are the **body** rows as lists of `n_columns` strings (all rows when `headers` is `null`).
  A position covered by a merged cell holds `""`. `table_grid(table)` returns all rows including the
  header rows.
- **Numeric fidelity.** Every cell text is a string exactly as read: `"94.70"` stays `"94.70"`,
  `"50,000"` stays `"50,000"`. No number is parsed, rounded or reformatted, and there is no numeric
  field. Whitespace is cleaned as in Phase 4 (outer removed, inner runs to one space).
- A page without table regions has `"tables": []`.
- **Failures.** A table that cannot be extracted (model error, malformed structure, region too small,
  no text found) stays in the file with its identity, boxes and caption, `source: null`, no cells, and
  `"error": "<type>: <message>"`. The other tables are processed normally.
- Timings are returned by `run_document_tables` and printed by the tool, not stored.
- `validate_tables_document(data)` checks the structure (cells inside the grid, no two cells on one
  position, `headers`/`header_rows`/`rows` consistent) and returns a list of problems.
- Missing or mismatching earlier artifacts raise a `TableInputError` and leave no `tables.json`.

## 9. Headers

`detect_header_rows(grid)`. The first row is **not** assumed to be a header.

- A cell is *numeric* when it has at least one digit and no letter (`94.70`, `1,024`, `-3.5%`, `12 ± 0.4`).
- Header rows are the leading rows without any numeric cell, **provided** every row after them has at
  least one numeric cell, and there are at most 3 of them.
- Otherwise `header_rows` and `headers` are `null` and `rows` holds all rows: a table of words only, a
  table whose first row already has numbers, or one whose body has a row without numbers.

`headers` has one label per column. With several header rows the labels are joined top-down, and a
merged header cell labels every column it covers: `"Accuracy Top-1"`, `"Accuracy Top-5"`. The original
cells, with `is_header: true`, stay in `cells`.

The structure model's own `<thead>` marking is not used: it did not emit one for the fixtures.

## 10. Merged cells

`row_span` and `col_span` are taken from the structure model's tokens (`rowspan="2"`, `colspan="2"`)
and kept in `cells`. They are available on the **hybrid and vision** paths. The native path always
gives 1/1 and sends tables it cannot explain to the hybrid path; no span is ever inferred from text
positions. The fixtures cover a column span only; **row spans are parsed and unit-tested with a fake
model but not exercised with the real one**.

## 11. Caption matching

`associate_captions(regions)`, geometry only, per page, on `layout.json` boxes in PDF points:

1. A `CAPTION` region can belong to a table when they overlap horizontally by at least 50 % of the
   narrower one and are at most 36 pt apart vertically (above or below).
2. A caption is a candidate only for the `TABLE` or `FIGURE` region nearest to it, so a figure's
   caption is not given to a table underneath.
3. A table takes its nearest candidate.
4. In steps 2 and 3, if the runner-up is less than 1.5 times as far away, the case is ambiguous and
   **nothing is assigned**.

The caption text is what Phase 4 stored for that region in `ocr.json` (native or OCR). Its wording is
not inspected: a caption is not required to start with "Table". No reading order is involved.

## 12. Provenance

| `source` | Structure (`structure_method`) | Cell text (`cells[].text_source`) | Cell `confidence` |
|---|---|---|---|
| `"native"` | `"geometry"`: positions of native lines | `"native"` | `null` |
| `"hybrid"` | `"model"`: SLANet_plus on the image | `"native"` | `null` |
| `"vision"` | `"model"`: SLANet_plus on the image | `"ocr"` | 0–1 |

Empty cells have `text_source: null`. `notes` lists why a native grid is doubtful (only non-empty when
`--table-mode native` forced it). `extractor` records the model and engine versions.

## 13. Metrics

`AIEngine/pipeline/tables_eval.py`, against ground truth with a known grid:

| Metric | Definition |
|---|---|
| Row / column count | expected vs. extracted, and whether they are equal |
| Exact cell accuracy | positions whose text is identical / positions of the expected grid. Whitespace-normalized only: `94.7` does not match `94.70` |
| Cell text accuracy | mean over the positions of `1 - min(1, CER)` |
| Structure accuracy | cells with the same row, column, row span and column span in both / cells in either |
| Header rows, caption | exact match |

```powershell
python AIEngine/tools/evaluate_tables.py temp/tables --truth tests/fixtures/tables/synthetic_tables.json
```

Fixtures (`tests/fixtures/tables/`, 3 tables, 64 grid positions; ground truth in `synthetic_tables.json`):

| Page | Table | Covers |
|---|---|---|
| 1 | 4 x 4, ruled grid, caption above | bordered, digital, captioned |
| 2 | 7 x 4, no lines at all, caption below, values such as `94.70` and `80.00` | borderless, multi-row, numeric |
| 3 | 5 x 4, ruled, `Accuracy` spanning two columns over `Top-1` / `Top-5` | merged header cell, two header rows |
| `synthetic_tables_scanned.pdf` | the same three pages as 150 DPI bitmaps | scanned |

Results with the real models (this machine):

| Input | Paths | Rows | Columns | Exact cells | Structure | Headers | Captions |
|---|---|---|---|---|---|---|---|
| `synthetic_tables.pdf` | native, native, hybrid | 3/3 | 3/3 | 64/64 | 1.000 | 3/3 | 3/3 |
| `synthetic_tables_scanned.pdf` | vision x 3 | 3/3 | 3/3 | 64/64 | 1.000 | 3/3 | 3/3 |
| `synthetic_tables.pdf --table-mode hybrid` | hybrid x 3 | 3/3 | 3/3 | 64/64 | 1.000 | 3/3 | 3/3 |
| `synthetic_tables.pdf --table-mode vision` | vision x 3 | 3/3 | 3/3 | 64/64 | 1.000 | 3/3 | 3/3 |

**These are engineering checks, not benchmark results.** Three small, clean, generated tables in one
font say that the code paths work; they say nothing about accuracy on real papers.

## 14. Command line

```powershell
# PDF: earlier phases run first (or are reused), then the tables
python AIEngine/tools/inspect_tables.py tests/fixtures/tables/synthetic_tables.pdf --output temp/tables

# Debug overlays
python AIEngine/tools/inspect_tables.py sample.pdf --output temp/sample --visualize-tables

# An existing working directory, forcing one path
python AIEngine/tools/inspect_tables.py temp/sample --table-mode hybrid

# Tests
python -m unittest discover -s tests/python -v
```

Options: `--output`, `--dpi`, `--layout-threshold`, `--table-mode auto|native|vision|hybrid`,
`--visualize-tables`, `--force`, `--device`.

**Reuse.** Phase 2 and Phase 3 are reused as in `inspect_ocr.py`. Phase 4 is reused when `ocr.json`
belongs to the current `layout.json`, and run with its defaults otherwise. `--force` runs all three
again. The tool prints what was reused.

**Visualization.** With `--visualize-tables` each extracted table is drawn on a copy of its page as
`<work_dir>/tables_debug/page_NNN_table_NNN_debug.png`: the table box (green), the row and column
boundaries (orange, halfway between neighbouring cells), and every cell box (blue; red for header
rows). Off by default; `tables_debug/` is git-ignored.

Exit code 0, or 2 with a one-line `Error: ...` on stderr. Summary on stdout:

```
Pages: 3
Table regions: 3
Tables extracted: 3

Native: 2
Vision: 0
Hybrid: 1

Cells: 63
Extraction failures: 0
Time: 4.6s

  p001_t001: native, 4 x 4, headers 1 row(s), merged cells 0, confidence 1.00, caption yes
  p002_t001: native, 7 x 4, headers 1 row(s), merged cells 0, confidence 1.00, caption yes
  p003_t001: hybrid, 5 x 4, headers 2 row(s), merged cells 1, confidence 1.00, caption yes
Tables with headers: 3; with caption: 3
Structure model: SLANet_plus, loaded; OCR engine: PaddleOCR, not loaded: no text needed OCR
Phase 2 pages: reused; Phase 3 layout: reused; Phase 4 OCR: reused
Tables: ...\metadata\tables.json
Table time: model init 4.14s, extraction 0.46s (slowest table 0.46s)
```

`Time` is the whole table step including model loading; `Cells` counts cell entries (a merged cell once).

## 15. Performance (reference, this machine, CPU)

12 logical cores, 16 GB RAM, 180 DPI page images, models already downloaded, single runs, earlier
phases reused:

| | |
|---|---|
| Native table | A few milliseconds; no model loaded |
| Structure model init, new process | 4.0–4.6 s (includes importing PaddlePaddle) |
| Structure model init, PaddlePaddle already imported in the process | 0.2 s |
| Hybrid table (structure only) | 0.4–0.5 s |
| Vision table (structure + OCR of 16–28 cells) | 1.3–1.8 s |
| `synthetic_tables.pdf`: 2 native + 1 hybrid | 4.6 s, of which 4.1 s model init |
| `synthetic_tables_scanned.pdf`: 3 vision | 9.2 s, of which 4.6 s model init |
| `synthetic_digital.pdf`: 1 native table | 0.0 s; neither model loaded |

Both models are loaded once per process and only when a table needs them (`load_count` is checked in
the tests). A document of regular digital tables loads neither. First use downloads 7.6 MB.

## 16. Known limitations

- **Tested on synthetic tables only**: three small tables, one font, clean renders. No real paper.
- The table must be found by Phase 3. A table the layout model misses, or boxes too tightly or too
  widely, is not repaired here. Tables spanning pages are separate tables.
- **Native path**: regular grids only. Cell text wrapped over several lines becomes several rows (or
  sends the table to the hybrid path through the fill check); numbers aligned on the decimal point and
  headers centred over a left-aligned column fail the alignment check and go to the hybrid path as well,
  which costs a model call but not accuracy of the text. Two columns closer than 2 pt merge.
- The native path works on Phase 2 **lines**, not words. A row written as one text line across
  several columns cannot be split; such a table appears as "not a grid" and goes to the hybrid path,
  where a line covering several cells is put in the one holding most of it.
- **SLANet_plus** is a small model. Large, dense or rotated tables, nested tables and tables with many
  merged cells are expected to be harder; this was not measured. Its cell boxes are approximate.
- Row spans come only from the model and were not exercised with the real model.
- In the vision path, OCR lines are matched to cells by overlap. Text of two neighbouring cells that
  the OCR detector joins into one line ends up in one cell.
- Header detection is a rule about numbers. A table of words has `headers: null`; a table whose header
  cells are numbers (years) has `headers: null`; scientific notation such as `3e-4` counts as text and
  can stop a header from being recognized, or make a body row look like a header row.
- `headers` joins multi-row headers with a space; the hierarchy itself is only in `cells`.
- Captions are matched by position only, on one page, within 36 pt. A caption on the previous page or
  beside the table is not found, and a table with a caption above and a note below at similar
  distances gets none.
- Cell confidences are the OCR engine's own, uncalibrated; no threshold is applied.
- No cell types, units or parsed numbers; no footnote marks handling; no export to CSV or HTML.
- CPU only, one table at a time.
- Not connected to the web application, V1 or the database.

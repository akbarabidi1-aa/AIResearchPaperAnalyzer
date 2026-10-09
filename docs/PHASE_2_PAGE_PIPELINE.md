# Phase 2 — Page pipeline (rendering + native page extraction)

Phase 2 turns a PDF into:

1. one PNG image per page,
2. the native (embedded) text of each page as blocks with bounding boxes,
3. page metadata, including the PDF-to-pixel scale,
4. a per-page `has_text_layer` flag.

It runs **alongside** V1. `ai_engine.py`, the ASP.NET app and the database are unchanged and do not call it
yet. There is no OCR, layout detection, table, equation or figure handling in this phase.

## Why PyMuPDF

V1 uses pdfminer.six, which returns one text stream for the whole document: no page images, and no
per-page boxes without extra work. Later phases need both, in one consistent coordinate system. PyMuPDF
(MuPDF bindings) renders pages and extracts positioned text from the same page object, is fast, and
installs as a single wheel with no external programs.

- Package: `PyMuPDF` (in `AIEngine/requirements.txt`). Verified with 1.28.2 on Python 3.10.
- Import name: `pymupdf`. The old name `fitz` still works but prints a deprecation warning in recent
  versions; the pipeline imports `pymupdf` and falls back to `fitz` on old installs.
- Licence: PyMuPDF is AGPL-3.0 (or commercial). Check this before distributing the application.

```powershell
python -m pip install -r AIEngine/requirements.txt
python -c "import fitz; print(fitz.__doc__)"
```

## Code layout

| File | Role |
|---|---|
| `AIEngine/pipeline/common.py` | Error types, `open_pdf` (missing / empty / corrupt / encrypted / zero-page checks) |
| `AIEngine/pipeline/render.py` | `render_pdf_pages(pdf_path, output_dir, dpi=180)` |
| `AIEngine/pipeline/native_extract.py` | `extract_native_pages(pdf_path)`, `page_has_text_layer(text)` |
| `AIEngine/pipeline/page_pipeline.py` | `process_pdf(pdf_path, work_dir, dpi=180)`: both steps + `pages.json`; `validate_pages_document` |
| `AIEngine/tools/inspect_pdf.py` | Thin command-line wrapper around `process_pdf` |
| `tests/python/test_page_pipeline.py` | Tests (standard-library `unittest`) |

## Example commands

```powershell
# Output goes to AIEngine/output/<name>-<hash>/
python AIEngine/tools/inspect_pdf.py tests/fixtures/digital/synthetic_digital.pdf

# Choose the working directory and resolution
python AIEngine/tools/inspect_pdf.py sample.pdf --output temp/sample --dpi 180

# Tests
python -m unittest discover -s tests/python -v
```

Output:

```
Pages: 2
Rendered: 2
Pages with native text: 2
Pages without native text: 0
Metadata: ...\metadata\pages.json
Time: open 0.003s, render 0.134s, extract 0.004s, total 0.161s
```

Exit code 0 on success, 2 with a one-line `Error: ...` on stderr when the PDF or the output directory
cannot be processed. From Python:

```python
import sys; sys.path.insert(0, "AIEngine")
from pipeline import process_pdf
result = process_pdf("sample.pdf", "temp/sample")     # result["document"] is the pages.json content
```

## Working directory

One directory per document:

```
<work_dir>/
  original.pdf          only with --copy-original / copy_original=True
  pages/
    page_001.png
    page_002.png
  metadata/
    pages.json
```

- Page images are named `page_NNN.png` with a 1-based, zero-padded page number.
- The command-line default is `AIEngine/output/<safe-stem>-<8 hex of the file's SHA-256>/`. The stem keeps only
  `[A-Za-z0-9_-]` (max 40 characters), so a raw file name never reaches the file system. The same file
  always maps to the same directory; different files with the same name map to different ones.
- Re-running into an existing directory replaces it: old `page_*.png` files and the old `pages.json` are
  removed first. If a run fails, no `pages.json` is left behind.
- `AIEngine/output/` is git-ignored.
- For the web application the intended layout is `Uploads/<guid>/...` with the GUID the server already
  generates. That wiring is **not** done in Phase 2 (no controller or database change).

## Render pipeline

`render_pdf_pages(pdf_path, output_dir, dpi=180)`:

1. Open the PDF (`open_pdf`).
2. For every page: `page.get_pixmap(matrix=Matrix(dpi/72, dpi/72), alpha=False)` and save as PNG (RGB, white
   background).
3. Return one dict per page: `page_number`, `image_path` (absolute), `width`, `height`, `dpi`, plus
   `pdf_width`, `pdf_height`, `image_width`, `image_height`, `scale_x`, `scale_y`.

180 DPI is the default (letter page: 1530 x 1980 px): enough detail for later layout detection and OCR
without very large files. Allowed range: 36–600.

Failures raise an exception; a failed page is never skipped:

| Situation | Exception |
|---|---|
| Path does not exist | `PdfNotFoundError` |
| 0-byte file, not a PDF, corrupt, password-protected | `InvalidPdfError` |
| PDF with no pages | `EmptyPdfError` |
| Output directory cannot be created / written | `OutputDirectoryError` |
| A page fails to render or save | `PageRenderError` (has `page_number`) |
| Native extraction fails on a page | `PageExtractionError` (has `page_number`) |

All derive from `PdfPipelineError`.

## Native extraction pipeline

`extract_native_pages(pdf_path)` returns, per page:

```json
{
  "page_number": 1,
  "width": 612.0,
  "height": 792.0,
  "has_text_layer": true,
  "text": "...",
  "blocks": [
    { "block_index": 0, "bbox": [72.0, 34.9, 353.8, 56.9], "text": "...", "block_type": "text" }
  ]
}
```

- `text` is `page.get_text("text")`: the full native text of the page.
- `blocks` come from `page.get_text("blocks")`. Only text blocks are kept; image blocks are ignored.
- Every block is `block_type: "text"`. Nothing is classified (no headings, tables, figures, equations).
- Blocks are in PDF content-stream order. Reading order is **not** reconstructed.
- A block is MuPDF's grouping of nearby lines, usually a paragraph. It is not a semantic unit: a heading
  and the paragraph under it are often one block.
- Since Phase 4 every block also has `lines`: `[{ "bbox": [x0, y0, x1, y1], "text": "..." }]`, from
  `page.get_text("dict")` on the same text page. Phase 4 uses them to give each layout region its own
  text. Nothing else in a block changed; a `pages.json` written before Phase 4 has no `lines` and must
  be regenerated before running Phase 4.

## Coordinate system

All boxes and page sizes use **PyMuPDF page coordinates**:

| Property | Value |
|---|---|
| Origin | Top-left corner of the page **as displayed** (after `/Rotate` and CropBox) |
| x axis | Increases to the right |
| y axis | Increases **downward** |
| Unit | PDF point = 1/72 inch |
| Box format | `[x0, y0, x1, y1]`, `x0 < x1`, `y0 < y1` (top-left and bottom-right corners) |
| Page width / height | Same unit (points), of the displayed page |
| Normalisation | None. Values are absolute points, not 0–1 |

Notes:

- This differs from the raw PDF convention (origin bottom-left, y up). PyMuPDF already flips it.
- Rotated pages: PyMuPDF reports text positions in the *unrotated* page. The extractor applies
  `page.rotation_matrix`, so boxes always match the displayed page and therefore the rendered image.
  A test renders a rotated page and checks this.
- Boxes are clipped to the page rectangle; a block that lies completely outside the page is dropped.

## PDF-to-image scaling

Per page:

```
scale_x = image_width  / pdf_width
scale_y = image_height / pdf_height

pixel_x = x * scale_x          x = pixel_x / scale_x
pixel_y = y * scale_y          y = pixel_y / scale_y
```

`pipeline.pdf_bbox_to_pixels(bbox, scale_x, scale_y)` does the forward mapping. Pixel coordinates have the
same origin (top-left) and direction (y down) as the page coordinates, so no flip is needed.

The scales are computed from the real image size. Because an image has whole pixels, they can differ
slightly from `dpi / 72` (for example 850 px / 612 pt = 1.3889 at 100 DPI on both axes, but an odd page size
can give slightly different x and y values). Always use the stored `scale_x` / `scale_y`, not `dpi / 72`.

Validation in the tests: for the two-column fixture, the digital fixture and a rotated odd-sized page,
every block box is scaled to pixels and compared with the rendered image. Each box must contain dark
(text) pixels, and no dark pixel may lie outside all boxes (2 px tolerance for anti-aliasing).

## `has_text_layer` heuristic

`page_has_text_layer(text)` is true when the native text of the page has **both**:

- at least **20 letters or digits** (`str.isalnum`), and
- at least **3 words of two or more letters**.

Whitespace, punctuation and undecodable glyphs (U+FFFD) do not count. Effect:

| Page content | Result |
|---|---|
| Normal digital page | `true` |
| Image-only (scanned) page | `false` |
| Only a page number, `Page 3 of 10`, a `DRAFT` stamp, digits only | `false` |
| Scan that already carries a hidden OCR text layer | `true` (it has usable text) |

The flag is per page, so a document can mix both kinds. `document.pages_with_text_layer` and
`pages_without_text_layer` give the totals. A `false` page is the signal for OCR in a later phase; Phase 2
does not run OCR. The thresholds are `MIN_TEXT_CHARS` and `MIN_TEXT_WORDS` in `native_extract.py`.

## `pages.json` schema

UTF-8, no BOM, non-ASCII characters written as-is.

```json
{
  "schema_version": 1,
  "document": {
    "source_pdf": "synthetic_digital.pdf",
    "source_sha256": "542b3ffc...",
    "page_count": 2,
    "dpi": 180,
    "pages_with_text_layer": 2,
    "pages_without_text_layer": 0,
    "coordinate_system": {
      "origin": "top-left",
      "y_axis": "down",
      "unit": "pt",
      "bbox_format": "x0,y0,x1,y1",
      "pixel_mapping": "pixel_x = x * scale_x; pixel_y = y * scale_y"
    }
  },
  "pages": [
    {
      "page_number": 1,
      "pdf_width": 612.0,
      "pdf_height": 792.0,
      "image_width": 1530,
      "image_height": 1980,
      "scale_x": 2.5,
      "scale_y": 2.5,
      "dpi": 180,
      "has_text_layer": true,
      "text": "...",
      "blocks": [
        { "block_index": 0, "bbox": [72.0, 34.9, 353.8, 56.9], "text": "...", "block_type": "text" }
      ],
      "image_path": "pages/page_001.png"
    }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `document.source_pdf` | string | File name only, no directory |
| `document.source_sha256` | string | Hash of the PDF bytes |
| `document.page_count` | int | Equals `pages.length` |
| `pages[].page_number` | int | 1-based, consecutive |
| `pages[].pdf_width`, `pdf_height` | number | Points |
| `pages[].image_width`, `image_height` | int | Pixels |
| `pages[].scale_x`, `scale_y` | number | Pixels per point |
| `pages[].has_text_layer` | bool | See heuristic |
| `pages[].text` | string | Full native page text, may be empty |
| `pages[].blocks[].bbox` | 4 numbers | Points, inside the page |
| `pages[].blocks[].block_type` | string | Always `"text"` in Phase 2 |
| `pages[].blocks[].lines[]` | `{bbox, text}` | The lines of the block, same coordinates. Added in Phase 4; joined with line breaks they give the block's `text` |
| `pages[].image_path` | string | Relative to the working directory, forward slashes |

`validate_pages_document(data)` checks this structure and returns a list of problems (empty when valid).
`process_pdf` runs it before writing the file. The same input always produces a byte-identical file;
timings are returned by `process_pdf` and printed by the tool but are not stored in `pages.json`.

## Performance (reference, this machine)

Measured with `inspect_pdf.py` at 180 DPI, single run each:

| PDF | Pages | Open | Render | Extract | Total |
|---|---|---|---|---|---|
| `synthetic_digital.pdf` | 2 | 0.001 s | 0.129 s | 0.004 s | 0.137 s |
| `synthetic_two_column.pdf` | 1 | 0.001 s | 0.061 s | 0.002 s | 0.066 s |
| `synthetic_scanned.pdf` | 1 | 0.001 s | 0.091 s | 0.000 s | 0.095 s |
| generated 20-page text PDF | 20 | 0.011 s | 1.657 s | 0.101 s | 1.778 s |

Rendering and PNG encoding dominate (roughly 0.06-0.09 s per page here). These are small synthetic files; real papers
with many images will be slower.

## Known limitations

- Not connected to the web application or the database yet; V1 results are unchanged.
- No OCR: image-only pages produce an image, empty text and `has_text_layer: false`.
- Blocks are generic and in content-stream order; multi-column reading order is not resolved. The
  two-column fixture yields separate left and right blocks, but this synthetic file does not prove real
  two-column papers are ordered correctly.
- Table cells come out as ordinary text (one block with one cell per line in the digital fixture).
- Image blocks, vector drawings and annotations are not listed in `blocks`.
- `has_text_layer` is a heuristic: a page with very little real text (a short title page) reads as `false`,
  and a text layer with wrong character encoding (garbled but alphabetic) reads as `true`.
- Password-protected PDFs are rejected.
- Pages are rendered one after another in a single process; all page text is held in memory and written to
  one `pages.json`. Fine for papers, not tuned for documents with thousands of pages.
- A damaged PDF that MuPDF can repair is processed with whatever pages it recovers.
- Only tested on the synthetic fixtures and generated PDFs; no real papers are in the repository.

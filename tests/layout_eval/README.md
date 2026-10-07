# Layout evaluation set (manual)

A small, hand-checked set of pages for judging layout detection quality by eye and with two simple
numbers (IoU and class accuracy). It is **not** a benchmark.

```
tests/layout_eval/
  pages/           put page images or short PDFs here (empty for now)
  ground_truth/    one JSON file per source document
  output/          scratch output, git-ignored
```

## What is here now

Only ground truth for two of the synthetic fixtures in `tests/fixtures/`:

| Ground truth | Source | Regions |
|---|---|---|
| `ground_truth/synthetic_digital.json` | `fixtures/digital/synthetic_digital.pdf` (2 pages) | 15 |
| `ground_truth/synthetic_two_column.json` | `fixtures/multicolumn/synthetic_two_column.pdf` | 3 |

The boxes were written by hand from the text positions in `make_synthetic_fixtures.py`.

**Do not quote results on these files as model accuracy.** They are clean, computer-generated pages with
one font and no noise; they only show that the pipeline and the coordinate mapping work. A meaningful
number needs real pages (below).

## Adding real pages (5–10 to start)

1. Choose pages you are allowed to redistribute: open-access papers under a licence that permits it
   (for example CC BY), your own documents, or synthetic pages. **Do not add copyrighted papers**, and do
   not download papers automatically. Pages you may not redistribute can be used locally but must not be
   committed.
2. Aim for variety: a title page, a two-column page, a page with a table, one with a figure and caption,
   one with display equations, a reference list, and a scanned page.
3. Put the file in `pages/`: a PNG/JPEG page image, or a PDF of a few pages. Keep each file under about 2 MB.
4. Write `ground_truth/<name>.json` (format below). To find coordinates, run
   `python AIEngine/tools/inspect_layout.py <pdf> --output tests/layout_eval/output/<name> --visualize-layout`
   and read pixel positions off `pages/page_NNN.png` in an image viewer. Draw the boxes from what **you**
   see on the page, not from the model's boxes.
5. Note the source and licence of every added file in the table above.

## Ground-truth format

```json
{
  "source": "../pages/my_paper_page3.png",
  "bbox_space": "image",
  "regions": [
    { "page_number": 1, "type": "TITLE", "bbox": [120, 80, 900, 130] },
    { "page_number": 1, "type": "FIGURE", "bbox": [150, 400, 1380, 1000] }
  ]
}
```

| Field | Meaning |
|---|---|
| `source` | PDF or page image, relative to the JSON file |
| `bbox_space` | `"image"`: pixels of the page image. `"pdf"`: PDF points (only for a PDF source). Origin top-left, y down |
| `regions[].page_number` | 1-based; `1` for an image |
| `regions[].type` | A canonical class: `TITLE TEXT TABLE FIGURE EQUATION CAPTION HEADER FOOTER PAGE_NUMBER LIST OTHER` |
| `regions[].bbox` | `[x0, y0, x1, y1]` |

For a PDF source with `"bbox_space": "image"`, the pixels are those of the page rendered at the default
180 DPI.

## Running the evaluation

```powershell
python AIEngine/tools/evaluate_layout.py tests/layout_eval/ground_truth/synthetic_digital.json
python AIEngine/tools/evaluate_layout.py tests/layout_eval/ground_truth/*.json
```

It prints one row per ground-truth region:

| Page | Ground-truth class | Ground-truth bbox | Predicted class | IoU | Correct class |
|---|---|---|---|---|---|
| 2 | TABLE | [72, 110, 426, 173] | TABLE | 0.97 | yes |
| 2 | CAPTION | [72, 183, 273, 197] | CAPTION | 0.85 | yes |

How a row is filled:

- Each ground-truth region is paired with the detected region on the same page that overlaps it most
  (`IoU = intersection / union`), if that IoU is at least `--iou-threshold` (default 0.5). A detected
  region is used for one ground-truth region only.
- **Predicted class** is the canonical class of that detection, `(none)` if nothing overlapped enough.
- **Correct class** is `yes` when a detection was paired and its class equals the ground-truth class.
- **Class accuracy** = rows with `yes` / all ground-truth rows. A missed region counts as wrong.

Detections that match no ground-truth region (false positives) are not listed or counted. The helpers
are `bbox_iou`, `match_regions` and `class_accuracy` in `AIEngine/pipeline/layout_eval.py`.

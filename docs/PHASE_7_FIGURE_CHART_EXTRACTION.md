# Phase 7 — Figure extraction and chart reading

Phase 7 turns every `FIGURE` region of Phase 3 into a figure entry, and reads the charts among them:

```
Phase 3 FIGURE region ("image" or "chart")
    ->  crop of the Phase 2 page image
    ->  caption (the CAPTION region above or below it) and figure number
    ->  text printed in the figure (native text, else the Phase 4 OCR engine)
    ->  figure type: chart / diagram / image / figure / unknown
    ->  charts only: axes, titles, tick labels, legend, series and their values
    ->  figures.json
```

```json
{ "figure_number": "Figure 3", "caption": "Accuracy of four models on the benchmark.",
  "figure_type": "chart",
  "chart_data": { "chart_type": "bar", "series": [ { "bars": [
      { "category": "ResNet", "value": 72.1, "value_source": "label", "visual_estimate": 72.3 } ] } ] } }
```

Extraction only. A figure keeps its image, its caption and the words printed in it. Nothing in this
phase describes or interprets what a figure shows: there is no field for prose, and no model that
could write any. No reading order, no section reconstruction, no summarization, no LLM. It runs
alongside V1 and Phases 2–6; `ai_engine.py`, the ASP.NET application and the database are unchanged
and do not call it.

## 1. Figure extraction strategy

**What the layout model already gives.** PP-DocLayoutV3 has 25 labels. Two of them become the
canonical class `FIGURE`: `image` and `chart`. So the model itself separates charts from other
figures; it has no label for "plot", "diagram" or "photo". On the fixtures of section 12 it labelled
all five charts `chart` and the three diagrams and the bitmap `image`. (`header_image` and
`footer_image` are `HEADER` / `FOOTER` and are not figures. `figure_title` is the caption label,
`vision_footnote` a note under a figure or table.)

**Per figure**, for `FIGURE` regions only (pages are not searched for figures):

1. `page_number`, `image_bbox`, `pdf_bbox`, `source_label` and the region id are copied unchanged.
2. Two boxes are cut from the Phase 2 page image: the region itself (saved, section 11) and the
   region grown by a padding (default 8 px, `--figure-padding`), which is what gets analysed. Layout
   boxes are tight and often cut through an axis title or a tick label; the padding keeps them whole.
   The padding never leaves the page and stops at a caption above or below the figure.
3. The caption is associated (section 6) and its figure number split off (section 7).
4. The text inside the figure is read (section 8).
5. The figure is typed (section 2). Only a chart goes on to the chart reader (section 3).

Every figure is handled on its own. One that fails keeps its entry with `status: "failed"`, an
`error` and, whenever it can be cut out, its crop. A chart that cannot be read is not a failed
figure: it keeps crop, caption and text, and its `chart_data.status` is `"failed"`.

### Code layout

| File | Role |
|---|---|
| `AIEngine/pipeline/figures.py` | `FigureExtractor` (`extract`, `classify`, `associate_caption`, `read_text`), captions, figure numbers, crops, `run_document_figures`, `validate_figures_document` |
| `AIEngine/pipeline/charts.py` | `ChartExtractor` interface, `RuleBasedChartExtractor` (`extract_chart`), plot area, axis calibration, marks |
| `AIEngine/pipeline/figures_visualize.py` | Chart overlay (debug) |
| `AIEngine/pipeline/figures_eval.py` | `evaluate_figures`, `evaluate_chart` |
| `AIEngine/tools/inspect_figures.py` | Command-line tool |
| `AIEngine/tools/evaluate_figures.py` | Accuracy against ground truth |
| `tests/python/test_figures.py` | Tests |
| `tests/fixtures/make_figure_fixtures.py`, `tests/fixtures/figures/` | Fixtures and their ground truth |

Generic figure handling and chart reading are separate modules. `figures.py` knows nothing about
bars or axes; `charts.py` knows nothing about pages, captions or files.

## 2. Figure and chart classification strategy

`figure_type` is one of `chart`, `diagram`, `image`, `figure`, `unknown`. `figure_type_source` says
which rule decided. `source_label` always keeps the layout model's own label.

| Order | Rule | Result | `figure_type_source` |
|---|---|---|---|
| 1 | The layout label is `chart` | `chart` | `layout_label` |
| 2 | The crop is blank | `unknown` | `visual_rule` |
| 3 | The crop has two axis lines that meet at the lower left, and an axis with at least three numeric tick labels that fit one linear scale | `chart` | `visual_rule` |
| 4 | At least 85 % of the pixels have one of the crop's eight most common colours | `diagram` | `visual_rule` |
| 5 | At most 50 % do | `image` | `visual_rule` |
| 6 | Anything between | `figure` | `visual_rule` |

Rule 1 is the model's decision and is never overruled. Rule 3 exists for a chart the model labelled
`image`, and for a detector without a chart label. Rules 4–6 separate drawings (flat fills on a plain
background) from continuous-tone pictures by colour statistics alone; `figure` is the honest answer
when that is not clear. No vision-language model and no classifier network is used.

**Chart type** (`chart_data.chart_type`): `bar`, `line`, `scatter` or `unknown`. The schema also
allows `histogram`, `box`, `area` and `pie`; the rules of this phase never assign them, such charts
come back `unknown`. The type is the kind of mark (bar, line, point) that accounts for most mark
pixels, and it is assigned only when that kind explains at least 60 % of them (and, for a scatter
plot, there are at least three points). Otherwise the type stays `unknown`: it is not forced.
`chart_type_score` is that share. It is a measure of how uniform the marks are, **not a probability**.

## 3. Chart extractor

Candidates were compared before anything was built:

| Approach | Decision | Reason |
|---|---|---|
| **OCR text + OpenCV geometry, rules** | **Used** | No model, no download, no new framework; uses the OCR engine and the OpenCV build already installed; every number can be traced to pixels and tick labels; a few hundredths of a second per chart |
| ChartOCR-like key-point networks | Not used | Research code with its own trained networks and a PyTorch stack; nothing packaged that fits the PaddlePaddle environment |
| DePlot / MatCha (chart-to-table) | Not installed | Need PyTorch and `transformers` (no PyTorch in this environment) and weights of about 1 GB; they generate a table as text, so a wrong number looks exactly like a right one and carries no provenance |
| PlotQA-style parsing | Not used | A detection network plus rules; again a second framework and trained weights |
| PaddleX chart-to-table (PP-Chart2Table) | Not used | A vision-language model, large and meant for GPU. Not downloaded, not measured |

What is said about the systems that were not installed is from general knowledge of the projects,
not re-checked online for this phase. **Only the rule-based reader was run.**

`RuleBasedChartExtractor` (`name: "rule-based-opencv"`) works on the padded crop and the figure's text
items:

1. **Plot area.** An axis is a thin, dark, unsaturated line over at least 40 % of the crop. The x axis
   is the lowest such horizontal line that a vertical one meets at its left end. Without that pair
   there is no plot area and the reading is `"failed"`.
2. **Text roles**, by position: below the x axis the first row is the tick labels and the next the
   axis title; left of the y axis upright text is tick labels and rotated text the axis title; above
   the plot area is the title; a label with a patch or short line of a series colour directly left of
   it is a legend entry; a number just above a bar is its value label; other text in the plot area is
   an annotation.
3. **Axis scale.** The numeric tick labels of an axis give (pixel, value) pairs. The scale is the
   linear one most labels agree with (within 4 px), fitted by least squares; more than half of the
   labels must agree. One misread label is left out; a logarithmic axis gives no scale at all.
4. **Marks.** Coloured pixels in the plot area are grouped by hue, dark unsaturated pixels form one
   more group; text and the frame of the plot area are removed. Each connected component of a group
   is a bar (a filled rectangle standing on the axis or on the zero line), a line (thin and wide),
   a point (small and compact) or something else. Pieces of a line that another series crosses are
   put together again.
5. **Series.** One series per colour group. Bars get their category from the tick label under them
   and their value from the top edge. A line is reduced to the corner points of the polyline that
   stays within 2 px of the centre of its stroke. A point's position is its centre of mass.

Another reader can replace it by subclassing `ChartExtractor`:

```python
class ChartExtractor:
    name: str
    def load(self)                                    # loads once; the rule reader has nothing to load
    def extract_chart(self, image, texts, origin)     # -> chart_data; raises ChartExtractionError
```

## 4. Licences

No model was added, so there are no new weights. OpenCV (`opencv-contrib-python`): Apache-2.0. numpy:
BSD-3-Clause. Pillow: MIT-CMU (HPND). PaddleOCR, PaddleX, PaddlePaddle and the PP-OCRv5 models used
for figure text: Apache-2.0. `matplotlib` (fixture generation only): its own BSD-style licence. The
DejaVu fonts embedded in the fixture PDF: Bitstream Vera / DejaVu licence (free to redistribute).
PyMuPDF from Phase 2 remains AGPL-3.0.

## 5. Dependencies

| | |
|---|---|
| Python | 3.10.9 (Windows 10) |
| `opencv-contrib-python` | 4.10.0.84 — already installed (the build PaddleX pins for its OCR modules); now imported directly, so `requirements.txt` lists it |
| `numpy` | 2.2.6 (a dependency of PaddleX; unchanged) |
| `Pillow` | 12.3.0 (crops, overlays) |
| `paddleocr` / `paddlepaddle` / `paddlex` | 3.7.0 / 3.3.1 CPU / 3.7.2 (unchanged) |
| OCR models | `PP-OCRv5_mobile_det` (4.7 MB) and `en_PP-OCRv5_mobile_rec` (7.6 MB), the ones Phase 4 uses |
| `matplotlib` | 3.10.9 — only to regenerate the figure fixtures; not a pipeline or test requirement |
| Model for figures or charts | none; nothing is downloaded |
| Device | CPU |

**Nothing was installed for Phase 7** and no installed package changed version.

## 6. Caption association

`associate_figure_captions(regions, texts)` works per page, on geometry and on the first words of
the caption text.

**Candidates** are the `CAPTION` regions (label `figure_title`), without notes (`vision_footnote`)
and without captions whose text starts with a table label ("Table 2: …"). A `TEXT` region whose text
starts like a figure caption ("Figure 3: …") is also a candidate: a caption the layout model did not
label as one.

A caption **can** belong to a figure when

- it lies above or below the figure, not beside it (both layouts are supported);
- the two overlap horizontally by at least 25 % of the narrower one. A caption in another column
  does not overlap the figure at all, however near it is;
- they are at most 36 pt apart vertically.

Then each caption is offered only to the figure (or table) nearest to it, and each figure takes the
nearest caption offered to it. In both steps the winner must be clear: if the runner-up is less than
1.5 times farther, **nothing is assigned** and `caption` stays null. A caption that starts with a
figure label is never offered to a table.

Nearest Euclidean distance is not used anywhere. Page 4 of the fixture is built to show why: the
caption of the right column is 27 pt from the left figure, that figure's own caption 30 pt.

A figure entry keeps `caption_region_id`, `caption_position` (`"above"` / `"below"`),
`caption_pdf_bbox`, `caption_text_source` (`"native"` / `"ocr"`) and `caption_confidence` (the OCR
confidence of Phase 4, null for native text). The caption text itself comes from `ocr.json`.

## 7. Figure numbering

`parse_figure_number(text)` splits the label from the caption:

| Caption text | `figure_number` | `caption` |
|---|---|---|
| `Figure 1: Overview of the model.` | `Figure 1` | `Overview of the model.` |
| `Fig. 2. Results on the test set.` | `Fig. 2` | `Results on the test set.` |
| `Figure 3(a) Detail of the encoder.` | `Figure 3(a)` | `Detail of the encoder.` |
| `Fig. 4b: Ablation.` | `Fig. 4b` | `Ablation.` |
| `Figure S1: …`, `Figure A.2: …`, `FIG. 7 \| …` | `Figure S1`, `Figure A.2`, `FIG. 7` | … |
| `Figure 3 shows that the loss falls.` | null | the whole text |
| `Overview of the model.` | null | the whole text |

The number is kept as written; only its spacing is made regular (`Fig.5` → `Fig. 5`). "Fig." is not
expanded to "Figure". A label that runs on into a sentence without a separator is not taken as a
number. `caption_raw` keeps the unsplit text.

## 8. OCR reuse

There is one OCR system in the pipeline, and Phase 7 adds none. `FigureExtractor` takes the same
`OCREngine` object Phase 4 uses (`PaddleOCREngine`, PP-OCRv5); the command-line tool creates one
engine and passes it to both phases. It is loaded once, and only if a figure needs it.

Text in a figure comes from the PDF where possible, exactly as in Phase 4:

| | `text_source` | `confidence` |
|---|---|---|
| Native text lines of the PDF lie inside the figure (vector drawings) | `native` | null |
| Otherwise the crop goes to the OCR engine (bitmaps, scanned pages) | `ocr` | the engine's own value, per line |
| `--no-figure-ocr`, or a blank crop | — (no text) | — |

Phase 4 itself skips `FIGURE` regions, so this text is not in `ocr.json`; captions are, and are
taken from there.

OCR inside figures also "reads" plot marks and picture texture. A line is kept only if its confidence
is at least 0.5, it holds a letter or digit, and its box is at least half as high as the median line
of the same figure. For charts the crop is read a second time turned a quarter, to get axis titles
that run bottom to top; a rotated line is kept from 3 characters and confidence 0.8.

The result is `ocr_text`: `[{text, confidence, text_source, role, vertical, image_bbox, pdf_bbox}]`.
The field keeps the name the schema asked for although its items can be native text; `text_source`
says which. For a diagram these are its labels ("Input", "Backbone", …) and nothing else: they are
not interpreted, ordered into a flow, or turned into a description.

## 9. Chart structure schema

```json
"chart_data": {
  "status": "parsed",                    // parsed | partial | failed
  "extractor": "rule-based-opencv",
  "chart_type": "line",                  // bar | line | scatter | unknown
  "chart_type_score": 1.0,               // share of mark pixels the type explains; not a probability
  "plot_area_image_bbox": [x0, y0, x1, y1],
  "title": "Validation accuracy",
  "x_axis": { "title": "Epoch",    "ticks": ["1", "2", "3", "4", "5"],          "numeric": true, "calibrated": true },
  "y_axis": { "title": "Accuracy", "ticks": ["0.2", "0.4", "0.6", "0.8", "1.0"], "numeric": true, "calibrated": true },
  "legend": ["Method A", "Method B"],
  "series": [
    { "name": "Method A", "colour": "#cc3311",
      "points": [ { "x": 0.986, "y": 0.522, "value_source": "visual_estimate", "image_point": [x, y] } ] }
  ],
  "annotations": [],
  "notes": []
}
```

A bar series has `"bars": [{category, value, value_source, visual_estimate, image_bbox}]` instead of
`points`. `status` is `parsed` when at least one number was recovered, `partial` when the axes were
found but the type or the numbers were not, `failed` when there is no plot area or the reader raised.
`notes` gives the reason in words ("y axis not calibrated: …", "chart type not determined: …").

Nothing is invented: a title that is not there is null, a legend that is not there is `[]`, a series
without a legend entry has `name: null`, a value without a calibrated axis is null. Every role
assigned to a text item is also written to that item in `ocr_text` (`role`: `title`, `x_axis_title`,
`y_axis_title`, `x_tick`, `y_tick`, `legend`, `value_label`, `annotation`), where its confidence is.

All pixel geometry is in pixels of the Phase 2 page image, like `image_bbox`.

## 10. Numeric-data provenance

Every number in `chart_data` says where it comes from:

| `value_source` | Meaning |
|---|---|
| `label` | A number printed in the chart, read from the PDF's own text. Exact |
| `ocr` | A number printed in the chart, read by OCR. As good as the OCR; its confidence is on the `ocr_text` item with role `value_label` |
| `visual_estimate` | Measured on the pixels against the axis scale. **An estimate, never an exact value** |

- A bar with a printed value gets that value (`label` / `ocr`) **and** keeps the measurement in
  `visual_estimate`; the two are not merged. A bar without a printed value has `value` equal to the
  estimate and `value_source: "visual_estimate"`.
- Line and scatter points are always `visual_estimate`.
- Estimates are rounded to what one pixel resolves on that axis (an axis with 0.42 units per pixel
  gives one decimal), so they do not show digits the image cannot carry.
- A value that could not be determined is null and has `value_source: null`. The validator rejects a
  value without a source and a source without a value.
- `image_bbox` / `image_point` give the pixels each estimate was taken from.

## 11. Output and command line

```
python AIEngine/tools/inspect_figures.py <pdf> --output <dir>
python AIEngine/tools/inspect_figures.py <pdf> --output <dir> --save-figure-debug
python AIEngine/tools/inspect_figures.py <working directory> --no-figure-ocr
python AIEngine/tools/evaluate_figures.py <dir> --truth tests/fixtures/figures/synthetic_figures.json
```

Options: `--figure-padding N`, `--no-figure-ocr`, `--save-figure-debug`, and the shared `--force`,
`--device`, `--dpi`, `--layout-threshold`. Phases 2, 3 and 4 are reused when the working directory
holds them for this PDF and these settings, and run otherwise. Exit code 0, or 2 when the input
cannot be processed.

| Written | When | Content |
|---|---|---|
| `metadata/figures.json` | always | `extractor` (name, version, device), `settings`, `source`, `pages[].figures[]`, `summary` |
| `figure_images/page_004_fig_001.png` | always | The crop of the figure, exactly the pixels of its region. `image_path` points to it |
| `figure_crops/page_004_fig_001.png` | `--save-figure-debug` | The padded crop that was analysed |
| `figure_crops/page_004_fig_001_overlay.png` | `--save-figure-debug`, charts | The padded crop with the plot area (green), text boxes (orange; legend violet) and the detected bars and points (magenta) |

`pages.json`, `layout.json`, `ocr.json`, `tables.json` and `equations.json` are not touched. Both
image directories are in `.gitignore`.

The crop in `figure_images/` is written on every run, not only on request. For a generic figure the
image is the content: an entry without it would be a caption and a box. That is also what keeps the
original crop when something fails. Only the debug output waits for a flag.

Each figure entry:

```
figure_id, page_number, source_region_id, source_label, source_model,
image_bbox, pdf_bbox, crop_bbox,
figure_type, figure_type_source,
caption, caption_raw, figure_number, caption_region_id, caption_position, caption_pdf_bbox,
caption_text_source, caption_confidence,
image_path, ocr_text, text_source, chart_data, status [, error]
```

## 12. Metrics

`evaluate_figures(truth, document)` pairs each ground-truth figure with the extracted figure at the
same place (IoU ≥ 0.3) and reports:

| Metric | Definition |
|---|---|
| Caption match accuracy | Caption text equal (whitespace aside). "No caption" is correct when the truth has none |
| Figure number accuracy | Figure number equal; null is correct when the truth has none |
| Figure-type accuracy | `figure_type` equal |
| Label accuracy | Words printed in the generic figures that were read |
| Chart type accuracy | `chart_type` equal |
| Axis label accuracy | Chart title and axis titles equal; tick labels found on the right axis (two rates) |
| Legend text accuracy | Legend entries found |
| Exact value accuracy | Values printed in a chart that were read exactly, with `value_source` `label` or `ocr`. An estimate that happens to hit the number does not count |
| Visual estimates | Share of the measured values (bar heights, line and scatter points) for which an estimate was matched; MAE and RMSE of those estimates, per chart in the units of its y axis and overall as a share of each chart's y-axis span |

All rates are over all ground-truth figures: one that was not found counts as wrong.

**Fixtures** (`tests/fixtures/figures/`, generated by `make_figure_fixtures.py`): 5 pages, 9 figures —
an architecture diagram, a bitmap image, a bar chart with printed values, a line chart without
legend and with its caption above, a scatter plot, a two-series line chart with legend, a two-column
page (bar chart left, diagram with its caption above on the right) and a diagram between two
caption-like lines at the same distance. One copy has native text, the other is the same pages as
150 DPI colour bitmaps.

**Measured** with the real PP-DocLayoutV3 and PP-OCRv5 models, whole pipeline, CPU:

| | Digital PDF (native text) | Scanned copy (OCR) |
|---|---|---|
| Figure regions found | 9/9 | 9/9 |
| Layout labels | 5 `chart`, 4 `image` | 5 `chart`, 4 `image` |
| Figure-type accuracy | 9/9 | 9/9 |
| Caption match accuracy | 8/9 | 9/9 |
| Figure number accuracy | 8/9 | 9/9 |
| Label accuracy (words in the diagrams) | 9/9 | 9/9 |
| Chart type accuracy | 5/5 (2 bar, 2 line, 1 scatter) | 5/5 |
| Axis label accuracy: titles | 12/12 | 12/12 |
| Axis label accuracy: tick labels | 51/51 | 51/51 |
| Legend text accuracy | 2/2 | 2/2 |
| Exact value accuracy (4 printed bar values) | 4/4, `value_source: "label"` | 4/4, `value_source: "ocr"` |
| Visual estimates matched | 35/35 | 35/35 |
| MAE / RMSE of the estimates, share of the y-axis span | 0.002 / 0.002 | 0.002 / 0.003 |

Per chart, in the units of its y axis (MAE / RMSE of the visual estimates):

| Chart | y-axis span | Digital | Scanned |
|---|---|---|---|
| Bar chart with printed values (4 bars) | 100 | 0.225 / 0.229 | 0.375 / 0.378 |
| Line chart (6 points) | 2.5 | 0.0075 / 0.0091 | 0.0060 / 0.0065 |
| Scatter plot (12 points) | 50 | 0.10 / 0.10 | 0.00 / 0.00 |
| Two-series line chart (10 points) | 0.8 | 0.0015 / 0.0016 | 0.0027 / 0.0031 |
| Bar chart without printed values (3 bars) | 15 | 0.043 / 0.044 | 0.020 / 0.026 |

The one miss is on the digital copy, on the figure that was built to be ambiguous: PP-DocLayoutV3
reports the caption-like line below the diagram and not the one above it, so there is a single
candidate and the figure gets "Figure 10". With both lines present as regions (the scanned copy, and
the unit tests) the result is the intended null. The wrong-column caption of page 4 is rejected on
both copies.

These are nine figures drawn by the same script that wrote the ground truth, in matplotlib's default
style. The numbers show that the code path works on that style. **They are not a benchmark** and say
nothing about charts from real papers.

## 13. Performance

Measured on this machine (Windows 10, CPU; the GPU was not used):

| | Digital PDF | Scanned copy |
|---|---|---|
| Figures | 9 | 9 |
| Text source | native (the bitmap figure goes to OCR and yields nothing) | OCR for all nine |
| Phase 7 stage, total | 1.4 s | 11.5 s |
| Per figure | 0.02-0.10 s for a vector figure | 1.26 s on average, slowest 2.25 s |
| Chart reader alone | 0.02-0.10 s per chart | the same; the rest is OCR (two passes per chart) |
| OCR engine load | 0.6 s after the layout model was loaded in the same process; 4.9 s in a process of its own | 0 s here: Phase 4 had loaded it |
| Model load for figures / charts | none | none |

Run again on a working directory that already has Phases 2-4 (so the process loads only the OCR
engine): 5.7 s for the digital copy, 16.4 s for the scanned one, of which 4.9 s is the engine load.

The chart reader has no model and nothing to load. The only model start-up in this phase is the
Phase 4 OCR engine, and only when a figure has no native text.

## 14. Limitations

- **Synthetic charts only.** Everything measured is matplotlib in its default style. Real charts
  were not tested.
- **Chart types.** Vertical bar charts, line charts and scatter plots. Horizontal and stacked bars,
  histograms whose bars touch, box plots, area charts, pie charts, heat maps and anything in 3-D come
  back `unknown` or wrong. Error bars are not understood.
- **Axes.** Two axis lines meeting at the lower left are required: a chart without a drawn y axis
  has no plot area. Linear numeric scales only: logarithmic axes, dates, scientific notation
  ("1e3", "10³") and decimal commas give no calibration and therefore no numbers. Secondary y axes
  and several plots in one figure (subplots) are not handled; the reader sees one plot area.
- **Series are told apart by hue.** Two series in shades of one colour, or in grey tones, merge.
  Dashed and dotted lines break into pieces that are not recognized as a line. Dark grid lines can
  be taken for marks.
- **Line points are corners, not samples.** For a line drawn from data point to data point the
  corners are the data points, with the error of a pixel measurement; data points that lie on a
  straight segment are not reported. A smooth curve gives the corners of an approximation.
- **Scatter points that touch** are counted as one; a note says how many marks look merged.
- **Rotated tick labels and multi-line tick labels** are not grouped correctly.
- **Text roles are positional.** A legend is found only by the colour patch left of its label; a
  legend with patches on the right, or a second line of title, is misread.
- **`diagram` / `image` / `figure` is a colour statistic.** A photo with large flat areas can come
  out as `diagram`, a shaded drawing as `image`. It says nothing about what the figure shows.
- **Captions.** Only captions above or below a figure; a caption beside it is not associated. A short
  caption set flush left under a centred figure can miss the 25 % overlap. A caption the layout model
  did not report as a region is not seen: on the digital fixture the model reports only one of the two
  lines around the ambiguous diagram, so that figure gets a caption where the ground truth says
  "undecidable". Sub-figures (a), (b) under one caption are separate figures; only one gets the caption.
- **Figure numbers** in languages other than English ("Abbildung 3", "图 3") are not recognized.
- **OCR in figures** is filtered by confidence and size; a real but tiny or unsure label is dropped,
  and a mark read with high confidence as a letter is kept.
- **No GPU test.** `--device gpu:0` is passed to the OCR engine but untested here.

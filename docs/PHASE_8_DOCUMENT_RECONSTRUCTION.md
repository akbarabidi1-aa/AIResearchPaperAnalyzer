# Phase 8 — Document reconstruction and reading order

Phase 8 combines the artifacts of Phases 2–7 into one structured document:

```
pages.json  layout.json  ocr.json  tables.json  equations.json  figures.json
    ->  one canonical block per layout region (text: native, else OCR)
    ->  tables, equations, figures attached to their blocks; captions and equation numbers consumed
    ->  reading order per page (columns, full-width blocks), then one order for the whole document
    ->  headings, section tree, a section for every block
    ->  paragraphs joined across column and page breaks; mentions ("Figure 3") linked
    ->  metadata/document.json, validated against AIEngine/schema/document.schema.json
```

Structure only. There is no summary, no research problem, no contribution or limitation extraction,
no LLM and no external API in this phase; nothing in `document.json` says what the paper means. It
runs alongside V1 and Phases 2–7; `ai_engine.py`, the ASP.NET application and the database are
unchanged and do not call it.

### Code layout

| File | Role |
|---|---|
| `AIEngine/pipeline/reading_order.py` | `detect_columns`, `order_blocks`, `pairwise_order_accuracy`, `exact_order` |
| `AIEngine/pipeline/sections.py` | `parse_heading`, `looks_like_heading`, `special_section_type`, `build_sections` |
| `AIEngine/pipeline/document_builder.py` | `build_document`, `load_artifacts`, `validate_document`, `check_document_integrity`, `check_information_loss`, `is_continuation`, `find_mentions` |
| `AIEngine/pipeline/document_eval.py` | `evaluate_document` |
| `AIEngine/pipeline/order_visualize.py` | Reading-order images |
| `AIEngine/schema/document.schema.json` | JSON Schema of `document.json` |
| `AIEngine/tools/build_document.py` | Command-line tool |
| `tests/python/test_document.py` | Tests |
| `tests/fixtures/make_document_fixtures.py`, `tests/fixtures/document/` | Fixtures and their ground truth |

`reading_order.py` sees only boxes. `sections.py` sees only ordered blocks with their text.
`document_builder.py` is the only module that reads the artifacts and writes the file.

New dependency: `jsonschema` 4.26.0 (MIT), with `attrs`, `referencing`, `jsonschema-specifications`
and `rpds-py`. No installed package changed version. No model is used by this phase.

## 1. Canonical block schema

Every region of `layout.json` becomes exactly one block. Nothing is dropped; what is not read as
body text is marked, not removed.

```json
{
  "block_id": "page_002_block_006",
  "page_number": 2,
  "type": "EQUATION",
  "bbox": [149.1, 233.6, 188.7, 254.0],
  "text": null,
  "source": null,
  "confidence": null,
  "section_id": "sec_3_1",
  "order": 17,
  "flow": "main",
  "column": 0,
  "full_width": false,
  "source_region_id": "p002_r006",
  "source_label": "display_formula",
  "references": [],
  "equation_id": "page_002_equation_001", "latex": "z=\\frac{x+y}{2}", "equation_number": "(1)", "status": "success"
}
```

| Field | Meaning |
|---|---|
| `type` | `TITLE`, `HEADING`, `PARAGRAPH`, `LIST`, `TABLE`, `FIGURE`, `CHART`, `EQUATION`, `CAPTION`, `HEADER`, `FOOTER`, `PAGE_NUMBER`, `OTHER` |
| `bbox` | PDF points, origin top-left, `[x0, y0, x1, y1]` (the region's `pdf_bbox`, unchanged) |
| `text`, `source`, `confidence` | The block's one text value and where it comes from: `native` (confidence null) or `ocr` (Phase 4's confidence). Null for a block without text |
| `flow` | `main`: read in order. `attached`: a caption or equation number that belongs to another block (`attached_to`). `merged`: the later part of a joined paragraph (`merged_into`). `furniture`: header, footer, page number |
| `order` | Position in the reading order of the whole document, 0…n-1, for `main` blocks; null otherwise |
| `column`, `full_width` | Column index on its page (0 = leftmost), or `full_width: true` for a block that spans the columns |
| `section_id` | The section it belongs to (null only for furniture) |
| `references` | Numbered objects its text mentions (section 1.1) |
| `source_region_id`, `source_label` | The layout region and the layout model's own label |

Type mapping from the layout classes: `doc_title` → `TITLE`; `paragraph_title` → `HEADING`; `TEXT` →
`PARAGRAPH`; `LIST`, `TABLE`, `EQUATION`, `CAPTION`, `HEADER`, `FOOTER`, `PAGE_NUMBER`, `OTHER`
keep their class; `FIGURE` → `CHART` when Phase 7 typed it as a chart, else `FIGURE`.

Tables, figures and equations are never turned into paragraphs. Their block carries the id of the
object and a few of its fields; the object itself is in the top-level `tables`, `figures`,
`equations` lists. A `TABLE` block also keeps the raw region text Phase 4 stored, as `text`; the
rows and cells are in the table object.

**Text.** Phase 4 already decided, per region, between native text and OCR; `ocr.json` holds one
text per region with its `text_source`. Phase 8 takes that one value. `pages.json` is not read for
text again, so no paragraph can appear twice. Lines are joined with a space (kept as lines for a
`LIST`); words are not re-hyphenated or corrected.

### 1.1 Mentions

`find_mentions` finds "Figure 3", "Fig. 3", "Table 2", "Eq. (4)", "Equation 4" and "Section 3.1" in
the text of paragraphs and lists. Each becomes `{"kind", "label", "target_block_id"}`. The target is
set only when exactly one object carries that number (a figure by its `figure_number`, a table by a
caption starting with "Table N", an equation by its `equation_number`, a section by its number);
otherwise it is null. "Table 7" in a paper with one table stays unresolved.

## 2. Reading-order algorithm

Per page, on the `main` blocks, in `reading_order.order_blocks`:

1. **Columns** are detected (section 3).
2. **Full-width blocks** are the blocks that reach across a gutter (section 4). Sorted from top to
   bottom they cut the page into horizontal bands.
3. Every other block belongs to the column its centre lies in, and to the band its vertical centre
   lies in.
4. The order is: band 1, column by column from left to right, top to bottom inside each column;
   then the full-width block below it; then band 2; and so on.

```
title (full width)                      1
left column   | right column            2 3 | 4 5
--------- wide figure ---------         6
left column   | right column            7 8 | 9
```

Inside a column, blocks that stand side by side (an equation and its number) are read left to right.

The page is never sorted by "y, then x". On a two-column page that reads across the gutter;
`test_sorting_by_y_then_x_would_be_wrong` keeps that failure on record.

**Across pages.** Pages are read in sequence and the per-page orders are concatenated into one
global `order` (0…n-1). `pages[].block_ids` and `sections[].blocks` list the same blocks in that order.

## 3. Column detection

`detect_columns(boxes, page_width)` returns `{"layout", "columns": [[x0, x1], …], "gutters": [x, …]}`.

A **gutter** is a vertical position, in the middle 60 % of the width being split, where

- at least one block lies wholly on each side, and
- two of them, one from each side, stand side by side (they overlap vertically).

Blocks that reach across a position are full-width blocks there. Among the qualifying positions the
gutter is where the fewest blocks reach across — so it falls between the columns, where only the
title and the wide figure cross, and not beside a short heading, where every paragraph of the column
would cross. The free band must be at least 4 pt wide; the gutter is its middle. Each side is then
searched again (with two blocks required per side) for further gutters.

| Result | When |
|---|---|
| `single_column` | No gutter, and no block confined to the right half |
| `two_column` | One gutter |
| `multi_column` | Two or more gutters (three or more columns) |
| `unknown` | No gutter, but blocks confined to each half that are not side by side. Columns cannot be told; the page is read top to bottom, like a single column |

Blocks narrower than 5 % of the page width (equation numbers) and `OTHER` blocks do not take part
in detecting columns. Nothing assumes that a paper has two columns: every page is decided on its own,
so a paper with a single-column first page and two-column pages after it is handled page by page.

## 4. Full-width handling

A block is full-width when it extends more than 3 pt beyond a gutter on both sides. That covers the
title, an abstract set across the page, a wide figure or table, and a short centred heading that
merely straddles the gutter. Such a block is not pushed into the left or the right column: it is a
separator. Column text above it is read in all columns before it, column text below it after it.

## 5. Section detection

A block is a heading on evidence from three sides:

| Evidence | Use |
|---|---|
| **Label** | The layout model's `paragraph_title` makes a `HEADING` (unless the region is longer than 120 characters: then it is running text and becomes a `PARAGRAPH`) |
| **Numbering** | `3 Methodology`, `3.1 Architecture`, `2. Related Work`, `IV. EXPERIMENTS`, `A. Datasets`, `Appendix B: Proofs`. A number counts only when a capitalized title follows it ("3 of the 5 runs" is not a heading) |
| **Form** | One line, at most 12 words, no full stop at its end, not set smaller than the body text (line height from the native lines, or from the OCR line boxes) |

A block the model did **not** label as a title is promoted to `HEADING` only when it has the form of
a heading **and** either a section number or the exact name of a special section. A name alone
("Related Work" as a plain line) is not enough, and a list of section names is used only for the
special sections of section 5.1. Font names and weights are not in `pages.json`, so boldness is not
available as evidence; line height is.

### 5.1 Special sections

`section_type` is one of `front_matter`, `abstract`, `body`, `references`, `acknowledgments`, `appendix`.

- **Front matter**: everything before the first heading (title, authors). The document title is the
  text of the first `TITLE` block, or null.
- **Abstract, References, Acknowledgments, Appendix**: a heading that is exactly that name
  ("Abstract", "References" / "Bibliography", "Acknowledg(e)ments", "Appendix" / "Appendix A").
  Sections after an appendix heading are `appendix` too.
- **Without a heading**: a block the layout model labelled `abstract`, or as a reference entry
  (`reference`, `reference_content`), opens an *implicit* section of that type. Its `title` is null
  and `implicit` is true: no heading was read, so none is invented.

References are kept in full and tagged, so that a later phase can leave them out.

## 6. Hierarchy logic

| Heading | Level | Parent |
|---|---|---|
| `3 Methodology` | 1 | none |
| `3.1 Feature Extraction` | 2 | the section numbered `3` |
| `3.1.2 …` | 3 | the section numbered `3.1` |
| `IV. EXPERIMENTS` | 1 | none |
| `A. Datasets` in a paper with Roman section numbers | 2 | the current Roman section |
| Unnumbered, set at least 8 % smaller than the top-level headings | 2 | the current level-1 section |
| Any other unnumbered heading | 1 | none |

A numbered subsection hangs **only** under the section whose number it extends. If "4.1" appears and
no section "4" was found, it keeps level 2 and gets `parent_id: null` — no parent is made up.
Without numbers, a lower level is assigned only on visible evidence (smaller type).

Section ids: `sec_3`, `sec_3_1`, `sec_ii`, `sec_ii_a`, `sec_appendix_b`; unnumbered sections get
`sec_` + the title in lower case (`sec_related_work`, `sec_references`); the front matter is
`sec_front_matter`. A repeated id gets `_2`, `_3`.

Each section: `section_id`, `title`, `number`, `level`, `parent_id`, `section_type`,
`heading_block_id`, `implicit`, `page_start`, `page_end`, `blocks` (the ids of its own main-flow
blocks, in order; the blocks of a subsection are listed in the subsection, not again in its parent).

## 7. Paragraph merging

Two paragraphs are joined only when **all** of this holds (`is_continuation`):

- both are `PARAGRAPH` blocks of the same section, neither a reference entry;
- the second is the next text block read after the first — only figures, charts and tables may
  stand between them, never a heading;
- they are in different columns or on different pages (a break inside one column is a new paragraph);
- the first does not end a sentence: no `. ! ? : ;` at its end, closing quotes and brackets aside;
- the second starts with a lowercase letter.

The first block then holds the whole text and `parts`: the original blocks with their id, page, box,
text and source. The later block stays in `blocks` with its own text, `flow: "merged"` and
`merged_into`; it has no `order` and is in no section list. A word split by a hyphen at the break is
joined without a space and keeps its hyphen; it is not re-hyphenated.

## 8. Table integration

A `TABLE` block points to an entry of `tables`:

`table_id` (`page_003_table_001`), `source_id` (the id in `tables.json`), `block_id`, `page_number`,
`bbox`, `status` (`ok` / `failed` / `not_extracted`), `caption`, `caption_block_id`, `n_rows`,
`n_columns`, `header_rows`, `headers`, `rows`, `extraction_source` (`native` / `vision` / `hybrid`), `error`.

The table stands in the reading order at the place of its region, as a table. Cell boxes and
confidences stay in `tables.json`, which `provenance.artifacts` names; they are not copied.

## 9. Equation integration

An `EQUATION` block points to an entry of `equations`: `equation_id`, `source_id`, `block_id`,
`status`, `latex`, `normalized_latex`, `equation_number`, `equation_number_block_id`, `image_path`, `error`.

An equation that was not recognized is kept: `status` is `invalid_latex` or `recognition_failed`,
`latex` is null and `image_path` is the crop Phase 6 saved. It keeps its place in the order.

The "(4)" beside an equation is usually a layout region of its own (`formula_number`). When its box
is the one Phase 6 read the number from, that block is attached to the equation and leaves the
reading order, like a caption.

## 10. Figure integration

A `FIGURE` or `CHART` block points to an entry of `figures`: `figure_id`, `source_id`, `block_id`,
`status`, `figure_type`, `figure_number`, `caption`, `caption_block_id`, `image_path`, `labels` (the
words printed in the figure), `chart_data` (as Phase 7 wrote it, with its `value_source` marks), `error`.

No description is generated. A figure is its image, its caption and the words printed in it.

If Phase 5, 6 or 7 was not run, the regions of that kind still become blocks and objects, with
`status: "not_extracted"`.

## 11. Caption deduplication

Phases 5 and 7 pair captions with tables and figures (`caption_region_id`). Phase 8 does not pair
anything again; it consumes what was paired:

- the caption block gets `flow: "attached"` and `attached_to` (the table or figure block), no
  `order`, and is in no section list;
- the object has the caption text and `caption_block_id`;
- the caption block itself stays in `blocks`, with its text, box and source.

So a consumed caption is in the document exactly once as content of its object, and once as a block
for provenance — never as a paragraph. A caption that no phase could pair (ambiguous, or its phase
did not run) stays a `CAPTION` block in the reading order; it is not guessed onto an object.
`check_document_integrity` rejects a caption attached to two objects, and one that is both attached
and listed in a section.

## 12. Header / footer handling

`HEADER`, `FOOTER` and `PAGE_NUMBER` blocks get `flow: "furniture"`: no `order`, no section, and
they take no part in column detection. They are kept three times over, cheaply: as blocks, in
`pages[].header_block_ids` / `footer_block_ids` / `page_number_block_ids`, and with their text in
`metadata.headers`, `metadata.footers`, `metadata.page_numbers`.

## 13. Document JSON schema

```json
{
  "document":  { "title", "page_count", "source_pdf", "source_sha256", "schema_version": "1.0" },
  "sections":  [ { "section_id", "title", "number", "level", "parent_id", "section_type", "blocks": ["page_001_block_004", …] } ],
  "blocks":    [ … every block, in layout order … ],
  "pages":     [ { "page_number", "width", "height", "layout", "columns", "gutters", "block_ids", "header_block_ids", … } ],
  "tables":    [ … ], "figures": [ … ], "equations": [ … ],
  "metadata":  { "headers": [ … ], "footers": [ … ], "page_numbers": [ … ] },
  "provenance": { "artifacts": { "pages", "layout", "ocr", "tables", "equations", "figures" }, "coordinate_system", "counts" },
  "summary":   { … counts … }
}
```

Large objects exist once. Sections and pages list block ids; blocks name their table, figure or
equation by id.

`AIEngine/schema/document.schema.json` (JSON Schema draft 2020-12) states the required fields of
every object explicitly, closes the top level, blocks, sections and pages against unknown fields,
and fixes the enumerations (block types, flows, section types, layouts, statuses). It also says what
depends on what: a `TABLE` block needs a `table_id`, a paragraph must not have one, a `main` block
needs an integer `order` and a section, any other block must have `order: null`.

`validate_document` returns one message per violation with its place, for example
`blocks/3/type: 'SIDEBAR' is not one of ['TITLE', …]`.

Three checks run on every build, and a document that fails any of them is not written
(`DocumentValidationError`):

| Check | What it covers |
|---|---|
| `validate_document` | The schema |
| `check_document_integrity` | Unique block, section and object ids; every block on an existing page and inside it; orders 0…n-1 exactly once, rising with the page number; every `section_id`, `parent_id`, object id, `attached_to`, `merged_into` and reference target exists; every main-flow block in exactly one section list and on its page list; no caption attached twice |
| `check_information_loss` | Every layout region is exactly one block; every text of `ocr.json` (native and OCR counted separately) is the text of its block, word for word; every table, equation and figure of Phases 5–7 is in the document, including failed ones with their image; every caption that was paired is attached. The counts on both sides are stored in `provenance.counts` |

## 14. Stable IDs

| Object | Id | Derived from |
|---|---|---|
| Block | `page_003_block_007` | Page number and position of the region in `layout.json` |
| Table | `page_003_table_001` | Page number and position among the tables of the page |
| Figure | `page_002_figure_002` | The same, for figures (charts included) |
| Equation | `page_002_equation_001` | The same, for equations |
| Section | `sec_3_1`, `sec_references` | Section number, or title |

No random ids and no timestamps: building the same working directory twice gives a byte-identical
`document.json` (tested). The ids are stable for a given `layout.json`; a layout run that finds
other regions gives other block ids. `source_id` and `source_region_id` keep the ids of the artifacts.

## 15. Command line

```
python AIEngine/tools/build_document.py <pdf> --output <dir>
python AIEngine/tools/build_document.py <pdf> --output <dir> --visualize-order
python AIEngine/tools/build_document.py <dir> --truth tests/fixtures/document/synthetic_papers.json
```

The tool reuses or creates Phases 2–7 in order (pages, layout, OCR, tables, equations, figures),
then orders, sections, builds, validates and prints a summary. One OCR engine serves Phases 4, 5
and 7. Options: `--visualize-order`, `--truth`, `--truth-name`, and the shared `--force`, `--device`,
`--dpi`, `--layout-threshold`. Exit code 0, or 2 when the input cannot be processed or the document
fails its checks.

| Written | When |
|---|---|
| `metadata/document.json` | always |
| `order_debug/page_001_order.png` | `--visualize-order`: each page with its blocks outlined and numbered 1, 2, 3 … in reading order (column blocks blue, full-width blocks orange, gutters green, a line from each block to the next) |

Both are in the working directory of the PDF, not in the repository; `order_debug/` is in `.gitignore`.

## 16. Evaluation metrics

`evaluate_document(truth, document)` pairs each ground-truth block with the document block that
covers it and reports:

| Metric | Definition |
|---|---|
| `pairwise_order_accuracy` | Of all pairs of main-flow blocks of the document, the share read in the expected relative order. A block that is missing from the order makes its pairs wrong |
| Exact order | Per page (`pages_in_exact_order`) and for the whole document (`exact_order`) |
| Heading detection accuracy | Expected headings that are `HEADING` blocks ÷ (expected headings + blocks that are `HEADING` without being one) |
| Section assignment accuracy | Blocks whose section is the expected one |
| Section hierarchy accuracy | Expected sections found with the right level and the right parent |
| Also | Block type accuracy; captions and equation numbers attached to the right block; headers and page numbers kept out of the order; continuations joined; pages whose layout was detected as expected |

**Fixtures** (`tests/fixtures/document/`): a three-page two-column paper and a two-page
single-column paper, 82 blocks together, covering a full-width title, a heading across the columns,
a full-width figure between column text, a full-width table, an equation and a chart inside a
column, a paragraph continued over a page break, nested subsections, references, and running header
and page number on every page.

**Measured** with the real models of Phases 3–7 (PP-DocLayoutV3, PP-OCRv5, SLANet_plus,
PP-FormulaNet_plus-S), whole pipeline, CPU:

| | Two-column paper | Single-column paper |
|---|---|---|
| Pages / layout detected | 3 / 3 two-column | 2 / 2 single-column |
| Blocks found | 49/49 | 33/33 |
| Pairwise order accuracy | 1.000 | 1.000 |
| Pages in exact order | 3/3, whole document exact | 2/2, whole document exact |
| Heading detection accuracy | 1.000 (9/9, 0 false) | 1.000 (5/5, 0 false) |
| Section assignment accuracy | 1.000 | 1.000 |
| Section hierarchy accuracy | 1.000 (9 sections) | 1.000 (5 sections) |
| Sections / subsections | 7 / 2 | 4 / 1 |
| Paragraphs, tables, figures, charts, equations | 24, 1, 1, 1, 1 | 16, 1, 0, 1, 1 |
| Captions attached / equation numbers attached | 3/3 / 1/1 | 2/2 / 1/1 |
| Headers and page numbers kept out | 6/6 | 4/4 |
| Continuations joined | 1/1 | 1/1 |
| Mentions found / resolved | 4 / 4 | 3 / 2 (the third names a table that does not exist) |
| Information loss | 49/49 regions, 46/46 texts, 1/1 table, 1/1 equation, 2/2 figures | 33/33, 31/31, 1/1, 1/1, 1/1 |
| Reconstruction time | 0.13 s (0.4 s with the reading-order images) | 0.11 s (0.3 s) |

Reconstruction itself takes a fraction of a second; a first run of the tool spends 25–40 s in
Phases 2–7.

Both papers were written by the script that wrote the ground truth, with clean layouts. The numbers
show that the code path works on them. **They are not a benchmark** and say nothing about real papers.

## 17. Limitations

- **Synthetic documents only.** No real paper was reconstructed in this phase, and the fixtures
  have native text; the OCR text path of Phase 8 is covered by a unit test with a fake engine, not
  by a scanned copy read with the real models.
- **The order is only as good as the layout regions.** A region the layout model draws around two
  paragraphs is one block; a paragraph it misses is not in the document at all. In an earlier draft
  of the fixture the model labelled a paragraph directly under a chart caption as a figure note
  (`vision_footnote`); such a block stays a `CAPTION` in the reading order and is not joined to
  anything.
- **Columns.** A column structure needs two blocks side by side. A page with a single block per
  column that do not overlap vertically is `unknown` and is read top to bottom. Three or more columns
  are handled by the same rule but were tested only on boxes, not on a document.
- **Partial spans.** A block that spans two of three columns separates all three. Text that wraps
  around a figure, margin notes and rotated pages are not modelled.
- **Footnotes** are `OTHER` blocks in the reading order at the foot of their column. They are not
  linked to their marks.
- **Headings.** Boldness and font are not available as evidence. An unnumbered heading the layout
  model did not label is not found, unless it names a special section. A heading set on the same line
  as the start of its paragraph ("Abstract — We propose …") is not split from it.
- **Hierarchy without numbers** rests on line height alone; two heading levels of the same size are
  both level 1.
- **Lettered sections** ("A. Datasets") are subsections only in a paper with Roman section numbers,
  and appendix sections only after an appendix heading; elsewhere such a line is an unnumbered heading.
- **Paragraph joining is conservative.** A continuation that starts with a capital letter, a number
  or a formula is not joined; a paragraph that ends without punctuation before an unrelated one that
  starts in lower case would be joined wrongly. Hyphenated breaks keep their hyphen.
- **Mentions** are found for English labels only, and resolved by number alone: "Figure 3" in a
  paper where two figures carry the number 3 stays unresolved, and "Table A1" is not found.
- **Languages.** Section names and labels are English.
- **Ids depend on the layout run** (section 14).

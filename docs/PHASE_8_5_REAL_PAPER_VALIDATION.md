# Phase 8.5 — Real paper validation

An audit of the Phase 2–8 pipeline (checkpoint `480f751`, HEAD `0845fce`) on a real research paper
and on an image-only copy of it. Nothing in the pipeline was changed: no fix listed here has been
implemented. Phase 9 was not started.

**STATUS: PHASE 8.5 INCOMPLETE**

**OVERALL PHASE 9 READINESS: NOT READY**

Reason:

- the rasterized OCR path loses important one-line regions (headings, captions, references,
  equation numbers);
- no real two-column paper has yet been evaluated;
- validation coverage is incomplete: one real paper, single-column, was available.

This document is a checkpoint of the evidence collected so far, not a completed audit.

## 1. Validation set

| Paper ID | File | Kind | Category |
|---|---|---|---|
| P1 | `tests/real_papers/digital_single_column/cancers-15-04172.pdf` | Real digital paper | Single column; also contains tables, figures, charts and equations |
| P1-R | `tests/real_papers/scanned/cancers-15-04172_rasterized_200dpi.pdf` | **Synthetic scanned variant / rasterized digital paper** | OCR path |

P1 is Abdusalomov, Mukhiddinov, Whangbo, *Brain Tumor Detection Based on Deep Learning Approaches and
Magnetic Resonance Imaging*, Cancers 2023, 15, 4172 (MDPI, CC BY). SHA-256 `838a2586…f099ed66`.

P1-R was made from P1 for this audit: every page rendered at 200 dpi, grayscale, JPEG quality 85,
placed on a page of the original size, no text layer. It is **not a physical scan**. It has no skew,
no blur, no bleed-through, no uneven illumination and no scanner artifacts; the only degradation is
the 200 dpi sampling and JPEG compression. It exercises the OCR path on real academic content and
says nothing about noisy scans.

Categories with no paper yet: `digital_two_column`, `table_heavy`, `equation_heavy`, `figure_heavy`,
and a real physical scan. P1 has 6 tables, 11 figures and 11 equations, but is not heavy in any of them.

Both files were run with `AIEngine/tools/build_document.py <pdf> --output <dir> --visualize-order`,
default settings, CPU. PDFs and all outputs are under `tests/real_papers/`, which `.gitignore` excludes.

| | P1 | P1-R |
|---|---|---|
| Pages | 29 | 29 |
| Phases 2–7 time | 201 s | 589 s |
| Reconstruction time | 3.9 s | 4.1 s |
| Schema, integrity, information-loss checks | pass, 0 problems | pass, 0 problems |

## 2. Summary

| Paper | Layout | Pages | Reading Order | Sections | Tables | Figures | Equations | References | Readiness |
|---|---|---|---|---|---|---|---|---|---|
| P1 (digital) | single column, margin sidebar on p. 1 | 29 | MINOR ERROR | PARTIAL | 4/6 correct, 2 with wrong cell text | 11/11 found, 7/11 with caption | 10/11 recognized, all faithful | CORRECT | READY WITH MINOR ISSUES (3 major, none blocking) |
| P1-R (rasterized) | same | 29 | MINOR ERROR | PARTIAL (3 of 14 subsections lost) | 6/6 extracted, 1/6 with caption, 1 wrong number | 11/11 found, 2/11 with number | 11/11 recognized, 5/11 numbered | CORRECT (4 of 96 entries empty) | NOT READY |

## 3. P1 — digital paper

| Field | Value |
|---|---|
| Paper ID | P1 |
| Filename | `cancers-15-04172.pdf` |
| Category | `digital_single_column` |
| Digital / rasterized | Digital |
| Layout type | Single column with a wide left margin; page 1 has a margin sidebar |
| Page count | 29 |
| Sections found | Front matter, abstract (implicit), 1 Introduction, 2 Related Works, 3 Materials and Methods (3.1–3.10), 4 Results (4.1–4.4), 5 Discussion and Future Work, 6 Conclusions, References |
| Table count | 6 of 6 |
| Figure count | 16 regions for 11 real figures (Figure 11 is 4 regions; 2 are page-1 badges) |
| Equation count | 11 of 11 |
| OCR used | 2 of 320 text regions (page-1 header images); the rest native |
| Reading order result | MINOR ERROR |
| Section result | PARTIAL |
| Table result | STRUCTURE and TEXT failures in Tables 5 and 6; Tables 1–4 correct |
| Figure result | Detection correct; CAPTION failures on Figures 3, 9, 10, 11 |
| Equation result | 10 faithful, 1 kept as a failed recognition |
| Caption result | 13 of 17 attached (6/6 table, 7/11 figure); 4 left standalone |
| References result | CORRECT |
| Cross-page result | 12 of 13 continuations joined, none joined wrongly |
| Readiness | READY WITH MINOR ISSUES |

### Reading order

Pages inspected against the rendered page: 1 (difficult layout), 2–6 (text), 11, 13, 18, 20, 21
(multimodal), 25 (references start). All 29 pages were read through in the block listing.

- Title, authors, abstract, body, figures, tables and equations are in the right order on every page.
- Tables and equations sit where they are printed. Captions that were attached leave the order;
  the four that were not stay directly after their figure.
- Page 1 is detected as two-column (gutter at x = 158). The margin sidebar (citation, editor, dates,
  licence) is read after the author line and before the affiliation and abstract. Everything stays
  in the front matter and the abstract is intact, so this is a minor error.
- Page 20 is `unknown` layout because of four images side by side; the order is still correct.
- 27 pages are `single_column`.

### Sections

| Section | Result |
|---|---|
| Title | CORRECT |
| Abstract | CORRECT, as an implicit section (the heading is run-in: "Abstract: …"). "Simple Summary" and "Keywords" are in it too |
| Introduction, Related Works | CORRECT |
| Method (3, 3.1–3.10) | CORRECT, all ten subsections under `sec_3` |
| Results (4, 4.1–4.4) | CORRECT |
| Discussion | CORRECT |
| Conclusion | PARTIAL: 7 of its 11 blocks are back matter |
| References | CORRECT |
| Appendix | Not present in the paper |
| Back matter (Author Contributions, Funding, Review Board, Consent, Data Availability, Acknowledgments, Conflicts of Interest) | MISSING as sections; assigned to Conclusions |

Numbered hierarchy and block assignment are correct for all 21 headings.

### Tables

| Table | Page | Detected | Rows × cols | Numbers | Headers | Caption | In document | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | 9 | yes | 6 × 5 | correct | correct | attached | once, in 3.2 | CORRECT |
| 2 | 10 | yes | 8 × 6 | correct | group header "Training Images" joined to one column only | attached | once, in 3.3 | MINOR (STRUCTURE) |
| 3 | 19 | yes | 8 × 7 | correct | correct | attached | once, in 4.2 | CORRECT |
| 4 | 19 | yes | 9 × 7 | correct | correct | attached | once, in 4.2 | CORRECT |
| 5 | 21 | yes | 6 × 10 | correct | "PR" placed under "Attention Mechanism" | attached | once, in 4.4 | MAJOR (TEXT, STRUCTURE) |
| 6 | 21 | yes | 11 × 10 | correct | no header rows found | attached | once, in 4.4 | MAJOR (TEXT, STRUCTURE) |

In Tables 5 and 6 the ✓/× marks are merged into the wrong rows (cells such as `√ ×`, others empty),
so which configuration produced which numbers cannot be read from the table. Table 6 also loses its
row label "YOLOv7 + CBAM" and gains an empty last row. No DETECTION, CAPTION or DOCUMENT INTEGRATION
failure.

### Equations

| Eq. | Page | Result |
|---|---|---|
| (1) | 13 | Faithful. Function names are letter-spaced (`M L P`, `A v g P o o l`) |
| (2) | 13 | `invalid_latex`: the prediction ends in a stray `\`. Kept with its crop and number; `latex` is null |
| (3)–(5) | 14–15 | Faithful, letter-spaced (`C o n v`, `R e s i z e`) |
| (6) | 16 | Faithful, letter-spaced |
| (7)–(11) | 17 | Faithful |

All 11 regions were detected, all 11 numbers were read and attached, and every equation is in the
right place and section. No variable substitution was found; the raw prediction for (2) matches the
paper apart from its last character. Confidence is null for
every equation. The paper has no aligned equations, so `&=` handling was not exercised. Inline
formulas are plain paragraph text.

### Figures and charts

| Figure | Page | Result |
|---|---|---|
| 1, 2, 4, 5, 6, 7, 8 | 3–16 | Detected, crop correct, caption and number attached |
| 3 | 11 | Detected, crop correct; caption not attached, no number |
| 9 (bar chart) | 18 | Typed `chart`; caption not attached; chart data extraction failed ("no pair of axis lines found") |
| 10 (confusion matrix) | 18 | Typed `chart`; caption not attached; chart data extraction failed |
| 11 | 20 | Split into 4 column strips, each across rows (a) and (b); caption not attached |
| — | 1 | "Check for updates" and CC-BY badges extracted as two figures in the front matter |

Chart data was extracted for 0 of 2 charts; the printed values (99.5, 99.3, …) are kept as labels.
No adjacent text was mislabelled as a figure note and no caption went to the wrong figure.

### Captions

- No caption appears twice: an attached caption has no order and is in no section list.
- The four unattached captions are genuinely ambiguous to the current rule and were left
  unresolved rather than guessed. On page 18 the Figure 9 caption is nearer to the chart below it
  than to its own chart, so a guess would have been wrong.

### References

- "References" is detected as a heading and opens `sec_references` (type `references`).
- All 96 entries are separate blocks labelled `reference_content`, in order 1–96, pages 25–29.
- No body text is in the section and no reference is outside it. One non-reference block, the
  publisher's disclaimer at the end of page 29, is in it.
- References can be excluded reliably. Not a Phase 9 blocker.

### Headers, footers, footnotes

- 36 headers, 2 footers and 14 page numbers are kept out of the reading order; none is in the body.
- "N of 29" is labelled as a page number on 14 pages, as a header on 5, and has no layout region on
  9 pages. Either way it is not in the body.
- The paper has no footnotes, so footnote behaviour was not exercised.

### Multimodal integrity

| Modality | Expected | Integrated | Missing |
|---|---|---|---|
| Layout regions → blocks | 347 | 347 | 0 |
| Native text regions | 318 | 318 | 0 |
| OCR text regions | 2 | 2 | 0 |
| Phase 5 tables | 6 | 6 | 0 |
| Phase 6 equations | 11 | 11 (1 as failed, with crop) | 0 |
| Phase 7 figures | 16 | 16 | 0 |

Checked independently against the PDF: of 16,047 words in the text layer, 30 (0.19 %) lie outside
every block. They are the page number on nine pages and the ". . ." between equations (4) and (5).

## 4. P1-R — synthetic scanned variant / rasterized digital paper

| Field | Value |
|---|---|
| Paper ID | P1-R |
| Filename | `cancers-15-04172_rasterized_200dpi.pdf` |
| Category | `scanned` (synthetic) |
| Digital / rasterized | Rasterized, 200 dpi grayscale, no added degradation |
| Layout type | As P1 |
| Page count | 29 |
| Sections found | As P1, except 3.1, 4.1 and 4.2 |
| Table count | 6 of 6 |
| Figure count | 16 regions (Figure 8 and Figure 11 split) |
| Equation count | 11 of 11 |
| OCR used | All 317 text regions; mean confidence 0.988 |
| Reading order result | MINOR ERROR (as P1; unattached equation numbers and captions add stray blocks) |
| Section result | PARTIAL |
| Table result | Structure better than P1; CAPTION failure on 5 of 6; one wrong number |
| Figure result | Detection correct; number found for 2 of 11 |
| Equation result | 11 recognized; 5 of 11 numbers attached |
| Caption result | 8 attached, 11 standalone; 9 caption blocks have empty text |
| References result | CORRECT section; 4 entries empty, 2 truncated to their number |
| Cross-page result | 12 joined, as P1 |
| Readiness | NOT READY |

### OCR checks

| Check | Result |
|---|---|
| `has_text_layer` false | Yes, 29 of 29 pages |
| Layout still detects regions | Yes: 350 regions against 347, same types |
| OCR invoked | Yes, 317 regions, 0 engine failures |
| Major headings recovered | All top-level headings and the title, exactly |
| Body paragraphs readable | Yes. Against P1's native text: mean similarity 0.977, about 0.8 % character error |
| Captions readable | No: 9 of 17 caption regions came back empty, 1 truncated |
| Tables survive for Phase 5 | Yes: 6 of 6, numbers correct except one cell |
| Equations survive for Phase 6 | Yes: 11 of 11, including (2), which failed on the digital copy |
| Document reconstruction works | Yes: valid `document.json`, same top-level sections |

**Single-line regions are the failure.** 21 of 317 OCR regions returned no text and 5 more were
cut short. All are one-line regions; multi-line paragraphs are unaffected.

| Lost or truncated | Count | Effect |
|---|---|---|
| Subsection headings (3.1, 4.1, 4.2) | 3 of 14 | Sections not created; their blocks fall into the parent section |
| Table captions | 5 of 6 empty | Tables 1, 3, 4, 5, 6 have no caption or number |
| Figure captions | 4 empty, 1 truncated | Figures 1, 2, 5, 7, 8 have no number |
| Reference entries | 4 empty, 2 reduced to "5." and "77." | Entries lost |
| Equation numbers | 1 empty, 5 read without brackets ("2") | 6 numbers unattached and left in the reading order |
| Short paragraphs | 3 empty, 2 truncated | Includes the correspondence line and most of the author line |

Mentions resolved fall from 29 of 35 to 6 of 35 as a result.

### Tables on the rasterized copy

- Tables 1, 3, 4: identical to the paper.
- Table 2: header groups are assigned correctly (better than P1), but `6381` is read as `631`.
- Tables 5 and 6: the ✓/× marks are in the right rows, unlike P1; one ✓ is missing in Table 6, and
  Table 6 again has no header rows, no row label and an empty last row.

### Equations on the rasterized copy

(2)–(11) are faithful. (1) has one error: `A v g P o o l` is read as `v(g P o o l`. This is
syntactically valid and mathematically wrong, and its status is `success` with no confidence.

### Multimodal integrity

| Modality | Expected | Integrated | Missing |
|---|---|---|---|
| Layout regions → blocks | 350 | 350 | 0 |
| OCR text regions | 317 | 317 (21 with empty text) | 0 |
| Tables | 6 | 6 | 0 |
| Equations | 11 | 11 | 0 |
| Figures | 16 | 16 | 0 |

The information-loss check passes because it compares the document with `ocr.json`, and the text was
already missing there.

## 5. Issues

No BLOCKING issue was found in the paper tested. The untested two-column layout is a gap in the
audit, not a finding.

### MAJOR

**M1 — OCR returns nothing for single-line regions**

| | |
|---|---|
| Paper | P1-R |
| Page | 3, 7, 9, 13, 15, 16, 17, 19, 21, 25, 26, 28 |
| Observed output | 21 one-line regions with empty text, 5 truncated (headings, captions, references, equation numbers) |
| Expected output | The line of text, as for the multi-line regions |
| Likely module | `ocr.py` (`recognize_region`, `PaddleOCREngine._recognize`) |
| Likely cause | Not confirmed. Text detection is run on a thin, wide crop of one line; either it finds no line there, or the line it finds is dropped by the centre-inside-region filter |
| Smallest safe fix | When a region yields no line, recognize the whole crop as one line without detection; alternatively pad thin crops to a minimum height before detection. Confirm the cause first with `--save-ocr-crops` |
| Phase 9 blocked? | YES for scanned input; NO for digital input |

**M2 — ✓/× cells of ablation tables assigned to the wrong rows**

| | |
|---|---|
| Paper | P1 |
| Page | 21 (Tables 5 and 6) |
| Observed output | Cells such as `√ ×` and empty cells; row label "YOLOv7 + CBAM" lost; Table 6 has no header rows and an empty last row |
| Expected output | One mark per cell in its own row |
| Likely module | `tables.py`, hybrid path (native text placed into the vision cell grid) |
| Likely cause | The rows are tightly spaced and the mark glyphs sit lower than the digits, so native words are assigned to the neighbouring row. The vision-only path on P1-R places them correctly |
| Smallest safe fix | Assign a native word to the row whose band contains its vertical centre, using row bands from the numeric columns; drop all-empty trailing rows |
| Phase 9 blocked? | NO. The numbers are correct; ablation attribution from these two tables is unreliable |

**M3 — Captions not attached when figures are stacked or split**

| | |
|---|---|
| Paper | P1 (and P1-R) |
| Page | 11 (Figure 3), 18 (Figures 9, 10), 20 (Figure 11) |
| Observed output | 4 of 11 figures have no caption and no number; the captions stay in the reading order after the figure; Figure 11 is four separate figures |
| Expected output | Each caption attached to the figure above it; Figure 11 as one figure |
| Likely module | `figures.py` (`associate_figure_captions`); `layout.py` for the split |
| Likely cause | A caption between two figures is about equally near both, and the ambiguity rule assigns it to neither. Nothing prefers "caption below the figure". Four sub-images share one caption |
| Smallest safe fix | When a figure caption has one candidate above and one below, prefer the figure above it if the page's other captions are below their figures. Let one caption attach to a row of side-by-side figures as a group |
| Phase 9 blocked? | NO. Caption text is not lost and stays next to its figure |

**M4 — Back matter assigned to Conclusions**

| | |
|---|---|
| Paper | P1 (and P1-R) |
| Page | 25 |
| Observed output | Author Contributions, Funding, Review Board, Consent, Data Availability, Acknowledgments and Conflicts of Interest are paragraphs of `sec_6` (7 of its 11 blocks) |
| Expected output | Back matter outside Conclusions; Acknowledgments typed `acknowledgments` |
| Likely module | `sections.py` |
| Likely cause | The headings are run-in ("Funding: This work …"), which is a documented Phase 8 limitation |
| Smallest safe fix | Treat a paragraph that starts with a known back-matter label followed by a colon as the start of a `back_matter` or `acknowledgments` section, without splitting the paragraph. The same rule would cover "Abstract:" and "Keywords:" |
| Phase 9 blocked? | NO, but a summary of Conclusions would include funding and conflict statements until fixed |

### MINOR

| # | Paper | Issue |
|---|---|---|
| m1 | P1, P1-R | Page 1: margin sidebar read between the author line and the affiliation |
| m2 | P1, P1-R | Two page-1 badges extracted as figures |
| m3 | P1 | Equation (2) rejected for a trailing `\`; the rest of the prediction was usable |
| m4 | both | Function names in equations are letter-spaced (`C o n v`) |
| m5 | both | Equation confidence is always null, so a wrong but valid result cannot be flagged (see m12) |
| m6 | both | Chart data extracted for 0 of 2 charts (gridlines without axis lines; confusion matrix) |
| m7 | both | Page 12 → 13: a continuation starting with "(MLP)" is not joined |
| m8 | both | Bulleted contributions (pages 3–4) are paragraphs, not a list |
| m9 | both | "N of 29" is labelled header on some pages and has no region on 9 |
| m10 | both | Publisher's disclaimer is inside References |
| m11 | P1 | Table 2: group header "Training Images" joined to one column |
| m12 | P1-R | Equation (1): `AvgPool` read as `v(gPool`, status `success` |
| m13 | P1-R | Table 2: `6381` read as `631`; Table 6: one ✓ missing |
| m14 | P1-R | Figure 8 split into two regions |
| m15 | both | MRI image figures typed `diagram` |

## 6. Recommended fixes, in order

1. M1, after confirming its cause on saved crops. It is the only finding that removes structure.
2. M3 and M4. Both are small rule changes and both affect every MDPI-style paper.
3. M2.
4. m3 (strip a trailing `\` before validating) and m5.

None of these has been implemented.

## 7. Phase 9 readiness

| Paper | Readiness |
|---|---|
| P1 (digital) | READY WITH MINOR ISSUES — no blocking error; three major issues that reduce quality without breaking structure |
| P1-R (rasterized) | NOT READY — subsection headings and most captions are lost |

| Condition | P1 | P1-R |
|---|---|---|
| No blocking reconstruction errors | met | met |
| Reading order broadly reliable | met | met |
| Key sections usable | met | top level only |
| References can be excluded | met | met |
| Tables, figures, equations survive integration | met | met |
| Scanned variant gives usable structure | — | partly |

**Overall: PHASE 9 NOT READY.**

- The audit covers one single-column paper. Two-column reading order, full-width blocks between
  columns, wrong-column captions, aligned equations and footnotes were not exercised by it.
- The OCR path loses one-line regions (M1).

For digital single-column papers of this kind the pipeline output is usable as it stands.

## 8. Still needed to complete this phase

- At least one real two-column paper (IEEE, ACM or CVPR style), ideally two.
- One table-heavy, one equation-heavy (with aligned and multi-line equations) and one figure-heavy
  paper; a paper may cover more than one category.
- A paper with footnotes and one with an appendix.
- Optionally a real physical scan, since P1-R is not one.

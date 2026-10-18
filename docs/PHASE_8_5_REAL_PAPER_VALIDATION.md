# Phase 8.5 — Real paper validation

An audit of the Phase 2–8 pipeline (checkpoint `480f751`) on two real research papers, one
single-column and one two-column, and on an image-only copy of each. Nothing in the pipeline was
changed: no fix listed here has been implemented. Phase 9 was not started.

Sections 1–8 are the first checkpoint (commit `a8bdd98`, paper P1 only) and are kept as recorded.
Sections 9–14 add the two-column paper P2 and its rasterized copy P2-R, and supersede the readiness
decision of Section 7.

**STATUS: PHASE 8.5 COMPLETE**

**OVERALL PHASE 9 READINESS: C — NOT READY, PHASE 8.6 FIXES REQUIRED** (Section 14)

Reason, in short:

- two-column reading order is correct on all 46 pages of P2, digital and rasterized;
- References are isolated exactly on both papers;
- the rasterized OCR path loses one-line regions on both papers (Issue 1), now with a confirmed
  cause: the text detector finds nothing in a one-line crop with 8 px padding;
- the digital hybrid table path silently drops or misplaces cells on both papers (Issue 2).

## 1. Validation set

| Paper ID | File | Kind | Category |
|---|---|---|---|
| P1 | `tests/real_papers/digital_single_column/cancers-15-04172.pdf` | Real digital paper | Single column; also contains tables, figures, charts and equations |
| P1-R | `tests/real_papers/scanned/cancers-15-04172_rasterized_200dpi.pdf` | **Synthetic scanned variant / rasterized digital paper** | OCR path |
| P2 | `tests/real_papers/digital_two_column/Marked_changes_for_Prof_Angelov.pdf` | Real digital paper (added in Section 9) | Two column; table-heavy (21), equation-heavy (41), 8 vector figures |
| P2-R | `tests/real_papers/scanned/Marked_changes_for_Prof_Angelov_rasterized_200dpi.pdf` | **Synthetic scanned variant / rasterized digital paper** (added in Section 10) | OCR path, two column |

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
| P2 (digital) | two column, full-width figures and tables | 46 | CORRECT (minor errors only) | PARTIAL (1 spurious level-1 section; 69/69 numbered headings found) | 21/21 detected; text complete; 17 native tables split per line; 3 hybrid tables lose numbers | 8/8 found as 11 regions; 6/8 captioned | 41/41 detected and numbered; 38 recognized | CORRECT (118/118) | READY WITH MINOR ISSUES (5 major, none blocking) |
| P2-R (rasterized) | same | 46 | CORRECT (minor errors only) | FAIL (25 of 69 numbered headings lost) | 21/21; vision path, better structure than P2 | 8/8 found as 9 regions; 7/8 captioned | 42/42 + 1 fragment; 38/41 numbered | CORRECT (118/118, none empty) | NOT READY |

Totals: 2 real papers (1 single-column, 1 two-column), 2 rasterized variants, 150 pages processed
(29 + 29 + 46 + 46).

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

Addressed by Sections 9–14: one real two-column paper that is also table-heavy and equation-heavy,
with footnotes, aligned equations and full-width figures. Still not covered: a figure-heavy paper with
raster figures, a paper with an appendix, and a physical scan.

---

# Expanded validation: two-column paper

Sections 9–14 were added after checkpoint `a8bdd98`. The pipeline is unchanged (HEAD before the
audit: `a8bdd98`). Both files were run with `AIEngine/tools/build_document.py <pdf> --output <dir>
--visualize-order`, default settings, CPU. Outputs are under `tests/real_papers/_output/` and are not
committed.

P2 is a pre-publication manuscript (Elsevier two-column template, LaTeX/pdfTeX, with tracked changes
shown as wavy underlines). It is identified here by file name and hash only; its title and authors
are not reproduced. SHA-256 `6d7a8d8b…57b2ae`. Page numbers below are PDF pages: PDF page 1 is a
"Highlights" page, so the printed page number is the PDF page minus one.

P2-R was made from P2 for this audit exactly as P1-R was: 200 dpi, grayscale, JPEG quality 85, on
pages of the original size, no text layer. It is not a physical scan.

| | P2 | P2-R |
|---|---|---|
| Pages | 46 | 46 |
| Phases 2–7 time | 286 s | 1093 s |
| Reconstruction time | 5.6 s | 7.2 s |
| Exit code | 0 | 0 |
| pages, layout, ocr, tables, equations, figures, document JSON present | all 7 | all 7 |
| Schema, integrity, information-loss checks | pass, 0 problems | pass, 0 problems |

Ground truth was taken from the PDF text layer: 69 numbered headings, 7 unnumbered front- and
back-matter headings, 21 table captions, 8 figure captions, 41 numbered equations and 118 reference
entries.

## 9. P2 — digital two-column paper

| Field | Value |
|---|---|
| Paper ID | P2 |
| Filename | `Marked_changes_for_Prof_Angelov.pdf` |
| Category | `digital_two_column`; also table-heavy and equation-heavy |
| Layout | Two column with full-width title block, abstract, figures and tables; 7 full-page-table or single-column pages |
| Page count | 46 |
| Digital / rasterized | Digital |
| Reading order | CORRECT — 0 column interleavings on 46 pages; minor errors only |
| Sections | PARTIAL — all 69 numbered headings found; 1 spurious level-1 section leaves 2.5 without a parent |
| Tables | 21/21 detected, all text present; STRUCTURE failures in 17 native tables, cells lost in 3 hybrid tables |
| Figures | 8/8 detected as 11 regions; 6/8 with caption and number; chart data 0/2 |
| Equations | 41/41 detected, 41/41 numbers attached; 38 recognized, 3 kept as failed with crops |
| Captions | 27 of 30 caption regions attached; 1 of them is the wrong text (Table 18) |
| References | CORRECT — 118/118 entries, in order, no leakage either way |
| OCR | 1 of 612 text regions (the rest native) |
| Cross-page / cross-column flow | 37 continuations joined, none wrongly; 5 body continuations and 1 reference continuation missed |
| Modality integrity | Complete — tables 21/21, equations 42/42, figures 11/11, all in order and in a section |
| Blocking issues | 0 |
| Major issues | 5 (P2-M1 … P2-M5) |
| Minor issues | 10 |
| Readiness | READY WITH MINOR ISSUES |

### 9.1 Two-column reading order

Checked programmatically on all 46 pages (within each band between full-width blocks, every
right-column block must follow every left-column block) and by eye on the rendered order images of
pages 1–2 (Highlights, title, abstract, first body page), 5 (full-width figure + equations in both
columns), 8–9 (section transitions), 10 (full-width chart), 13 (full-width table + stacked figure),
21 (stacked figure + aligned equation), 32 (two full-width tables), 33 (single-column figure inside
the left column), 39 (back matter and start of References) and 40–46 (two-column bibliography).

| Check | Result |
|---|---|
| Left/right interleaving | None: 0 violations on 39 two-column pages |
| Right column before the left column completes | Never |
| Full-width title, authors, affiliations | Correct, before the abstract |
| Abstract and keywords | Correct, full width, before Section 1 |
| Section headings | In place on every page inspected |
| Full-width figures and tables | Read at their position (top of page, before the columns) on every page |
| Captions | Full-width captions stay with their full-width object; none crosses a column |
| Equations | In the column and position where printed, including both columns on page 5 |
| Page transitions | Correct on all 45 |
| Layout classification | 39 two-column, 7 single-column (full-page tables, the Highlights page, last reference page) — all correct |

| # | Severity | Page | Observation |
|---|---|---|---|
| RO-1 | MINOR | 2 | The corresponding-author and e-mail footnote at the foot of the left column is read in the body, between the left and right columns, as two `OTHER` blocks of Section 1 |
| RO-2 | MINOR | 5 | The bottom of Figure 2 (a "Hybrid §3.7" bar and two legend lines with an equation) is outside the figure region; one fragment becomes a separate EQUATION block between the figure and its caption, and 57 words are in no block |
| RO-3 | MINOR | 1 | The "Highlights" label is in no block |

Independent text check: of 34,579 words in the PDF text layer, 70 (0.20 %) lie outside every block:
57 are RO-2; the rest are RO-3, accent glyphs drawn above inline math (pages 12, 14, 15) and one URL
fragment on page 43.

### 9.2 Sections

| Section | Result |
|---|---|
| Title | CORRECT |
| Abstract | CORRECT (explicit heading) |
| 1 Introduction … 7 Conclusion | CORRECT: all 7 top-level and 62 numbered subsections found (69/69), three levels deep |
| Hierarchy | 68 of 69 numbered sections have the right parent; 2.5 has none (P2-M1) |
| Conclusion | CORRECT: 4 blocks, nothing from the back matter |
| Back matter | Declaration of competing interest, CRediT authorship contribution statement, Funding, Data availability, Declaration of generative AI: each a separate section holding only its own text. Typed `body`, not back matter (m-P2-1) |
| References | CORRECT |
| Appendix | Not present in the paper |

Known Issue 4 (run-in back-matter headings absorbed into Conclusions) does **not** recur: here the
back-matter headings are stand-alone lines and each is recognized. The run-in form is not present, so
the rule that fails on P1 was not exercised.

Two numbered list items ("2. Temporal survival…", "3. Decoder independence…", pages 16–17) were
correctly not taken as section headings.

### 9.3 References

| Check | Result |
|---|---|
| "References" heading detected | Yes, page 39; opens `sec_references` (type `references`) |
| Bibliography assigned to References | 118 of 118 entries, in 119 `reference_content` blocks |
| Order | [1]–[118] strictly increasing through the left and right columns of pages 39–46 |
| Body text in References | None: every block is an entry or the continuation of one |
| References outside the section | None |
| Defect | The continuation of [13] at the top of the right column of page 40 is a separate block (m-P2-2) |

References can be isolated reliably. Not a Phase 9 blocker.

### 9.4 Tables

| Table | Page | Path | Rows × cols | Result |
|---|---|---|---|---|
| 1 | 4 | native | 8 × 8 | Values correct; an extra row of `::` marks from the tracked-change underlines (paper-specific) |
| 2, 5–14, 16, 17, 19–21 | 7–38 | native | 16–38 rows | Text complete and in the right columns; every line of a multi-line cell becomes a new row; no header row detected |
| 3 | 8 | hybrid | 8 × 4 | **7 values lost** ("Input level 3 3 0" → `'', '', '0'`); header "Instances Peer-rev. Preprint" in one cell |
| 4 | 9 | hybrid | 7 × 11 | **7 values lost**; last row shifted one column ("15" joined to the row label); second header row (In … ≥2025) lost |
| 15 | 28 | hybrid | 18 × 7 | 4 words lost; wrapped "Reported scale" text spills into the next row in 5 rows |
| 18 | 32 | hybrid | 9 × 10 | **65.0, 87.4 and "±0.8, not defined" lost**; "Avg." and "Unc." merged into one column; **caption is the table footnote** |

Word-level check (PDF words inside each table box against the words in its extracted cells):

| Path | Tables | Words in PDF | Words missing from cells | Numeric values missing |
|---|---|---|---|---|
| native | 17 | 3,812 | 0 | 0 |
| hybrid | 4 | 682 | 34 | 17 (+ 2 header labels) |

In every table the full native text is still in the table block's `text`, so nothing is lost from
`document.json`; the loss is in the structured `rows`.

- Detection: 21/21. Caption association: 20/21 correct. Table 18 has its footnote as caption, and its
  real caption ("Table 18: …") is a standalone block between Tables 17 and 18 (P2-M3).
- Document placement: every table is in reading order in the section where it is printed.
- Inspected: a simple table (1), borderless booktabs tables (all 21), a two-level header (4), symbol
  cells (1: S/P/–; 18: ×, ±). No merged body cells and no table spanning pages in this paper.
- Table 17 (within-paper ablations, native path): every number is in its correct row; continuation
  lines become extra rows.
- **Known Issue 2 (✓/× rows) could not be re-tested in its exact form: P2 has no ✓/× cells.** The
  same hybrid mechanism (native words placed into the vision cell grid) loses or misplaces cells in
  Tables 3, 4, 15 and 18. On P2-R the vision-only path gets Table 4 fully right, loses 1 cell in
  Table 3 and 2 in Table 18, and keeps logical rows in Table 6.

### 9.5 Figures and captions

| Figure | Page | Regions | Crop | Caption / number | Result |
|---|---|---|---|---|---|
| 1 | 3 | 1, full width | correct | attached, "Figure 1" | CORRECT |
| 2 | 5 | 1, full width | **bottom annotation cut off** (RO-2) | **not attached**; stays after the figure | MAJOR (P2-M4) |
| 3 (timeline chart) | 10 | **2 with an identical box** (`chart` + `diagram`) | correct | **not attached** | MAJOR (P2-M4) |
| 4 | 12 | 1 | correct | attached | CORRECT |
| 5 | 13 | 2, stacked (overview + panels A–E) | correct as a pair | attached to the lower region | MINOR (split) |
| 6 | 18 | 1 | correct | attached | CORRECT |
| 7 | 21 | 2, stacked (overview + panels A–D) | correct as a pair | attached to the lower region | MINOR (split) |
| 8 (scatter chart) | 33 | 1, in the left column | correct | attached | CORRECT |

- Chart data extracted for 0 of 2 charts ("no pair of axis lines found"), as on P1.
- No caption went to the wrong figure; no caption is duplicated.
- **Known Issue 3** recurs once, on Figure 3: two regions with the same box are equally near the
  caption, so the ambiguity rule refuses to choose. The cause here is a duplicate detection, not
  stacking. For the stacked Figures 5 and 7 the caption went to the nearer region, which is
  acceptable. The conservative behaviour is appropriate: in no case would a guess have been more
  correct, and the caption text stays next to its figure.
- Figure 2's caption is the only candidate, below a single figure, yet it is not attached; the stray
  EQUATION fragment between them is the likely reason (not confirmed).

### 9.6 Equations

41 numbered display equations; 42 regions (the extra one is the Figure 2 fragment). All 41 numbers
found and attached; 38 recognized; 3 `invalid_latex`, kept with crop and number, `latex` null.

16 equations compared by eye with the rendered PDF:

| Eq. | Kind | Result |
|---|---|---|
| (2) | simple, bold vectors, Greek | CORRECT |
| (8) | expectation, subscripts, Greek | CORRECT (operator letter-spaced: `l o g`) |
| (18) | fraction, summation, nested sub/superscripts | CORRECT |
| (21) | fraction, summation, two equations on a line | CORRECT (`*` for `⋆`) |
| (31) | arg min with limit, norm | CORRECT |
| (36) | sum, arg max, two equations | CORRECT |
| (39) | conditional entropy, inequality | CORRECT |
| (13) | big union, calligraphic letters | MINOR SYMBOL ERROR: italic v read as ν; calligraphic C read as C |
| (24) | **two-line aligned** | MINOR SYMBOL ERROR: structure right; "action" read as `a c i o n`, `\mathcal\ L` broken in line 2 |
| (28) | conditional probability | MINOR SYMBOL ERROR: `t:t+H-1` read as `t;t+H-1` |
| (25) | arg max, hat, interval | MAJOR SEMANTIC ERROR: ν̂ read as ŷ (wrong variable); status `success` |
| (35) | **two-line aligned**, arg min, sum | MAJOR SEMANTIC ERROR: `c_ctrl` read as `c` & `_{rrr}`, split across the alignment point; status `success` |
| (41) | calligraphic letters | MAJOR SEMANTIC ERROR: calligraphic O and S read as hatted Ô, Ŝ; status `success` |
| (9) | min with limit, divergence | FAILED (`invalid_latex`; garbled near `\|`) |
| (26) | inline words, approx | FAILED (`invalid_latex`) |
| (29) | big oplus | FAILED (`invalid_latex`; trailing `\`, as P1 Eq. (2)) |

Of 16 inspected: 7 correct, 3 minor, 3 major, 3 failed. All 13 non-failed outputs are syntactically
valid LaTeX; 3 of them are mathematically wrong while reported as `success`, and there is no
confidence to flag them (P1 minor m5 recurs). The paper has no matrices or cases environments.

### 9.7 Cross-column and cross-page flow

- 37 paragraph continuations joined across column and page boundaries. None of the joined parts
  starts with a capital letter, and those inspected (pages 3, 5, 13, 21) are correct.
- 5 body continuations not joined (pages 7, 15→16, 29, 33, 39); e.g. page 33: "…SpatialActor, and" |
  "DepthVLA fuse geometry…". 1 reference continuation not joined (page 40). Minor (m-P2-3).
- No paragraph was joined across a section boundary.

### 9.8 Modality integrity

| Modality | Expected | Integrated | Missing |
|---|---|---|---|
| Layout regions → blocks | 665 | 665 | 0 |
| Native text regions | 611 | 611 | 0 |
| OCR text regions | 1 | 1 | 0 |
| Phase 5 tables | 21 | 21 (all in order, all in a section) | 0 |
| Phase 6 equations | 42 | 42 (3 as failed, with crop) | 0 |
| Phase 7 figures | 11 | 11 (all in order, all in a section) | 0 |

No unexplained information loss. The losses that exist are inside structures (hybrid table cells,
Section 9.4) or outside every layout region (RO-2, 57 words).

### 9.9 P2 issues

**MAJOR**

| # | Issue | Pages | Likely module |
|---|---|---|---|
| P2-M1 | A one-sentence paragraph ("Methods are compared along four axes.") detected as `paragraph_title` opens an unnumbered **level-1** section inside 2.4; it takes 2.4's 8 blocks, and 2.5 gets no parent | 8–9 | `sections.py` (an unnumbered heading inside a numbered section is placed at level 1) |
| P2-M2 | Hybrid tables lose or misplace cells: 17 numeric values in Tables 3, 4, 18; row spill in Table 15 (Issue 2 mechanism) | 8, 9, 28, 32 | `tables.py`, hybrid path |
| P2-M3 | Table 18 takes its footnote as caption; its real caption is standalone | 32 | table caption association |
| P2-M4 | Figure 2 and Figure 3 captions not attached (Figure 3 detected twice with an identical box) | 5, 10 | `figures.py`; the duplicate comes from `layout.py` |
| P2-M5 | 3 equations semantically wrong with status `success` (25, 35, 41); both aligned equations have errors | 16, 21, 26 | `math_ocr.py` (model limit) |

**MINOR**

| # | Issue |
|---|---|
| m-P2-1 | Back-matter sections typed `body` |
| m-P2-2 | Continuation of reference [13] not joined |
| m-P2-3 | 5 body continuations not joined |
| m-P2-4 | Native-path tables: one row per text line, no header rows (17 tables; text complete and in the right columns) |
| m-P2-5 | Figures 5 and 7 split into two stacked regions |
| m-P2-6 | Bottom annotation of Figure 2 outside the figure (RO-2) |
| m-P2-7 | Page-2 footnote read in the body (RO-1) |
| m-P2-8 | 3 equations `invalid_latex`; 3 with minor symbol errors |
| m-P2-9 | Chart data 0/2 |
| m-P2-10 | Table 1: row of tracked-change `::` marks (paper-specific) |

m-P2-4 is listed as minor because no value is lost or moved, but a consumer that reads `rows` as
logical rows sees fragments; it is the first minor item to fix.

## 10. P2-R — synthetic scanned variant / rasterized two-column paper

| Field | Value |
|---|---|
| Paper ID | P2-R |
| Filename | `Marked_changes_for_Prof_Angelov_rasterized_200dpi.pdf` |
| Layout | As P2 |
| Page count | 46 |
| Digital / rasterized | Rasterized, 200 dpi grayscale, no added degradation |
| Reading order | CORRECT — 0 column interleavings; same layout classification as P2 |
| Sections | FAIL — 44 of 69 numbered sections; 25 lost; 6 subsections without a parent |
| Tables | 21/21, all vision-only; logical rows (better than P2); 3 cells lost in 2 tables; Table 18 caption as on P2 |
| Figures | 8/8 as 9 regions (Figure 3 no longer duplicated, Figure 7 one region); 7/8 with caption |
| Equations | 43 regions; 42 recognized, 1 failed; 38 of 41 numbers attached |
| Captions | 28 attached, 2 standalone; no caption text lost |
| References | CORRECT — 118/118, none empty, in order |
| OCR | 591 regions, mean confidence 0.987; 35 empty, 7 truncated; multi-line text similarity to P2 0.987 |
| Cross-page flow | 38 joined |
| Modality integrity | Blocks 664/664, tables 21/21, equations 43/43, figures 9/9 |
| Blocking issues | 1 for Phase 9 on scanned input (Issue 1) |
| Readiness | NOT READY |

### 10.1 Known Issue 1 re-test: one-line OCR loss

Every OCR region was matched with the PDF words inside its box. 35 regions returned no text and 7
were cut short; **39 of the 42 failures are one-line regions.**

| Lines in region | Regions | Failed |
|---|---|---|
| 1 | 158 | 39 (25 %) |
| 2–6 | 259 | 3 |
| 7 or more | 174 | 0 |

| Region type | Regions | Failed |
|---|---|---|
| Section headings (`TITLE`) | 78 | **26** |
| Body text (`TEXT`) | 396 | 7 (5 one-line lines or list items, 2 partial) |
| Page numbers | 44 | 7 (empty, or first digit lost) |
| Equation numbers (`OTHER`) | 43 | 2 ("(3)" → "3", "(6)" → "6") |
| Captions | 30 | 0 (all multi-line in this paper) |
| Reference entries | (in `TEXT`) | 0 of 118 entries (all multi-line) |

Lost headings, with crop size in page pixels (180 dpi, 8 px padding included):

| Page | Region | Expected text | Crop w × h | OCR output |
|---|---|---|---|---|
| 3 | heading | 1.1. Relation to existing surveys | 349 × 45 | "" |
| 4 | heading | 1.2. Scope and contributions | 319 × 46 | "" |
| 8 | heading (P2-M1) | Methods are compared along four axes. | 417 × 44 | "" |
| 12 | heading | 3.3.3. Multi-view reprojection into canonical views | 543 × 44 | "" |
| 12 | heading | 3.4.1. Fusion problem and boundary | 398 × 45 | "" |
| 16 | heading | 3.4.7. Trade-offs and open problems | 394 × 45 | "" |
| 16 | heading | 3.5.1. Fusion problem and boundary | 400 × 45 | "" |
| 18 | heading | 3.5.4. Descriptor fields and correspondence | 472 × 45 | "" |
| 19 | heading | 3.5.5. Object-centric and relational representations | 547 × 44 | "" |
| 22 | heading | 3.7. Hybrid and hierarchical fusion | 385 × 45 | "" |
| 22 | heading | 3.7.1. Semantic-geometric fusion across stages | 502 × 44 | "" |
| 24 | heading | 3.7.3. Whole-body fusion across control rates | 487 × 44 | "" |
| 29 | heading | 4.2. A task-oriented taxonomy of resources | 462 × 45 | "" |
| 29 | heading | 4.3. What current protocols measure, and what they do not | 622 × 45 | "" |
| 30 | heading | 5.2. A same-protocol comparison across stages | 506 × 43 | "" |
| 31 | heading | 5.3. Comparative behavior by fusion stage | 457 × 45 | "" |
| 34 | heading | 5.5. Comparative behavior by geometry carrier | 510 × 45 | "" |
| 34 | heading | 5.6. Robustness, transfer, and efficiency | 425 × 44 | "" |
| 34 | heading | 5.7. What the evidence supports, and what it does not | 566 × 46 | "" |
| 34 | heading | 6. Fusion Bottlenecks and Research Directions | 525 × 44 | "" |
| 36 | heading | 6.3. Semantic-geometric grounding across reference frames | 630 × 45 | "" |
| 36 | heading | 6.4. Temporal-spatial fusion under partial observability | 591 × 45 | "" |
| 36 | heading | 6.5. Action-conditioned and contact-aware fusion | 528 × 44 | "" |
| 37 | heading | 6.6. Information loss across abstraction levels | 495 × 45 | "" |
| 37 | heading | 6.9. Evaluating fusion quality rather than task success | 577 × 45 | "" |
| 39 | heading | CRediT authorship contribution statement | 482 × 45 | "" |
| 4 | equation number | (3) | 46 × 42 | "3" |
| 5 | equation number | (6) | 46 × 42 | "6" |
| 14 | page number | 13 | 45 × 42 | "3" |
| 8 | page number | 7 | 34 × 41 | "" |

Other one-line losses: page 9 "• the intersection of these two subsets;" (416 × 42), page 14
"Fusion operator. The simplest operator is positional addition…" (651 × 47), page 22 "with equality
only when geometry is action-redundant." (572 × 45), page 37 two bullet lines (460 × 43, 597 × 43),
and page numbers on pages 12, 16, 18, 20, 26.

Effect: 25 of 69 numbered sections are not created (24 empty headings, plus "1.Introduction", read
without its space and so not parsed as numbered). Their blocks fall into the preceding section (for
example 5.1 holds the text of 5.1–5.3; Declaration of competing interest holds the CRediT statement),
and 3.7.2, 6.1, 6.2, 6.7, 6.8 and 6.10 have no parent.

**Cause — confirmed.** The unchanged engine was run again on 8 failed and 3 successful one-line crops
(read-only use of `AIEngine/pipeline/ocr.py`, no code changed):

| Factor | Finding |
|---|---|
| Crop height | Not the cause. One-line crops that succeed are 38–48 px high (median 44); those that fail 41–47 px (median 44) |
| Crop width | Correlates but does not decide: successful one-line crops have median width 59 px (mostly page and equation numbers), failed ones 462 px; yet 32 wide one-line crops succeed |
| Padding | **The cause.** With the default 8 px padding the text detector returns **0 boxes** for all 8 failed crops. With 24 px it finds the line and the text is read correctly in **8 of 8**; with 48 px also 8 of 8 |
| Preprocessing | Not involved: the crop is passed to the engine as is |
| Recognition | Not the cause: recognition alone on the whole crop, without detection, reads 8 of 8 correctly (scores 0.97–1.00) |
| Confidence filtering | Not the cause: `ocr.py` applies no confidence threshold, and no failed region has lines or raw text in `ocr.json` |
| Result parsing / centre filter | Not the cause: with no detected box there is nothing to parse or filter |

The loss happens in text detection: when one line nearly fills a crop with only 8 px of margin, the
detector finds no text in about half of such regions. The bracket loss on equation numbers and the
first-digit loss on page numbers are a smaller effect of the same tight margin.

## 11. Known issues — generalization

| Issue | Recurrence | Papers affected | Classification |
|---|---|---|---|
| **1 — OCR one-line loss** | 2 of 2 rasterized papers. P1-R: 21 empty + 5 truncated; P2-R: 35 empty + 7 truncated | P1-R, P2-R | **GENERAL** — the cause is in the shared detection step, independent of the paper |
| **2 — table ✓/× row assignment** | ✓/× form: 1 of 1 paper that has ✓/× cells (P2 has none). The underlying hybrid cell-placement fault: 2 of 2 digital papers (P1 Tables 5, 6; P2 Tables 3, 4, 15, 18) | P1, P2 | ✓/× form: UNKNOWN (not re-testable on P2). Hybrid cell placement: **GENERAL** |
| **3 — caption ambiguity** | 2 of 2 digital papers. P1: 4 figures; P2: Figure 3 (duplicate region) and Figure 2 (fragment in between) | P1, P1-R, P2 | **GENERAL**, low impact: caption text is never lost and stays next to its figure. The conservative rule is appropriate and should not be weakened |
| **4 — run-in back-matter headings** | 1 of 1 paper with run-in back matter (P1). P2's back-matter headings are stand-alone and all separate correctly | P1, P1-R | **PAPER-SPECIFIC** to the run-in style (MDPI and similar): it will recur on every paper of that style and does not occur with stand-alone headings |

New on P2, may generalize: P2-M1 (an unnumbered one-line paragraph detected as a heading becomes a
level-1 section and breaks the numbered hierarchy). 1 occurrence; UNKNOWN.

## 12. Two-column result

Two-column reading order — the main open question of the first checkpoint — is **correct** on a real
paper: no interleaving, full-width blocks in place, column and page transitions right, and the
bibliography in order across both columns, on the digital and on the rasterized copy. The defects
found on P2 are in section hierarchy, table structure, caption association and equation recognition,
not in reading order.

## 13. Readiness by paper

| Paper | Readiness |
|---|---|
| P1 (digital, single column) | READY WITH MINOR ISSUES |
| P1-R (rasterized) | NOT READY |
| **P2 (digital, two column)** | **READY WITH MINOR ISSUES** — no blocking error; reading order and References correct; five major issues that lower quality without breaking structure |
| P2-R (rasterized, two column) | NOT READY — 36 % of numbered headings lost |

## 14. Phase 9 decision

**C. PHASE 9 NOT READY — PHASE 8.6 FIXES REQUIRED**

The digital path is usable for Phase 9 on both layouts. Two defects, each confirmed on two papers,
would give Phase 9 wrong input with no signal that anything is wrong:

- on rasterized input, a quarter of one-line regions return no text, so the section structure that
  Phase 9 relies on is lost (Issue 1);
- on digital input, the hybrid table path silently drops or moves numbers in the structured rows
  (Issue 2 mechanism), so any Phase 9 use of table values reads wrong numbers.

Smallest required Phase 8.6 fix set, in priority order (none implemented):

1. **OCR one-line loss (Issue 1).** In `ocr.py`, when detection returns no box for a region, retry
   with a larger padding (24 px) or recognize the whole crop as one line without detection. Both
   recovered 8 of 8 sampled failures. Acceptance: re-run P1-R and P2-R with no empty heading.
2. **Hybrid table cell placement (Issue 2).** In the hybrid path of `tables.py`, assign each native
   word to the cell whose row and column bands contain its centre, and never drop a native word.
   Acceptance: the word-level check of Section 9.4 reports 0 missing words on P1 Tables 5–6 and P2
   Tables 3, 4, 15, 18.

Recommended in the same phase but not required for Phase 9: P2-M1 (an unnumbered heading inside a
numbered section must not open a level-1 section), P2-M3 (a table footnote must not be taken as the
caption), Issue 4 (run-in back-matter headings), and m-P2-4 (join continuation lines of native-path
tables). Issue 3 should stay conservative.

This confirms the "NOT READY" of Section 7, for narrower reasons: two-column layout is no longer an
open risk.

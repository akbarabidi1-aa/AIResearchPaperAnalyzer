"""
Generates the SYNTHETIC fixtures of Phase 8 (reading order and document reconstruction). Needs PyMuPDF,
matplotlib, numpy and Pillow (it reuses the drawings of the equation and figure fixtures). Run from anywhere:

    python tests/fixtures/make_document_fixtures.py

Writes:
    document/synthetic_paper_two_column.pdf     3 pages: full-width title, authors, abstract heading and
                                                abstract, then two columns with numbered sections and
                                                subsections, an equation inside a column, a full-width
                                                figure between column text, a figure inside one column, a
                                                paragraph that continues on the next page, a full-width
                                                table, references; running header and page number
    document/synthetic_paper_single_column.pdf  2 pages in one column: title, abstract, sections with a
                                                subsection, an equation, a paragraph that continues on
                                                the next page, a table, a figure, references
    document/synthetic_papers.json              for both: every block in reading order (type, position,
                                                text, section) and the section tree
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_equation_fixtures as equation_fixtures  # noqa: E402
import make_figure_fixtures as figure_fixtures  # noqa: E402

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "document")

LEFT, RIGHT, TOP = 54.0, 541.0, 72.0
TWO_COLUMNS = ((54.0, 286.0), (309.0, 541.0))
BODY_SIZE, HEADING_SIZE, SUBHEADING_SIZE, CAPTION_SIZE = 9.5, 11.5, 10.0, 8.5
HEADER_TEXT = "Synthetic Journal of Document Analysis 12 (2024)"

SENTENCES = [
    "The approach is evaluated on three collections of scanned and digital documents.",
    "Each page is rendered once and every later stage works on that rendering.",
    "Regions are located first and their content is recognized afterwards.",
    "The order in which regions are read follows the columns of the page.",
    "A block that spans both columns separates the text above it from the text below it.",
    "Captions are attached to the object they describe and are not read a second time.",
    "Running headers and page numbers are kept apart from the body of the document.",
    "All coordinates are stored in points with the origin at the top left corner.",
]


def paragraph(opening, count=3, start=0):
    """Running text that starts with `opening` (unique per block, so that a block can be told apart)."""
    return " ".join([opening] + [SENTENCES[(start + i) % len(SENTENCES)] for i in range(count)])


TABLE_ROWS = [["Dataset", "Pages", "Regions", "Accuracy"], ["Alpha", "120", "1,840", "91.2"],
              ["Beta", "310", "4,975", "88.7"], ["Gamma", "95", "1,102", "93.5"]]


class Paper:
    """Places blocks from top to bottom of the columns of a page and records each one, in reading
    order, as ground truth."""

    def __init__(self, doc, columns):
        self.doc, self.columns = doc, columns
        self.blocks, self.sections, self.section = [], [], None
        self.page, self.cursor = None, []
        self.begin_section("front", None, None, 1, None, "front_matter")

    # -- structure ---------------------------------------------------------------------------------
    def begin_section(self, key, title, number, level, parent, section_type="body"):
        self.sections.append({"key": key, "title": title, "number": number, "level": level, "parent": parent,
                              "section_type": section_type})
        self.section = key

    def new_page(self):
        self.page = self.doc.new_page()
        number = self.page.number + 1
        self.cursor = [TOP for _ in self.columns]
        self.line("header", f"p{number}_header", HEADER_TEXT, (LEFT, 28.0), 8, flow="furniture", type_="HEADER")
        text = str(number)
        width = pymupdf.get_text_length(text, fontname="helv", fontsize=9)
        self.line("number", f"p{number}_page_number", text, ((LEFT + RIGHT - width) / 2, 800.0), 9,
                  flow="furniture", type_="PAGE_NUMBER")

    def record(self, key, label, type_, box, text, flow="main", **extra):
        entry = {"key": key, "page_number": self.page.number + 1, "type": type_, "label": label,
                 "pdf_bbox": [round(v, 2) for v in box], "text": text, "flow": flow,
                 "section": None if flow == "furniture" else self.section, **extra}
        self.blocks.append(entry)
        return entry

    # -- drawing -----------------------------------------------------------------------------------
    def line(self, label, key, text, origin, size, font="helv", flow="main", type_="PARAGRAPH", **extra):
        """One line of text with its top-left corner at `origin`."""
        x, y = origin
        self.page.insert_text((x, y + size * 1.075), text, fontsize=size, fontname=font)
        width = pymupdf.get_text_length(text, fontname=font, fontsize=size)
        return self.record(key, label, type_, (x, y, x + width, y + size * 1.374), text, flow, **extra)

    def span(self, column):
        """Horizontal extent of a column index, or of the whole text area for "full"."""
        return (LEFT, RIGHT) if column == "full" else self.columns[column]

    def top(self, column):
        return max(self.cursor) if column == "full" else self.cursor[column]

    def advance(self, column, y):
        if column == "full":
            self.cursor = [y for _ in self.columns]
        else:
            self.cursor[column] = y

    def text(self, key, text, column, label="text", type_="PARAGRAPH", inset=0.0, gap=9.0, **extra):
        x0, x1 = self.span(column)
        y = self.top(column)
        box = pymupdf.Rect(x0 + inset, y, x1 - inset, y + 400)
        unused = self.page.insert_textbox(box, text, fontsize=BODY_SIZE, lineheight=1.25)
        bottom = y + 400 - unused
        self.advance(column, bottom + gap)
        return self.record(key, label, type_, (x0 + inset, y, x1 - inset, bottom), text, **extra)

    def heading(self, key, text, column, size=HEADING_SIZE, centred=False, label="paragraph_title", type_="HEADING"):
        x0, x1 = self.span(column)
        y = self.top(column) + 3
        width = pymupdf.get_text_length(text, fontname="hebo", fontsize=size)
        x = (x0 + x1 - width) / 2 if centred else x0
        entry = self.line(label, key, text, (x, y), size, font="hebo", type_=type_)
        self.advance(column, y + size * 1.374 + 6)
        return entry

    def caption(self, key, text, column, owner, centred=False):
        x0, x1 = self.span(column)
        y = self.top(column)
        if centred:
            x0 = (x0 + x1 - pymupdf.get_text_length(text, fontname="helv", fontsize=CAPTION_SIZE)) / 2
        entry = self.line("figure_title", key, text, (x0, y), CAPTION_SIZE, flow="attached", type_="CAPTION",
                          attached_to=owner)
        self.advance(column, y + CAPTION_SIZE * 1.374 + 8)
        return entry

    def drawing(self, key, pdf_bytes, column, label, type_, **extra):
        x0, x1 = self.span(column)
        y = self.top(column)
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as source:
            width, height = source[0].rect.width, source[0].rect.height
            x = (x0 + x1 - width) / 2
            box = pymupdf.Rect(x, y, x + width, y + height)
            self.page.show_pdf_page(box, source, 0)
        self.advance(column, y + height + 6)
        return self.record(key, label, type_, box, None, **extra)

    def equation(self, key, spec, column):
        x0, x1 = self.span(column)
        entry = self.drawing(key, equation_fixtures.draw_equation(spec), column, "display_formula", "EQUATION",
                             latex=spec["latex"], equation_number=spec["number"])
        box = entry["pdf_bbox"]
        width = pymupdf.get_text_length(spec["number"], fontname="helv", fontsize=BODY_SIZE)
        self.line("formula_number", key + "_number", spec["number"],
                  (x1 - width, (box[1] + box[3]) / 2 - BODY_SIZE * 0.7), BODY_SIZE, flow="attached", type_="OTHER",
                  attached_to=key)
        self.advance(column, box[3] + 8)
        return entry

    def table(self, key, column, rows=TABLE_ROWS, row_height=17.0):
        x0, x1 = self.span(column)
        y = self.top(column)
        width = (x1 - x0) / len(rows[0])
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                cell_box = pymupdf.Rect(x0 + c * width, y + r * row_height, x0 + (c + 1) * width, y + (r + 1) * row_height)
                self.page.draw_rect(cell_box, color=(0, 0, 0), width=0.6)
                self.page.insert_text((cell_box.x0 + 6, cell_box.y0 + 12), cell, fontsize=9,
                                      fontname="hebo" if r == 0 else "helv")
        bottom = y + len(rows) * row_height
        self.advance(column, bottom + 10)
        return self.record(key, "table", "TABLE", (x0, y, x1, bottom), None, grid=rows)


def references(paper, column, count=3):
    for index in range(1, count + 1):
        paper.text(f"reference_{index}", f"[{index}] A. Author and B. Writer. A study of synthetic documents, part "
                   f"{index}. Proceedings of the Workshop on Fixtures, pages {index}0-{index}9, 202{index}.",
                   column, label="reference_content", gap=4.0)


def build_two_column(doc):
    paper = Paper(doc, TWO_COLUMNS)
    bar = dict(figure_fixtures.BAR_PLAIN)

    paper.new_page()
    paper.heading("title", "Reading Order in Two-Column Scientific Documents", "full", size=16, centred=True,
                  label="doc_title", type_="TITLE")
    authors = "Ada Example, Ben Sample and Cy Placeholder"
    width = pymupdf.get_text_length(authors, fontname="helv", fontsize=10)
    paper.line("text", "authors", authors, ((LEFT + RIGHT - width) / 2, paper.top("full") + 2), 10)
    paper.advance("full", paper.top("full") + 26)
    paper.begin_section("abstract", "Abstract", None, 1, None, "abstract")
    paper.heading("abstract_heading", "Abstract", "full", centred=True)
    paper.text("abstract_text", paragraph("This abstract spans the full width of the page.", 4), "full",
               label="abstract", inset=36.0, gap=16.0)
    paper.begin_section("1", "Introduction", "1", 1, None)
    paper.heading("h_introduction", "1 Introduction", 0)
    paper.text("intro_1", paragraph("Introduction opens in the left column.", 5), 0)
    paper.text("intro_2", paragraph("Second introduction paragraph stays in the left column.", 6, 2), 0)
    paper.text("intro_3", paragraph("Third introduction paragraph closes the left column.", 6, 4), 0)
    paper.text("intro_4", paragraph("Fourth introduction paragraph starts the right column.", 5, 1), 1)
    paper.begin_section("2", "Related Work", "2", 1, None)
    paper.heading("h_related", "2 Related Work", 1)
    paper.text("related_1", paragraph("Related work is discussed in the right column.", 6, 3), 1)
    paper.text("related_2", paragraph("Earlier systems sorted regions from top to bottom.", 6, 5), 1)

    paper.new_page()
    paper.begin_section("3", "Methodology", "3", 1, None)
    paper.heading("h_method", "3 Methodology", 0)
    paper.text("method_1", paragraph("Methodology begins on the second page.", 3, 2), 0)
    paper.begin_section("3.1", "Feature Extraction", "3.1", 2, "3")
    paper.heading("h_features", "3.1 Feature Extraction", 0, size=SUBHEADING_SIZE)
    paper.text("features_1", paragraph("Features are computed for every region as", 1, 6), 0, gap=4.0)
    paper.equation("equation_1", {"latex": r"z = \frac{x+y}{2}", "font": "stix", "number": "(1)"}, 0)
    paper.text("features_2", paragraph("The equation above, Eq. (1), stands inside the left column.", 2, 7), 0)
    paper.text("features_3", paragraph("Right column text above the wide figure.", 6, 1), 1)
    paper.text("features_4", paragraph("More right column text above the wide figure.", 4, 4), 1)
    paper.advance("full", paper.top("full") + 6)
    paper.drawing("figure_wide", figure_fixtures.draw_diagram(["Pages", "Layout", "Text", "Order", "Document"],
                                                              width=440, height=56),
                  "full", "image", "FIGURE", figure_number="Figure 1", caption="Overview of the processing stages.")
    paper.caption("figure_wide_caption", "Figure 1: Overview of the processing stages.", "full", "figure_wide")
    paper.advance("full", paper.top("full") + 6)
    paper.begin_section("3.2", "Training Objective", "3.2", 2, "3")
    paper.heading("h_objective", "3.2 Training Objective", 0, size=SUBHEADING_SIZE)
    paper.text("objective_1", paragraph("Below Figure 1 the left column continues the method of Section 3.1.", 5, 3), 0)
    paper.text("objective_2", paragraph("Another left column paragraph below the figure.", 5, 6), 0)
    paper.text("objective_3", "The chart below sits inside the right column. " + SENTENCES[2] + " The remaining "
               "terms of the objective are weighted by a factor that depends on", 1, continued_by="objective_4")
    paper.drawing("figure_column", figure_fixtures.draw_bar(bar, width=215, height=140, color="#4477aa"), 1, "chart",
                  "CHART", figure_number="Figure 2", caption="Error on three datasets.")
    paper.caption("figure_column_caption", "Figure 2: Error on three datasets.", 1, "figure_column")

    paper.new_page()
    paper.text("objective_4", "the number of training steps completed so far. " + SENTENCES[5] + " " + SENTENCES[0],
               0, continues="objective_3")
    paper.begin_section("4", "Experiments", "4", 1, None)
    paper.heading("h_experiments", "4 Experiments", 0)
    paper.text("experiments_1", paragraph("Experiments are reported in the left column and in Table 1.", 5, 2), 0)
    paper.text("experiments_2", paragraph("Right column text above the wide table.", 6, 5), 1)
    paper.text("experiments_3", paragraph("Last right column paragraph above the table.", 2, 1), 1)
    paper.advance("full", paper.top("full") + 6)
    paper.caption("table_caption", "Table 1: Results on three datasets.", "full", "table_wide")
    paper.table("table_wide", "full")
    paper.blocks[-1]["caption"] = "Table 1: Results on three datasets."
    paper.blocks[-2], paper.blocks[-1] = paper.blocks[-1], paper.blocks[-2]        # the table, then its caption
    paper.advance("full", paper.top("full") + 6)
    paper.begin_section("5", "Conclusion", "5", 1, None)
    paper.heading("h_conclusion", "5 Conclusion", 0)
    paper.text("conclusion_1", paragraph("The conclusion is the last section of the body.", 5, 4), 0)
    paper.text("conclusion_2", paragraph("Final remarks end the left column.", 4, 7), 0)
    paper.begin_section("references", "References", None, 1, None, "references")
    paper.heading("h_references", "References", 1)
    references(paper, 1)
    return paper


def build_single_column(doc):
    paper = Paper(doc, ((LEFT, RIGHT),))
    bar = dict(figure_fixtures.BAR_PLAIN)

    paper.new_page()
    paper.heading("title", "Document Reconstruction in a Single Column", 0, size=16, label="doc_title", type_="TITLE")
    paper.line("text", "authors", "Dee Specimen and Eli Template", (LEFT, paper.top(0)), 10)
    paper.advance(0, paper.top(0) + 24)
    paper.begin_section("abstract", "Abstract", None, 1, None, "abstract")
    paper.heading("abstract_heading", "Abstract", 0)
    paper.text("abstract_text", paragraph("This abstract belongs to the single-column paper.", 4), 0, label="abstract",
               gap=14.0)
    paper.begin_section("1", "Introduction", "1", 1, None)
    paper.heading("h_introduction", "1 Introduction", 0)
    paper.text("intro_1", paragraph("Single-column introduction, first paragraph.", 5), 0)
    paper.text("intro_2", paragraph("Single-column introduction, second paragraph.", 5, 3), 0)
    paper.begin_section("1.1", "Background", "1.1", 2, "1")
    paper.heading("h_background", "1.1 Background", 0, size=SUBHEADING_SIZE)
    paper.text("background_1", paragraph("Background material introduces the relation", 2, 6), 0, gap=4.0)
    paper.equation("equation_1", {"latex": r"E = mc^2", "font": "cm", "number": "(1)"}, 0)
    paper.text("background_2", paragraph("After the equation the background goes on.", 6, 1), 0)
    paper.text("background_3", paragraph("A further background paragraph fills the page.", 7, 4), 0)
    paper.text("background_4", paragraph("One more background paragraph before the page ends.", 7, 6), 0)
    paper.text("background_5", paragraph("The page is almost full at this point.", 6, 2), 0)
    paper.text("background_6", "The last paragraph of the first page is cut by the page break. " + SENTENCES[3]
               + " Its final sentence carries on over the page boundary and", 0, continued_by="background_7")

    paper.new_page()
    paper.text("background_7", "ends only on the second page. " + SENTENCES[6], 0, continues="background_6")
    paper.begin_section("2", "Results", "2", 1, None)
    paper.heading("h_results", "2 Results", 0)
    paper.text("results_1", paragraph("Results are summarized in Table 1 and Figure 1; Table 7 does not exist.", 3, 5), 0)
    paper.caption("table_caption", "Table 1: Results on three datasets.", 0, "table_1")
    paper.table("table_1", 0)
    paper.blocks[-1]["caption"] = "Table 1: Results on three datasets."
    paper.blocks[-2], paper.blocks[-1] = paper.blocks[-1], paper.blocks[-2]
    paper.text("results_2", paragraph("The chart that follows shows the same numbers.", 3, 2), 0)
    paper.drawing("figure_1", figure_fixtures.draw_bar(bar, width=260, height=150, color="#228833"), 0, "chart", "CHART",
                  figure_number="Figure 1", caption="Error on three datasets.")
    paper.caption("figure_1_caption", "Figure 1: Error on three datasets.", 0, "figure_1", centred=True)
    paper.text("results_3", paragraph("A closing paragraph follows the chart.", 3, 7), 0)
    paper.begin_section("references", "References", None, 1, None, "references")
    paper.heading("h_references", "References", 0)
    references(paper, 0)
    return paper


def save(doc, name):
    os.makedirs(HERE, exist_ok=True)
    target = os.path.join(HERE, name)
    doc.save(target, garbage=4, deflate=True, no_new_id=True)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")


def main():
    truth = {}
    for name, build in (("synthetic_paper_two_column", build_two_column),
                        ("synthetic_paper_single_column", build_single_column)):
        with pymupdf.open() as doc:
            paper = build(doc)
            save(doc, name + ".pdf")
            truth[name] = {"pages": doc.page_count, "columns": len(paper.columns), "sections": paper.sections,
                           "blocks": paper.blocks}
    with open(os.path.join(HERE, "synthetic_papers.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"documents": truth}, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()

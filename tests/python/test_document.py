"""Phase 8 tests: column detection, reading order, sections, document.json, its schema and checks, metrics.

    python -m unittest discover -s tests/python -v

The unit tests use a fake layout detector, a fake OCR engine, a fake table structure model and a fake
math recognizer; they need jsonschema but neither PaddleOCR nor a model download. The classes named
*IntegrationTests run the real models of Phases 3-7; they are skipped when paddleocr, tokenizers or ftfy
is not installed or when one of the AIRPA_SKIP_*_MODEL variables is 1.
"""
import copy
import json
import os
import subprocess
import sys
import unittest

from test_equations import HAVE_MATH_MODEL, FakeRecognizer
from test_layout import make_image_only_copy, sha256
from test_ocr import FakeEngine, OcrTestCase
from test_tables import FakeStructureModel

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "AIEngine")
FIXTURES = os.path.join(REPO, "tests", "fixtures")
sys.path.insert(0, ENGINE)

from pipeline import (  # noqa: E402
    DocumentInputError,
    FigureExtractor,
    TableExtractor,
    build_document,
    build_sections,
    check_document_integrity,
    check_information_loss,
    detect_columns,
    evaluate_document,
    exact_order,
    find_mentions,
    is_continuation,
    load_artifacts,
    looks_like_heading,
    order_blocks,
    pairwise_order_accuracy,
    parse_heading,
    run_document_equations,
    run_document_figures,
    run_document_ocr,
    run_document_tables,
    special_section_type,
    validate_document,
)
from pipeline.document_builder import join_text  # noqa: E402

TRUTH_PATH = os.path.join(FIXTURES, "document", "synthetic_papers.json")
with open(TRUTH_PATH, encoding="utf-8") as _fh:
    TRUTH = json.load(_fh)["documents"]
TWO, ONE = "synthetic_paper_two_column", "synthetic_paper_single_column"


def pdf_of(name):
    return os.path.join(FIXTURES, "document", name + ".pdf")


def regions_of(name):
    """The fixture's blocks as layout regions {page: [(model label, PDF box)]}, in REVERSE reading
    order: the builder must not get the order from the order of the regions."""
    regions = {}
    for block in reversed(TRUTH[name]["blocks"]):
        regions.setdefault(block["page_number"], []).append((block["label"], tuple(block["pdf_bbox"])))
    return regions


# Boxes of a two-column page (points): title across, two columns, a wide figure, two columns again.
LEFT, RIGHT = (54, 286), (309, 541)


def box(column, y0, y1):
    return (column[0], y0, column[1], y1)


TWO_COLUMN_PAGE = {
    "title": (120, 70, 480, 90), "l1": box(LEFT, 110, 200), "l2": box(LEFT, 210, 300), "r1": box(RIGHT, 110, 180),
    "r2": box(RIGHT, 190, 300), "figure": (80, 320, 520, 400), "l3": box(LEFT, 420, 600), "r3": box(RIGHT, 420, 500),
    "r4": box(RIGHT, 510, 600),
}
TWO_COLUMN_ORDER = ["title", "l1", "l2", "r1", "r2", "figure", "l3", "r3", "r4"]


def read_order(named):
    names = list(named)
    boxes = [named[name] for name in names]
    layout = detect_columns(boxes, 595.0)
    return [names[i] for i in order_blocks(boxes, layout)], layout


class ColumnDetectionTests(unittest.TestCase):
    def test_single_column(self):
        layout = detect_columns([(54, 100, 541, 200), (54, 210, 541, 300), (54, 80, 160, 95)], 595.0)
        self.assertEqual((layout["layout"], layout["gutters"], layout["columns"]), ("single_column", [], [[54, 541]]))

    def test_two_columns_with_full_width_blocks(self):
        layout = detect_columns(list(TWO_COLUMN_PAGE.values()), 595.0)
        self.assertEqual(layout["layout"], "two_column")
        self.assertEqual(layout["columns"], [[54, 286], [309, 541]])
        self.assertTrue(286 < layout["gutters"][0] < 309)

    def test_gutter_is_between_the_columns_not_beside_a_short_heading(self):
        boxes = [(54, 100, 130, 112), box(LEFT, 120, 300), box(LEFT, 310, 500), box(RIGHT, 100, 300),
                 (309, 310, 380, 322), box(RIGHT, 330, 500)]
        layout = detect_columns(boxes, 595.0)
        self.assertEqual(layout["layout"], "two_column")
        self.assertTrue(286 < layout["gutters"][0] < 309)

    def test_three_columns(self):
        columns = ((40, 200), (215, 375), (390, 550))
        layout = detect_columns([box(c, y, y + 80) for c in columns for y in (100, 200, 300)], 595.0)
        self.assertEqual((layout["layout"], len(layout["columns"]), len(layout["gutters"])), ("multi_column", 3, 2))

    def test_blocks_on_both_sides_that_are_not_side_by_side_are_unknown(self):
        layout = detect_columns([(54, 100, 250, 200), (350, 300, 541, 400)], 595.0)
        self.assertEqual((layout["layout"], layout["gutters"]), ("unknown", []))

    def test_narrow_blocks_do_not_make_columns(self):
        # A single column with equation numbers at the right margin and short headings at the left.
        boxes = [(54, 100, 541, 200), (54, 210, 140, 222), (200, 230, 400, 260), (525, 238, 541, 250),
                 (54, 270, 541, 400)]
        self.assertEqual(detect_columns(boxes, 595.0)["layout"], "single_column")

    def test_empty_page(self):
        self.assertEqual(detect_columns([], 595.0), {"layout": "single_column", "columns": [], "gutters": []})


class ReadingOrderTests(unittest.TestCase):
    def test_single_column_is_read_top_to_bottom(self):
        named = {"c": (54, 300, 541, 400), "a": (54, 80, 300, 95), "b": (54, 100, 541, 290), "d": (150, 410, 450, 500)}
        order, layout = read_order(named)
        self.assertEqual((order, layout["layout"]), (["a", "b", "c", "d"], "single_column"))

    def test_two_columns_with_title_and_wide_figure(self):
        shuffled = {name: TWO_COLUMN_PAGE[name] for name in reversed(TWO_COLUMN_ORDER)}
        order, layout = read_order(shuffled)
        self.assertEqual(layout["layout"], "two_column")
        self.assertEqual(order, TWO_COLUMN_ORDER)

    def test_sorting_by_y_then_x_would_be_wrong(self):
        naive = sorted(TWO_COLUMN_PAGE, key=lambda name: (TWO_COLUMN_PAGE[name][1], TWO_COLUMN_PAGE[name][0]))
        self.assertNotEqual(naive, TWO_COLUMN_ORDER)
        self.assertLess(pairwise_order_accuracy(TWO_COLUMN_ORDER, naive), 1.0)

    def test_wide_table_and_heading_across_the_columns(self):
        named = {"l1": box(LEFT, 100, 200), "r1": box(RIGHT, 100, 200), "table": (54, 220, 541, 300),
                 "heading": (250, 320, 345, 334), "l2": box(LEFT, 350, 500), "r2": box(RIGHT, 350, 500)}
        self.assertEqual(read_order(named)[0], ["l1", "r1", "table", "heading", "l2", "r2"])

    def test_equation_and_figure_inside_a_column_stay_in_it(self):
        named = {"l1": box(LEFT, 100, 200), "equation": (130, 210, 210, 230), "number": (272, 214, 286, 226),
                 "l2": box(LEFT, 240, 400), "r1": box(RIGHT, 100, 200), "figure": (330, 210, 520, 330),
                 "r2": box(RIGHT, 340, 400)}
        self.assertEqual(read_order(named)[0], ["l1", "equation", "number", "l2", "r1", "figure", "r2"])

    def test_unknown_layout_falls_back_to_top_to_bottom(self):
        named = {"b": (350, 300, 541, 400), "a": (54, 100, 250, 200)}
        order, layout = read_order(named)
        self.assertEqual((order, layout["layout"]), (["a", "b"], "unknown"))

    def test_order_metrics(self):
        self.assertEqual(pairwise_order_accuracy("ABCD", "ABCD"), 1.0)
        self.assertEqual(pairwise_order_accuracy("ABCD", "DCBA"), 0.0)
        self.assertAlmostEqual(pairwise_order_accuracy("ABCD", "ABDC"), 5 / 6)
        self.assertAlmostEqual(pairwise_order_accuracy("ABCD", "ABC"), 3 / 6)       # a missing block: its pairs are wrong
        self.assertIsNone(pairwise_order_accuracy("A", "A"))
        self.assertTrue(exact_order("ABC", "AxByC"))
        self.assertFalse(exact_order("ABC", "ACB"))
        self.assertFalse(exact_order("ABC", "AB"))


def block(block_id, text, block_type="PARAGRAPH", label="text", page=1, height=11.0, lines=3):
    return {"block_id": block_id, "type": block_type, "text": text, "page_number": page, "source_label": label,
            "line_height": height, "line_count": lines}


def heading(block_id, text, height=13.0, **options):
    return block(block_id, text, "HEADING", "paragraph_title", height=height, lines=1, **options)


class SectionTests(unittest.TestCase):
    def test_parse_heading(self):
        cases = {
            "3 Methodology": ("3", "Methodology", "numeric"), "3.1 Feature Extraction": ("3.1", "Feature Extraction", "numeric"),
            "2. Related Work": ("2", "Related Work", "numeric"), "3.2.1 Loss": ("3.2.1", "Loss", "numeric"),
            "IV. EXPERIMENTS": ("IV", "EXPERIMENTS", "roman"), "A. Datasets": ("A", "Datasets", "letter"),
            "Appendix B: Proofs": ("B", "Proofs", "appendix"), "Appendix A": ("A", None, "appendix"),
            "Related Work": (None, "Related Work", None), "3 of the 5 runs failed": (None, "3 of the 5 runs failed", None),
            "2020 Results": (None, "2020 Results", None),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = parse_heading(text)
                self.assertEqual((parsed["number"], parsed["title"], parsed["numbering"]), expected)

    def test_special_section_names(self):
        for text, expected in {"Abstract": "abstract", "ABSTRACT": "abstract", "References": "references",
                               "Bibliography": "references", "Acknowledgements": "acknowledgments",
                               "Acknowledgments.": "acknowledgments", "Appendix": "appendix",
                               "Introduction": None, "References to earlier work": None, "": None}.items():
            with self.subTest(text=text):
                self.assertEqual(special_section_type(text), expected)

    def test_unlabelled_line_is_a_heading_only_with_number_or_special_name_and_heading_form(self):
        def line(text, **options):
            return block("b", text, lines=1, **options)

        self.assertTrue(looks_like_heading(line("4 Experiments"), 11.0))
        self.assertTrue(looks_like_heading(line("References"), 11.0))
        self.assertTrue(looks_like_heading(line("II. RELATED WORK"), 11.0))
        self.assertFalse(looks_like_heading(line("Related Work"), 11.0))                    # a name alone is not enough
        self.assertFalse(looks_like_heading(line("4 Experiments were run on each set."), 11.0))
        self.assertFalse(looks_like_heading(block("b", "4 Experiments", lines=2), 11.0))
        self.assertFalse(looks_like_heading(line("4 Experiments", height=8.0), 11.0))       # set smaller than the body
        self.assertFalse(looks_like_heading(line("3 Methodology " + "and more words " * 5), 11.0))

    def sections(self, blocks):
        sections, title = build_sections(blocks)
        return {s["section_id"]: s for s in sections}, title, blocks

    def test_numbered_hierarchy(self):
        sections, title, blocks = self.sections([
            block("t", "A Paper", "TITLE", "doc_title"), block("a", "Authors"),
            heading("h0", "Abstract"), block("p0", "Abstract text."),
            heading("h3", "3 Methodology"), block("p1", "Text."),
            heading("h31", "3.1 Feature Extraction", 11.5), block("p2", "Text."),
            heading("h311", "3.1.1 Details", 11.0), block("p3", "Text."),
            heading("h32", "3.2 Training Objective", 11.5), heading("h4", "4 Experiments"), block("p4", "Text.", page=2),
            heading("hr", "References", page=2), block("r1", "[1] Ref.", label="reference_content", page=2)])
        self.assertEqual(title, "A Paper")
        self.assertEqual(list(sections), ["sec_front_matter", "sec_abstract", "sec_3", "sec_3_1", "sec_3_1_1", "sec_3_2",
                                          "sec_4", "sec_references"])
        self.assertEqual([(s["level"], s["parent_id"]) for s in sections.values()],
                         [(1, None), (1, None), (1, None), (2, "sec_3"), (3, "sec_3_1"), (2, "sec_3"), (1, None), (1, None)])
        self.assertEqual([s["section_type"] for s in sections.values()],
                         ["front_matter", "abstract", "body", "body", "body", "body", "body", "references"])
        self.assertEqual((sections["sec_3_1"]["title"], sections["sec_3_1"]["number"]), ("Feature Extraction", "3.1"))
        self.assertEqual(sections["sec_3_1"]["blocks"], ["h31", "p2"])
        self.assertEqual((sections["sec_4"]["page_start"], sections["sec_4"]["page_end"]), (1, 2))
        self.assertEqual([b["section_id"] for b in blocks if b["block_id"] in ("t", "p3", "r1")],
                         ["sec_front_matter", "sec_3_1_1", "sec_references"])
        self.assertFalse(any(s["implicit"] for s in sections.values()))

    def test_subsection_without_its_parent_gets_no_parent(self):
        sections, _, _ = self.sections([heading("h2", "2 Data"), heading("h41", "4.1 Setup", 11.5)])
        self.assertEqual((sections["sec_4_1"]["level"], sections["sec_4_1"]["parent_id"]), (2, None))

    def test_unnumbered_headings_are_top_level_unless_set_smaller(self):
        sections, _, _ = self.sections([heading("h1", "Introduction"), heading("h2", "Method"),
                                        heading("h3", "Details", 10.0), heading("h4", "Results")])
        self.assertEqual([(s["section_id"], s["level"], s["parent_id"]) for s in sections.values()],
                         [("sec_introduction", 1, None), ("sec_method", 1, None), ("sec_details", 2, "sec_method"),
                          ("sec_results", 1, None)])

    def test_roman_sections_with_lettered_subsections(self):
        sections, _, _ = self.sections([heading("h1", "I. INTRODUCTION"), heading("h2", "II. METHOD"),
                                        heading("ha", "A. Datasets", 11.0), heading("hb", "B. Training", 11.0)])
        self.assertEqual([(s["section_id"], s["level"], s["parent_id"]) for s in sections.values()],
                         [("sec_i", 1, None), ("sec_ii", 1, None), ("sec_ii_a", 2, "sec_ii"), ("sec_ii_b", 2, "sec_ii")])

    def test_labelled_abstract_and_references_without_headings_open_implicit_sections(self):
        sections, _, blocks = self.sections([
            block("t", "Title", "TITLE", "doc_title"), block("a", "Abstract text.", label="abstract"),
            heading("h1", "1 Introduction"), block("p", "Text."),
            block("r1", "[1] Ref.", label="reference_content"), block("r2", "[2] Ref.", label="reference_content")])
        self.assertEqual([(s["section_id"], s["section_type"], s["title"], s["implicit"]) for s in sections.values()],
                         [("sec_front_matter", "front_matter", None, False), ("sec_abstract", "abstract", None, True),
                          ("sec_1", "body", "Introduction", False), ("sec_references", "references", None, True)])
        self.assertEqual(sections["sec_references"]["blocks"], ["r1", "r2"])

    def test_unlabelled_numbered_line_is_promoted_and_long_title_region_demoted(self):
        long_title = heading("x", "This region was labelled as a title but it is a whole sentence of running text "
                             "that goes on and on for far more than a heading ever would in a paper.")
        _, _, blocks = self.sections([block("p0", "Body text.", lines=4), block("h", "2 Related Work", lines=1),
                                      long_title, block("p1", "More body text.", lines=4)])
        self.assertEqual([b["type"] for b in blocks], ["PARAGRAPH", "HEADING", "PARAGRAPH", "PARAGRAPH"])
        self.assertEqual([b["section_id"] for b in blocks], ["sec_front_matter", "sec_2", "sec_2", "sec_2"])

    def test_appendix_and_acknowledgments(self):
        sections, _, _ = self.sections([heading("h1", "1 Introduction"), heading("ha", "Acknowledgments"),
                                        heading("hr", "References"), heading("hx", "Appendix A: Proofs"),
                                        heading("hy", "Appendix B")])
        self.assertEqual([(s["section_id"], s["section_type"]) for s in sections.values()],
                         [("sec_1", "body"), ("sec_acknowledgments", "acknowledgments"), ("sec_references", "references"),
                          ("sec_appendix_a", "appendix"), ("sec_appendix_b", "appendix")])

    def test_repeated_titles_get_distinct_ids(self):
        sections, _, _ = self.sections([heading("h1", "Results"), heading("h2", "Results")])
        self.assertEqual(list(sections), ["sec_results", "sec_results_2"])

    def test_no_title_block_gives_no_title(self):
        self.assertIsNone(self.sections([block("p", "Text.")])[1])


class TextHelperTests(unittest.TestCase):
    def paragraph(self, text, page=1, column=0, section="s", label="text", block_type="PARAGRAPH"):
        return {"type": block_type, "text": text, "page_number": page, "column": column, "section_id": section,
                "source_label": label}

    def test_join_text(self):
        self.assertEqual(join_text("two  words\nnext line \n"), "two words next line")
        self.assertEqual(join_text("- one\n-  two\n", keep_lines=True), "- one\n- two")
        self.assertIsNone(join_text(None))

    def test_continuation_needs_all_its_signals(self):
        first = self.paragraph("The weights depend on")
        self.assertTrue(is_continuation(first, self.paragraph("the number of steps.", column=1)))
        self.assertTrue(is_continuation(first, self.paragraph("the number of steps.", page=2)))
        self.assertTrue(is_continuation(self.paragraph("as he said, “wait"), self.paragraph("for it", page=2)))
        self.assertFalse(is_continuation(first, self.paragraph("the number of steps.")))                    # same column
        self.assertFalse(is_continuation(first, self.paragraph("The number of steps.", column=1)))          # capital
        self.assertFalse(is_continuation(self.paragraph("It ends here."), self.paragraph("and more", column=1)))
        self.assertFalse(is_continuation(self.paragraph("It ends here.”"), self.paragraph("and more", column=1)))
        self.assertFalse(is_continuation(first, self.paragraph("the steps.", column=1, section="other")))
        self.assertFalse(is_continuation(first, self.paragraph("the steps.", column=1, block_type="HEADING")))
        self.assertFalse(is_continuation(self.paragraph("[1] A. Author", label="reference_content"),
                                         self.paragraph("and others", column=1, label="reference_content")))
        self.assertFalse(is_continuation(first, self.paragraph("", column=1)))

    def test_mentions(self):
        mentions = find_mentions("As Fig. 3 and Figure 3 show (see Table 2, Eq. (4), Equation 5 and Section 3.1), Figs. 7")
        self.assertEqual([(m["kind"], m["number"]) for m in mentions],
                         [("figure", "3"), ("table", "2"), ("equation", "4"), ("equation", "5"), ("section", "3.1"),
                          ("figure", "7")])
        self.assertEqual(mentions[0]["label"], "Fig. 3")
        self.assertEqual(find_mentions("No numbered object is named here, only 3 tables."), [])
        self.assertEqual(find_mentions(None), [])


class DocumentTestCase(OcrTestCase):
    ANSWERS = {TWO: r"z=\frac{x+y}{2}", ONE: r"E=m c^{2}"}

    def prepare_document(self, name=TWO, phases=("tables", "equations", "figures"), recognizer=None, pdf=None,
                         work_name="doc"):
        """Phases 2-7 on a fixture with a fake layout (the fixture's own regions) and fake models."""
        work = self.prepare(pdf or pdf_of(name), regions_of(name), work_name)
        engine = FakeEngine()
        run_document_ocr(work, engine)
        if "tables" in phases:
            run_document_tables(work, TableExtractor(FakeStructureModel(), engine))
        if "equations" in phases:
            run_document_equations(work, recognizer or FakeRecognizer([self.ANSWERS[name]]))
        if "figures" in phases:
            run_document_figures(work, FigureExtractor(engine))
        return work

    def build(self, name=TWO, **options):
        self.work = self.prepare_document(name, **options)
        self.result = build_document(self.work)
        return self.result["document"]

    def by_key(self, document, name=TWO):
        evaluation = evaluate_document(TRUTH[name], document)
        blocks = {b["block_id"]: b for b in document["blocks"]}
        return {row["key"]: blocks[row["block_id"]] for row in evaluation["rows"]}


class DocumentBuilderTests(DocumentTestCase):
    def test_both_fixtures_are_reconstructed_exactly(self):
        for name in (TWO, ONE):
            with self.subTest(fixture=name):
                document = self.build(name, work_name=name)
                result = evaluate_document(TRUTH[name], document)
                self.assertEqual((result["found"], result["blocks"]), (len(TRUTH[name]["blocks"]),) * 2)
                self.assertEqual(result["pairwise_order_accuracy"], 1.0)
                self.assertTrue(result["exact_order"])
                self.assertEqual(result["pages_in_exact_order"], result["pages"])
                self.assertEqual(result["layout_pages_correct"], result["pages"])
                for key in ("type_accuracy", "heading_detection_accuracy", "section_assignment_accuracy",
                            "section_hierarchy_accuracy"):
                    self.assertEqual(result[key], 1.0, key)
                self.assertEqual(result["false_headings"], 0)
                self.assertEqual(result["attached_correct"], result["attached"])
                self.assertEqual(result["furniture_correct"], result["furniture"])
                self.assertEqual(result["continuations_joined"], result["continuations"], 1)

    def test_document_json_is_written_valid_and_deterministic(self):
        document = self.build()
        path = os.path.join(self.work, "metadata", "document.json")
        self.assertEqual(self.result["document_path"], path)
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
        self.assertEqual(stored, json.loads(json.dumps(document)))
        self.assertEqual(validate_document(stored), [])
        self.assertEqual(check_document_integrity(stored), [])
        self.assertEqual(stored["document"]["schema_version"], "1.0")
        self.assertEqual(stored["document"]["title"], "Reading Order in Two-Column Scientific Documents")
        self.assertEqual(stored["document"]["page_count"], 3)
        self.assertEqual(set(stored), {"document", "sections", "blocks", "pages", "tables", "figures", "equations",
                                       "metadata", "provenance", "summary"})
        first = sha256(path)
        build_document(self.work)
        self.assertEqual(sha256(path), first)

    def test_every_region_is_one_block_with_a_stable_id(self):
        document = self.build()
        with open(os.path.join(self.work, "metadata", "layout.json"), encoding="utf-8") as fh:
            regions = [r for page in json.load(fh)["pages"] for r in page["regions"]]
        self.assertEqual([b["source_region_id"] for b in document["blocks"]], [r["region_id"] for r in regions])
        self.assertEqual([b["bbox"] for b in document["blocks"]], [r["pdf_bbox"] for r in regions])
        ids = [b["block_id"] for b in document["blocks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "page_001_block_001")
        self.assertRegex(ids[-1], r"^page_003_block_\d{3}$")
        self.assertEqual((document["tables"][0]["table_id"], document["figures"][1]["figure_id"],
                          document["equations"][0]["equation_id"]),
                         ("page_003_table_001", "page_002_figure_002", "page_002_equation_001"))

    def test_two_column_pages_and_full_width_blocks(self):
        document = self.build()
        self.assertEqual([page["layout"] for page in document["pages"]], ["two_column"] * 3)
        self.assertEqual(document["summary"]["layout"], {"two_column": 3})
        blocks = self.by_key(document)
        for key in ("title", "authors", "abstract_heading", "abstract_text", "figure_wide", "table_wide"):
            self.assertTrue(blocks[key]["full_width"], key)
            self.assertIsNone(blocks[key]["column"], key)
        self.assertEqual([blocks[key]["column"] for key in ("intro_1", "intro_4", "equation_1", "figure_column")],
                         [0, 1, 0, 1])
        order = {key: b["order"] for key, b in blocks.items()}
        self.assertTrue(order["features_2"] < order["features_3"] < order["features_4"] < order["figure_wide"]
                        < order["h_objective"] < order["objective_2"] < order["objective_3"] < order["figure_column"])
        self.assertTrue(order["experiments_3"] < order["table_wide"] < order["h_conclusion"] < order["h_references"])

    def test_single_column_document(self):
        document = self.build(ONE)
        self.assertEqual([page["layout"] for page in document["pages"]], ["single_column"] * 2)
        self.assertTrue(all(b["column"] == 0 for b in document["blocks"] if b["flow"] == "main"))

    def test_sections_of_the_two_column_fixture(self):
        document = self.build()
        sections = document["sections"]
        self.assertEqual([s["section_id"] for s in sections],
                         ["sec_front_matter", "sec_abstract", "sec_1", "sec_2", "sec_3", "sec_3_1", "sec_3_2", "sec_4",
                          "sec_5", "sec_references"])
        self.assertEqual([(s["level"], s["parent_id"]) for s in sections if s["number"] in ("3.1", "3.2")],
                         [(2, "sec_3"), (2, "sec_3")])
        self.assertEqual({s["section_type"] for s in sections}, {"front_matter", "abstract", "body", "references"})
        self.assertEqual((sections[6]["page_start"], sections[6]["page_end"]), (2, 3))      # 3.2 runs over the page break
        references = sections[-1]
        self.assertEqual((references["title"], references["section_type"], len(references["blocks"])),
                         ("References", "references", 4))
        self.assertEqual((document["summary"]["sections"], document["summary"]["subsections"]), (7, 2))

    def test_order_is_global_and_continues_over_pages(self):
        document = self.build()
        main = sorted((b for b in document["blocks"] if b["flow"] == "main"), key=lambda b: b["order"])
        self.assertEqual([b["order"] for b in main], list(range(len(main))))
        self.assertEqual([b["page_number"] for b in main], sorted(b["page_number"] for b in main))
        self.assertEqual([b["block_id"] for b in main], [i for page in document["pages"] for i in page["block_ids"]])
        self.assertEqual([i for s in document["sections"] for i in s["blocks"]], [b["block_id"] for b in main])

    def test_paragraph_cut_by_the_page_break_is_joined(self):
        document = self.build()
        blocks = self.by_key(document)
        head, tail = blocks["objective_3"], blocks["objective_4"]
        self.assertEqual((tail["flow"], tail["merged_into"], tail["order"]), ("merged", head["block_id"], None))
        self.assertEqual([part["block_id"] for part in head["parts"]], [head["block_id"], tail["block_id"]])
        self.assertEqual([part["page_number"] for part in head["parts"]], [2, 3])
        self.assertIn("a factor that depends on the number of training steps completed so far.", head["text"])
        self.assertEqual(head["text"], head["parts"][0]["text"] + " " + tail["text"])
        self.assertTrue(tail["text"].startswith("the number of training steps"))       # the part keeps its own text
        self.assertEqual(tail["section_id"], head["section_id"], "sec_3_2")
        self.assertNotIn(tail["block_id"], [i for s in document["sections"] for i in s["blocks"]])
        self.assertEqual(document["summary"]["paragraphs_merged"], 1)
        self.assertEqual(sum(1 for b in document["blocks"] if "parts" in b), 1)        # nothing else was joined

    def test_table_is_integrated_as_a_table(self):
        document = self.build()
        table, blocks = document["tables"][0], self.by_key(document)
        block, caption = blocks["table_wide"], blocks["table_caption"]
        self.assertEqual((block["type"], block["table_id"], block["caption"]),
                         ("TABLE", table["table_id"], "Table 1: Results on three datasets."))
        self.assertEqual((table["status"], table["source_id"], table["block_id"]), ("ok", "p003_t001", block["block_id"]))
        self.assertEqual(table["headers"], ["Dataset", "Pages", "Regions", "Accuracy"])
        self.assertEqual(table["rows"], [["Alpha", "120", "1,840", "91.2"], ["Beta", "310", "4,975", "88.7"],
                                         ["Gamma", "95", "1,102", "93.5"]])
        self.assertEqual((table["n_rows"], table["n_columns"], table["extraction_source"]), (4, 4, "native"))
        self.assertEqual(table["caption_block_id"], caption["block_id"])
        self.assertEqual((caption["flow"], caption["attached_to"], caption["order"]), ("attached", block["block_id"], None))
        self.assertEqual(caption["section_id"], block["section_id"], "sec_4")

    def test_equation_is_integrated_with_its_number(self):
        document = self.build()
        equation, blocks = document["equations"][0], self.by_key(document)
        block, number = blocks["equation_1"], blocks["equation_1_number"]
        self.assertEqual((block["type"], block["latex"], block["equation_number"], block["status"]),
                         ("EQUATION", r"z=\frac{x+y}{2}", "(1)", "success"))
        self.assertEqual((equation["equation_id"], equation["block_id"]), (block["equation_id"], block["block_id"]))
        self.assertEqual(equation["equation_number_block_id"], number["block_id"])
        self.assertEqual((number["flow"], number["attached_to"], number["text"]), ("attached", block["block_id"], "(1)"))
        self.assertEqual(block["section_id"], "sec_3_1")
        self.assertIsNone(block["text"])

    def test_equation_that_was_not_recognized_is_kept_with_its_image(self):
        document = self.build(recognizer=FakeRecognizer([RuntimeError("simulated failure")]))
        equation = document["equations"][0]
        self.assertEqual((equation["status"], equation["latex"]), ("recognition_failed", None))
        self.assertTrue(os.path.isfile(os.path.join(self.work, *equation["image_path"].split("/"))))
        self.assertIn("simulated failure", equation["error"])
        block = self.by_key(document)["equation_1"]
        self.assertEqual((block["type"], block["latex"], block["status"], block["flow"]),
                         ("EQUATION", None, "recognition_failed", "main"))
        counts = document["provenance"]["counts"]
        self.assertEqual((counts["equations_not_recognized_source"], counts["equations_not_recognized"]), (1, 1))

    def test_figures_and_charts_are_integrated(self):
        document = self.build()
        blocks = self.by_key(document)
        wide, chart = blocks["figure_wide"], blocks["figure_column"]
        self.assertEqual((wide["type"], wide["figure_type"], wide["figure_number"], wide["caption"]),
                         ("FIGURE", "diagram", "Figure 1", "Overview of the processing stages."))
        self.assertEqual((chart["type"], chart["figure_type"], chart["figure_number"]), ("CHART", "chart", "Figure 2"))
        by_id = {figure["figure_id"]: figure for figure in document["figures"]}
        diagram, bars = by_id[wide["figure_id"]], by_id[chart["figure_id"]]
        self.assertEqual(diagram["labels"], ["Pages", "Layout", "Text", "Order", "Document"])
        self.assertIsNone(diagram["chart_data"])
        self.assertEqual(bars["chart_data"]["chart_type"], "bar")
        self.assertEqual([bar["category"] for bar in bars["chart_data"]["series"][0]["bars"]], ["Set A", "Set B", "Set C"])
        for figure in (diagram, bars):
            self.assertTrue(os.path.isfile(os.path.join(self.work, *figure["image_path"].split("/"))))
            caption = "figure_wide_caption" if figure is diagram else "figure_column_caption"
            self.assertEqual(blocks[caption]["block_id"], figure["caption_block_id"])
            self.assertFalse({"description", "summary", "interpretation"} & set(figure))
        self.assertEqual((document["summary"]["figures"], document["summary"]["charts"]), (1, 1))

    def test_attached_captions_are_not_read_a_second_time(self):
        document = self.build()
        main = [b for b in document["blocks"] if b["flow"] == "main"]
        self.assertFalse(any(b["type"] == "CAPTION" for b in main))
        texts = [b["text"] for b in main if b["text"]]
        for caption in ("Figure 1: Overview of the processing stages.", "Table 1: Results on three datasets."):
            self.assertFalse(any(caption in text for text in texts), caption)
            self.assertEqual(sum(1 for b in document["blocks"] if b["text"] == caption), 1)    # kept once, attached
        self.assertEqual((document["summary"]["captions_attached"], document["summary"]["captions_standalone"]), (3, 0))

    def test_headers_and_page_numbers_stay_in_metadata_only(self):
        document = self.build()
        metadata = document["metadata"]
        self.assertEqual([entry["text"] for entry in metadata["headers"]],
                         ["Synthetic Journal of Document Analysis 12 (2024)"] * 3)
        self.assertEqual([(entry["page_number"], entry["text"]) for entry in metadata["page_numbers"]],
                         [(1, "1"), (2, "2"), (3, "3")])
        furniture = [b for b in document["blocks"] if b["flow"] == "furniture"]
        self.assertEqual(len(furniture), 6)
        self.assertTrue(all(b["order"] is None and b["section_id"] is None for b in furniture))
        listed = {i for s in document["sections"] for i in s["blocks"]} | {i for p in document["pages"] for i in p["block_ids"]}
        self.assertFalse(listed & {b["block_id"] for b in furniture})
        self.assertEqual(document["pages"][0]["header_block_ids"], [metadata["headers"][0]["block_id"]])

    def test_mentions_point_to_the_blocks_they_name(self):
        document = self.build()
        blocks = self.by_key(document)
        self.assertEqual(blocks["features_2"]["references"],
                         [{"kind": "equation", "label": "Eq. (1)", "target_block_id": blocks["equation_1"]["block_id"]}])
        self.assertEqual([(r["kind"], r["target_block_id"]) for r in blocks["objective_1"]["references"]],
                         [("figure", blocks["figure_wide"]["block_id"]), ("section", blocks["h_features"]["block_id"])])
        self.assertEqual(blocks["experiments_1"]["references"][0]["target_block_id"], blocks["table_wide"]["block_id"])
        single = self.build(ONE, work_name="single")
        mentions = self.by_key(single, ONE)["results_1"]["references"]
        self.assertEqual([(r["label"], r["target_block_id"] is not None) for r in mentions],
                         [("Table 1", True), ("Figure 1", True), ("Table 7", False)])       # nothing is made up for Table 7

    def test_text_has_one_value_with_its_source(self):
        document = self.build()
        with open(os.path.join(self.work, "metadata", "ocr.json"), encoding="utf-8") as fh:
            regions = {r["region_id"]: r for page in json.load(fh)["pages"] for r in page["regions"]}
        for b in document["blocks"]:
            source = regions[b["source_region_id"]]
            if source["text_source"] == "none":
                self.assertEqual((b["text"], b["source"]), (None, None))
            else:
                self.assertEqual((b["source"], b["confidence"]), ("native", None))
                own = b["parts"][0]["text"] if "parts" in b else b["text"]
                self.assertEqual(own.split(), source["text"].split())
        counts = document["provenance"]["counts"]
        self.assertEqual(counts["native_text_blocks"], counts["native_text_regions"])
        self.assertEqual((counts["ocr_text_regions"], counts["problems"]), (0, []))

    def test_ocr_text_is_used_where_there_is_no_native_text(self):
        scanned = self.path("scanned.pdf")
        make_image_only_copy(pdf_of(ONE), scanned)
        document = self.build(ONE, pdf=scanned, phases=())
        texts = [b for b in document["blocks"] if b["text"] is not None]
        self.assertTrue(texts)
        self.assertTrue(all(b["source"] == "ocr" and b["confidence"] == FakeEngine.CONFIDENCE for b in texts))
        counts = document["provenance"]["counts"]
        self.assertEqual((counts["ocr_text_blocks"], counts["native_text_regions"]), (counts["ocr_text_regions"], 0))
        self.assertGreater(counts["ocr_text_regions"], 20)

    def test_without_the_extraction_phases_the_objects_are_kept_as_not_extracted(self):
        document = self.build(phases=())
        self.assertEqual(document["provenance"]["artifacts"]["tables"], None)
        self.assertEqual([t["status"] for t in document["tables"]], ["not_extracted"])
        self.assertEqual([e["status"] for e in document["equations"]], ["not_extracted"])
        self.assertEqual([f["status"] for f in document["figures"]], ["not_extracted"] * 2)
        self.assertEqual(document["summary"]["captions_standalone"], 3)         # nothing paired them: they are read
        self.assertEqual(validate_document(document) + check_document_integrity(document), [])
        counts = document["provenance"]["counts"]
        self.assertEqual((counts["tables_source"], counts["tables"], counts["figures"]), (None, 1, 2))

    def test_no_information_is_lost(self):
        document = self.build()
        artifacts = load_artifacts(self.work)
        counts = check_information_loss(document, artifacts)
        self.assertEqual(counts["problems"], [])
        self.assertEqual((counts["layout_regions"], counts["blocks"]), (49, 49))
        self.assertEqual((counts["tables_source"], counts["tables"], counts["equations_source"], counts["equations"],
                          counts["figures_source"], counts["figures"]), (1, 1, 1, 1, 2, 2))
        self.assertEqual((counts["tables_captions_source"], counts["tables_captions_attached"],
                          counts["figures_captions_source"], counts["figures_captions_attached"]), (1, 1, 2, 2))

        def problems(change):
            broken = copy.deepcopy(document)
            change(broken)
            return check_information_loss(broken, artifacts)["problems"]

        self.assertTrue(problems(lambda d: d["blocks"].pop(5)))
        self.assertTrue(problems(lambda d: d["tables"].clear()))
        self.assertTrue(problems(lambda d: d["figures"].pop()))
        self.assertTrue(problems(lambda d: d["equations"].clear()))
        self.assertTrue(problems(lambda d: d["blocks"][3].update(text="other words")))
        self.assertTrue(problems(lambda d: d["figures"][0].update(caption_block_id=None)))

    def test_malformed_documents_fail_the_schema_with_a_clear_message(self):
        document = self.build()

        def errors(change):
            broken = copy.deepcopy(document)
            change(broken)
            return validate_document(broken)

        self.assertIn("'sections' is a required property", errors(lambda d: d.pop("sections"))[0])
        message = errors(lambda d: d["blocks"][3].update(type="SIDEBAR"))[0]
        self.assertTrue(message.startswith("blocks/3/type: 'SIDEBAR' is not one of"), message)
        self.assertTrue(errors(lambda d: d["blocks"][3].pop("bbox"))[0].startswith("blocks/3: 'bbox' is a required"))
        self.assertTrue(errors(lambda d: d["blocks"][3].update(bbox=[1, 2, 3])))
        self.assertTrue(errors(lambda d: d["blocks"][3].update(order="7")))
        self.assertTrue(errors(lambda d: d["blocks"][3].update(note="x")))                       # unknown field
        self.assertTrue(errors(lambda d: next(b for b in d["blocks"] if b["flow"] == "furniture").update(order=3)))
        self.assertTrue(errors(lambda d: d["document"].update(schema_version="2.0")))
        self.assertTrue(errors(lambda d: d["sections"][1].update(section_type="intro")))
        self.assertTrue(errors(lambda d: d["sections"][1].update(level=0)))
        self.assertTrue(errors(lambda d: d["tables"][0].pop("rows")))
        self.assertTrue(errors(lambda d: d["equations"][0].update(status="maybe")))
        self.assertTrue(errors(lambda d: d["pages"][0].update(layout="three")))
        self.assertTrue(errors(lambda d: next(b for b in d["blocks"] if b["type"] == "TABLE").pop("table_id")))
        self.assertTrue(errors(lambda d: d["blocks"][3].update(table_id="page_001_table_001")))  # a paragraph with a table
        self.assertTrue(errors(lambda d: d["provenance"]["counts"]["problems"].append("lost")))

    def test_integrity_checks_find_inconsistent_documents(self):
        document = self.build()

        def problems(change):
            broken = copy.deepcopy(document)
            change(broken)
            return check_document_integrity(broken)

        def main(d, index):
            return [b for b in d["blocks"] if b["flow"] == "main"][index]

        self.assertTrue(problems(lambda d: main(d, 4).update(order=main(d, 5)["order"])))            # an order twice
        self.assertTrue(problems(lambda d: main(d, 4).update(section_id="sec_missing")))
        self.assertTrue(problems(lambda d: d["blocks"][3].update(block_id=d["blocks"][4]["block_id"])))
        self.assertTrue(problems(lambda d: d["blocks"][3].update(page_number=9)))
        self.assertTrue(problems(lambda d: d["blocks"][3].update(bbox=[0, 0, 9000, 10])))
        self.assertTrue(problems(lambda d: d["sections"][5].update(parent_id="sec_nowhere")))
        self.assertTrue(problems(lambda d: d["sections"][2]["blocks"].pop()))                        # a block in no section
        self.assertTrue(problems(lambda d: d["tables"][0].update(block_id="page_001_block_001")))
        self.assertTrue(problems(lambda d: d["figures"][0].update(caption_block_id=d["tables"][0]["caption_block_id"])))
        self.assertTrue(problems(lambda d: next(b for b in d["blocks"] if b["references"])["references"][0].update(
            target_block_id="page_009_block_001")))
        self.assertTrue(problems(lambda d: next(b for b in d["blocks"] if b["flow"] == "attached").pop("attached_to")))
        self.assertTrue(problems(lambda d: d["pages"][1]["block_ids"].reverse()))

    def test_other_artifacts_are_not_modified(self):
        self.work = self.prepare_document()
        metadata = os.path.join(self.work, "metadata")
        before = {name: sha256(os.path.join(metadata, name)) for name in sorted(os.listdir(metadata))}
        self.assertEqual(len(before), 6)
        build_document(self.work)
        after = {name: sha256(os.path.join(metadata, name)) for name in before}
        self.assertEqual(after, before)
        self.assertEqual(sorted(os.listdir(metadata)), sorted(list(before) + ["document.json"]))
        self.assertFalse(os.path.exists(os.path.join(self.work, "order_debug")))

    def test_reading_order_images_only_on_request(self):
        self.work = self.prepare_document()
        result = build_document(self.work, visualize=True)
        self.assertEqual([os.path.basename(path) for path in result["visualizations"]],
                         ["page_001_order.png", "page_002_order.png", "page_003_order.png"])
        self.assertTrue(all(os.path.getsize(path) > 0 for path in result["visualizations"]))

    def test_missing_or_foreign_artifacts_are_refused(self):
        self.work = self.prepare_document()
        build_document(self.work)
        target = os.path.join(self.work, "metadata", "document.json")
        tables = os.path.join(self.work, "metadata", "tables.json")
        with open(tables, encoding="utf-8") as fh:
            foreign = json.load(fh)
        foreign["source"]["source_sha256"] = "0" * 64
        with open(tables, "w", encoding="utf-8") as fh:
            json.dump(foreign, fh)
        with self.assertRaises(DocumentInputError):
            build_document(self.work)
        self.assertFalse(os.path.exists(target))            # no stale document.json survives
        os.remove(tables)
        os.remove(os.path.join(self.work, "metadata", "ocr.json"))
        with self.assertRaises(DocumentInputError):
            build_document(self.work)

    def test_wrong_order_lowers_the_metrics(self):
        document = copy.deepcopy(self.build())
        main = sorted((b for b in document["blocks"] if b["flow"] == "main"), key=lambda b: b["order"])
        main[6]["order"], main[9]["order"] = main[9]["order"], main[6]["order"]
        next(b for b in main if b["type"] == "HEADING")["type"] = "PARAGRAPH"
        result = evaluate_document(TRUTH[TWO], document)
        self.assertLess(result["pairwise_order_accuracy"], 1.0)
        self.assertFalse(result["exact_order"])
        self.assertEqual(result["pages_in_exact_order"], 2)
        self.assertLess(result["heading_detection_accuracy"], 1.0)
        self.assertEqual(result["headings_detected"], result["headings"] - 1)

    def test_no_research_intelligence_fields(self):
        text = json.dumps(self.build()).lower()
        for field in ('"summary_text"', '"research_problem"', '"contributions"', '"limitations"', '"method_summary"',
                      '"dataset_summary"', '"keywords"'):
            self.assertNotIn(field, text)


class BuildDocumentToolErrorTests(OcrTestCase):
    def test_missing_pdf_exits_with_2(self):
        completed = subprocess.run([sys.executable, os.path.join(ENGINE, "tools", "build_document.py"),
                                    self.path("missing.pdf"), "--output", self.path("out")],
                                   capture_output=True, text=True, timeout=600)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Error:", completed.stderr)
        self.assertFalse(os.path.exists(self.path("out", "metadata", "document.json")))


@unittest.skipUnless(HAVE_MATH_MODEL, "paddleocr, tokenizers or ftfy is missing, or the model tests are switched off")
class BuildDocumentToolIntegrationTests(OcrTestCase):
    """The whole pipeline, Phases 2-8, with the real models, through the command-line tool."""

    def run_tool(self, *arguments):
        return subprocess.run([sys.executable, os.path.join(ENGINE, "tools", "build_document.py"), *arguments],
                              capture_output=True, text=True, timeout=900)

    def check(self, name, expected_lines):
        work = self.path(name)
        completed = self.run_tool(pdf_of(name), "--output", work, "--visualize-order", "--truth", TRUTH_PATH)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        for line in expected_lines + ("Schema valid: yes", "Pairwise order accuracy: 1.000", "whole document: yes",
                                      "Heading detection accuracy: 1.000", "Section assignment accuracy: 1.000",
                                      "Section hierarchy accuracy: 1.000", "problems 0",
                                      "Phase 5 tables: created; Phase 6 equations: created; Phase 7 figures: created"):
            self.assertIn(line, completed.stdout)
        with open(os.path.join(work, "metadata", "document.json"), encoding="utf-8") as fh:
            document = json.load(fh)
        self.assertEqual(validate_document(document) + check_document_integrity(document), [])
        self.assertEqual(len(os.listdir(os.path.join(work, "order_debug"))), len(document["pages"]))
        return work, document

    def test_two_column_paper(self):
        work, document = self.check(TWO, ("Pages: 3", "Sections: 7", "Subsections: 2", "Tables: 1", "Figures: 1",
                                          "Charts: 1", "Equations: 1", "two-column pages: 3",
                                          "Paragraph continuations joined: 1/1"))
        self.assertEqual(document["equations"][0]["status"], "success")
        self.assertEqual(document["equations"][0]["equation_number"], "(1)")
        self.assertEqual(document["summary"]["captions_standalone"], 0)
        again = self.run_tool(work)
        self.assertEqual(again.returncode, 0, again.stderr[-2000:])
        self.assertIn("Phase 4 OCR: reused; Phase 5 tables: reused; Phase 6 equations: reused; Phase 7 figures: reused",
                      again.stdout)

    def test_single_column_paper(self):
        self.check(ONE, ("Pages: 2", "Sections: 4", "Subsections: 1", "Tables: 1", "Charts: 1", "Equations: 1",
                         "single-column pages: 2", "Paragraph continuations joined: 1/1"))


if __name__ == "__main__":
    unittest.main()

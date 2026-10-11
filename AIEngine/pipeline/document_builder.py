"""Phase 8 document reconstruction: the artifacts of Phases 2-7 -> one ordered, sectioned document.json.

    layout.json   every region becomes exactly one canonical block (nothing is dropped)
    ocr.json      the block's one text: native where the PDF has it, OCR otherwise
    tables.json, equations.json, figures.json
                  TABLE / EQUATION / FIGURE / CHART blocks point to the extracted object; its caption
                  and its equation number are attached to it and leave the reading order
    reading_order page by page: columns, full-width blocks, top to bottom -> one global order
    sections      headings, section tree, a section for every block
    paragraphs    a paragraph cut by a column or page break is joined, when the text says so
    references    "Figure 3", "Table 2", "Eq. (4)", "Section 3.1" in the text -> the block meant
        ->  <work_dir>/metadata/document.json, checked against AIEngine/schema/document.schema.json

Structure only. No summary, no research problem, no contributions: nothing here says what the
document means.
"""
import json
import os
import re
import statistics
import time

from .common import PdfPipelineError, ensure_directory
from .figures import FIGURES_JSON, TYPE_CHART, validate_figures_document
from .layout import LAYOUT_JSON, RegionType, load_pages_document
from .layout_eval import bbox_iou, is_valid_bbox
from .math_ocr import EQUATIONS_JSON, validate_equations_document
from .ocr import (
    OCR_JSON,
    TEXT_SOURCE_NATIVE,
    TEXT_SOURCE_OCR,
    group_rows,
    load_layout_document,
    native_lines_in_region,
)
from .page_pipeline import METADATA_DIR, PAGES_JSON, _write_json
from .reading_order import LAYOUTS, column_index, detect_columns, order_blocks, spans_columns
from .sections import TYPE_HEADING, TYPE_PARAGRAPH, TYPE_TITLE, build_sections
from .tables import TABLES_JSON, _remove_stale, load_ocr_document, validate_tables_document

DOCUMENT_SCHEMA_VERSION = "1.0"
DOCUMENT_JSON = "document.json"
ORDER_DEBUG_DIR = "order_debug"
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema",
                           "document.schema.json")

BLOCK_TYPES = ("TITLE", "HEADING", "PARAGRAPH", "LIST", "TABLE", "FIGURE", "CHART", "EQUATION", "CAPTION",
               "HEADER", "FOOTER", "PAGE_NUMBER", "OTHER")
FLOAT_TYPES = ("TABLE", "FIGURE", "CHART")

FLOW_MAIN = "main"              # read in order
FLOW_ATTACHED = "attached"      # a caption or an equation number that belongs to another block
FLOW_MERGED = "merged"          # the later part of a paragraph that was joined to its beginning
FLOW_FURNITURE = "furniture"    # running header, footer, page number
FLOWS = (FLOW_MAIN, FLOW_ATTACHED, FLOW_MERGED, FLOW_FURNITURE)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_NOT_EXTRACTED = "not_extracted"      # the phase that extracts this kind of object was not run

_REGION_TO_BLOCK = {
    RegionType.TEXT.value: TYPE_PARAGRAPH, RegionType.LIST.value: "LIST", RegionType.TABLE.value: "TABLE",
    RegionType.FIGURE.value: "FIGURE", RegionType.EQUATION.value: "EQUATION", RegionType.CAPTION.value: "CAPTION",
    RegionType.HEADER.value: "HEADER", RegionType.FOOTER.value: "FOOTER",
    RegionType.PAGE_NUMBER.value: "PAGE_NUMBER", RegionType.OTHER.value: "OTHER",
}
_FURNITURE = ("HEADER", "FOOTER", "PAGE_NUMBER")
_TEXT_TYPES = (TYPE_TITLE, TYPE_HEADING, TYPE_PARAGRAPH, "CAPTION", "HEADER", "FOOTER", "PAGE_NUMBER", "OTHER")
_TERMINAL = ".!?:;"
_CLOSERS = "\"')]”’"

# Mentions of numbered objects in running text.
_MENTIONS = (
    ("figure", re.compile(r"\b(?:Figures?|Figs?\.?)\s*(\d{1,3})")),
    ("table", re.compile(r"\bTables?\s*(\d{1,3})")),
    ("equation", re.compile(r"\b(?:Equations?|Eqs?\.?|Eqn\.?)\s*\(?(\d{1,3})\)?")),
    ("section", re.compile(r"\b(?:Sections?|Secs?\.?)\s*(\d{1,2}(?:\.\d{1,2})*)")),
)
_LEADING_NUMBER = re.compile(r"\D*?(\d{1,3})")


class DocumentError(PdfPipelineError):
    """Base class for document reconstruction failures."""


class DocumentInputError(DocumentError):
    """The artifacts of the earlier phases are missing, invalid or do not belong together."""


class DocumentValidationError(DocumentError):
    """The reconstructed document does not pass its own checks."""


def load_artifacts(work_dir):
    """Read the artifacts of a working directory: {"pages", "layout", "ocr", "tables", "equations",
    "figures"}. Phases 2-4 are required. tables.json, equations.json and figures.json are used when
    present (None otherwise); one that is invalid or was made from another PDF is refused, not
    skipped. Raises DocumentInputError."""
    work_dir = os.fspath(work_dir)
    try:
        pages = load_pages_document(work_dir)
        layout = load_layout_document(work_dir, pages)
        ocr = load_ocr_document(work_dir, layout)
    except PdfPipelineError as exc:
        raise DocumentInputError(str(exc)) from exc
    if any("lines" not in block for page in pages["pages"] for block in page["blocks"]):
        raise DocumentInputError("pages.json has no native text lines (written before Phase 4); run Phase 2 again")
    sha = pages["document"].get("source_sha256")

    def optional(name, validate, key):
        path = os.path.join(work_dir, METADATA_DIR, name)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                document = json.load(fh)
        except (OSError, ValueError) as exc:
            raise DocumentInputError(f"Cannot read {path}: {exc}") from exc
        problems = validate(document)
        if problems:
            raise DocumentInputError(f"Invalid {path}: " + "; ".join(problems[:5]))
        region_ids = {r["region_id"] for page in layout["pages"] for r in page["regions"]}
        entries = [entry for page in document["pages"] for entry in page[key]]
        if document.get("source", {}).get("source_sha256") != sha or any(
                entry.get("source_region_id", entry.get("region_id")) not in region_ids for entry in entries):
            raise DocumentInputError(f"{path} was made from another layout.json; run its phase again")
        return document

    return {"pages": pages, "layout": layout, "ocr": ocr,
            "tables": optional(TABLES_JSON, validate_tables_document, "tables"),
            "equations": optional(EQUATIONS_JSON, validate_equations_document, "equations"),
            "figures": optional(FIGURES_JSON, validate_figures_document, "figures")}


def join_text(text, keep_lines=False):
    """The one text value of a block from the text Phase 4 stored for its region: lines joined with a
    space (kept as lines for a list). Words are not re-hyphenated or corrected."""
    if text is None:
        return None
    if keep_lines:
        return "\n".join(" ".join(line.split()) for line in text.split("\n") if line.strip())
    return " ".join(text.split())


def is_continuation(first, second):
    """True when `second` (a paragraph block) goes on where `first` stopped. All of this must hold:

      - both are PARAGRAPH blocks of the same section, neither a reference entry;
      - they are in different columns or on different pages (a break inside one column is a new paragraph);
      - the text of `first` does not end a sentence (no . ! ? : ; at its end, closing quotes aside);
      - the text of `second` starts with a lowercase letter.
    """
    if first["type"] != TYPE_PARAGRAPH or second["type"] != TYPE_PARAGRAPH:
        return False
    if first["section_id"] != second["section_id"] or not first["text"] or not second["text"]:
        return False
    if "reference" in (first["source_label"] or "") or "reference" in (second["source_label"] or ""):
        return False
    if (first["page_number"], first["column"]) == (second["page_number"], second["column"]):
        return False
    ending = first["text"].rstrip().rstrip(_CLOSERS).rstrip()
    return bool(ending) and ending[-1] not in _TERMINAL and second["text"].lstrip()[:1].islower()


def find_mentions(text):
    """Mentions of numbered objects in a text: [{"kind", "label", "number"}], in order of appearance.
    "Figure 3", "Fig. 3", "Table 2", "Eq. (4)", "Equation 4", "Section 3.1". Each (kind, number) once."""
    found = []
    for kind, pattern in _MENTIONS:
        for match in pattern.finditer(text or ""):
            entry = (match.start(), {"kind": kind, "label": " ".join(match.group(0).split()), "number": match.group(1)})
            if not any(e["kind"] == kind and e["number"] == match.group(1) for _, e in found):
                found.append(entry)
    return [entry for _, entry in sorted(found, key=lambda pair: pair[0])]


def _line_metrics(region, page, ocr_region):
    """(median line height, number of lines) of a text region, or (None, None)."""
    if ocr_region["text_source"] == TEXT_SOURCE_OCR:
        boxes = [line["pdf_bbox"] for line in ocr_region.get("lines", ())]
    elif ocr_region["text_source"] == TEXT_SOURCE_NATIVE:
        boxes = [bbox for bbox, _, _ in native_lines_in_region(region["pdf_bbox"], page)]
    else:
        boxes = []
    if not boxes:
        return None, None
    return statistics.median(box[3] - box[1] for box in boxes), len(group_rows([(box,) for box in boxes]))


def _number_of(text):
    match = _LEADING_NUMBER.match(text or "")
    return match.group(1) if match else None


def build_document(work_dir, visualize=False):
    """Reconstruct the document of a working directory and write metadata/document.json.

    Needs the Phase 2, 3 and 4 artifacts; tables.json, equations.json and figures.json are used when
    present. None of them is modified. See the module text for the steps. With visualize=True every
    page is also saved as <work_dir>/order_debug/page_NNN_order.png with its blocks numbered in
    reading order.

    Returns {"document_path", "work_dir", "document", "timings", "visualizations"}; `timings`
    ({"total_s"}) is not written to the file.
    Raises DocumentInputError for unusable input and DocumentValidationError when the result fails
    the schema, the integrity checks or the information-loss check; no document.json is left then.
    """
    started = time.perf_counter()
    work_dir = os.path.abspath(os.fspath(work_dir))
    document_path = os.path.join(work_dir, METADATA_DIR, DOCUMENT_JSON)
    _remove_stale(document_path)
    artifacts = load_artifacts(work_dir)
    pages_document, layout_document, ocr_document = artifacts["pages"], artifacts["layout"], artifacts["ocr"]

    def by_region(name, key):
        document = artifacts[name]
        if document is None:
            return None
        return {entry.get("source_region_id", entry.get("region_id")): entry
                for page in document["pages"] for entry in page[key]}

    tables_by_region = by_region("tables", "tables")
    equations_by_region = by_region("equations", "equations")
    figures_by_region = by_region("figures", "figures")

    blocks, pages, tables, figures, equations = [], [], [], [], []
    main_order = []                                         # main-flow blocks of the document, in reading order
    for page, layout_page, ocr_page in zip(pages_document["pages"], layout_document["pages"], ocr_document["pages"]):
        number = page["page_number"]
        page_blocks, by_region_id = [], {}
        for index, (region, ocr_region) in enumerate(zip(layout_page["regions"], ocr_page["regions"]), 1):
            block_type = _REGION_TO_BLOCK.get(region["type"], "OTHER")
            if region["type"] == RegionType.TITLE.value:
                block_type = TYPE_TITLE if region["source_label"] == "doc_title" else TYPE_HEADING
            height, lines = _line_metrics(region, page, ocr_region)
            has_text = ocr_region["text_source"] in (TEXT_SOURCE_NATIVE, TEXT_SOURCE_OCR)
            block = {
                "block_id": f"page_{number:03d}_block_{index:03d}",
                "page_number": number,
                "type": block_type,
                "bbox": list(region["pdf_bbox"]),
                "text": join_text(ocr_region["text"], block_type == "LIST") if has_text else None,
                "source": ocr_region["text_source"] if has_text else None,
                "confidence": ocr_region["confidence"],
                "section_id": None,
                "order": None,
                "flow": FLOW_FURNITURE if block_type in _FURNITURE else FLOW_MAIN,
                "column": None,
                "full_width": False,
                "source_region_id": region["region_id"],
                "source_label": region["source_label"],
                "references": [],
                "line_height": height,
                "line_count": lines,
            }
            page_blocks.append(block)
            by_region_id[region["region_id"]] = block

        # Tables, figures and equations: the block points to the extracted object.
        counters = {"table": 0, "figure": 0, "equation": 0}
        for block, region in zip(page_blocks, layout_page["regions"]):
            if block["type"] == "TABLE":
                counters["table"] += 1
                entry = _table_entry(block, counters["table"], tables_by_region, by_region_id)
                tables.append(entry)
                block.update(table_id=entry["table_id"], caption=entry["caption"], status=entry["status"])
            elif block["type"] == "FIGURE":
                counters["figure"] += 1
                entry = _figure_entry(block, counters["figure"], figures_by_region, by_region_id)
                figures.append(entry)
                if entry["figure_type"] == TYPE_CHART:
                    block["type"] = "CHART"
                block.update(figure_id=entry["figure_id"], figure_type=entry["figure_type"],
                             figure_number=entry["figure_number"], caption=entry["caption"], status=entry["status"])
            elif block["type"] == "EQUATION":
                counters["equation"] += 1
                entry = _equation_entry(block, counters["equation"], equations_by_region, page_blocks)
                equations.append(entry)
                block.update(equation_id=entry["equation_id"], latex=entry["latex"],
                             equation_number=entry["equation_number"], status=entry["status"])

        # Reading order of the page.
        flowing = [block for block in page_blocks if block["flow"] == FLOW_MAIN]
        layout = detect_columns([b["bbox"] for b in flowing if b["type"] != "OTHER"], page["pdf_width"])
        for block in flowing:
            block["full_width"] = spans_columns(block["bbox"], layout["gutters"])
            block["column"] = None if block["full_width"] else column_index(block["bbox"], layout["gutters"])
        ordered = [flowing[i] for i in order_blocks([b["bbox"] for b in flowing], layout)]
        main_order.extend(ordered)
        blocks.extend(page_blocks)
        pages.append({
            "page_number": number, "width": page["pdf_width"], "height": page["pdf_height"],
            "layout": layout["layout"], "columns": layout["columns"], "gutters": layout["gutters"],
            "block_ids": [],
            "header_block_ids": [b["block_id"] for b in page_blocks if b["type"] == "HEADER"],
            "footer_block_ids": [b["block_id"] for b in page_blocks if b["type"] == "FOOTER"],
            "page_number_block_ids": [b["block_id"] for b in page_blocks if b["type"] == "PAGE_NUMBER"],
        })

    sections, title = build_sections(main_order)
    by_id = {block["block_id"]: block for block in blocks}
    for block in blocks:                                    # an attached block lives in its owner's section
        if block["flow"] == FLOW_ATTACHED:
            block["section_id"] = by_id[block["attached_to"]]["section_id"]

    # Paragraphs cut by a column or page break.
    head = last = None                                      # the paragraph being read, and its latest part
    for block in main_order:
        if block["type"] in FLOAT_TYPES:                    # a figure or table may stand inside a paragraph
            continue
        if last is not None and is_continuation(last, block):
            if "parts" not in head:
                head["parts"] = [_part(head)]
            head["parts"].append(_part(block))
            last = dict(block)                              # the part as it was, for the next comparison
            head["text"] += ("" if head["text"].endswith("-") else " ") + block["text"]
            block.update(flow=FLOW_MERGED, merged_into=head["block_id"])
            for section in sections:
                if block["block_id"] in section["blocks"]:
                    section["blocks"].remove(block["block_id"])
            continue
        head = last = block
        last = dict(block)
    main_order = [block for block in main_order if block["flow"] == FLOW_MAIN]
    for order, block in enumerate(main_order):
        block["order"] = order
        pages[block["page_number"] - 1]["block_ids"].append(block["block_id"])

    _resolve_references(main_order, blocks, sections, tables, figures, equations)
    for block in blocks:
        del block["line_height"], block["line_count"]

    def furniture(kind):
        return [{"block_id": b["block_id"], "page_number": b["page_number"], "text": b["text"]}
                for b in blocks if b["type"] == kind]

    document = {
        "document": {
            "title": title,
            "page_count": len(pages),
            "source_pdf": pages_document["document"].get("source_pdf"),
            "source_sha256": pages_document["document"].get("source_sha256"),
            "schema_version": DOCUMENT_SCHEMA_VERSION,
        },
        "sections": sections,
        "blocks": blocks,
        "pages": pages,
        "tables": tables,
        "figures": figures,
        "equations": equations,
        "metadata": {"headers": furniture("HEADER"), "footers": furniture("FOOTER"),
                     "page_numbers": furniture("PAGE_NUMBER")},
        "provenance": {
            "artifacts": {
                "pages": f"{METADATA_DIR}/{PAGES_JSON}", "layout": f"{METADATA_DIR}/{LAYOUT_JSON}",
                "ocr": f"{METADATA_DIR}/{OCR_JSON}",
                "tables": f"{METADATA_DIR}/{TABLES_JSON}" if artifacts["tables"] else None,
                "equations": f"{METADATA_DIR}/{EQUATIONS_JSON}" if artifacts["equations"] else None,
                "figures": f"{METADATA_DIR}/{FIGURES_JSON}" if artifacts["figures"] else None,
            },
            "coordinate_system": {"bbox_format": "x0,y0,x1,y1", "origin": "top-left", "unit": "pt"},
        },
    }
    document["summary"] = summarize_document(document)
    document["provenance"]["counts"] = check_information_loss(document, artifacts)
    problems = (validate_document(document) + check_document_integrity(document)
                + document["provenance"]["counts"]["problems"])
    if problems:
        raise DocumentValidationError("document.json would be invalid: " + "; ".join(problems[:8]))

    visualizations = []
    if visualize:
        from .order_visualize import draw_reading_order, order_image_name
        debug_dir = ensure_directory(os.path.join(work_dir, ORDER_DEBUG_DIR))
        for page, entry in zip(pages_document["pages"], pages):
            target = os.path.join(debug_dir, order_image_name(page["page_number"]))
            draw_reading_order(os.path.join(work_dir, *page["image_path"].split("/")),
                               [by_id[block_id] for block_id in entry["block_ids"]], entry["gutters"],
                               page["scale_x"], page["scale_y"], target)
            visualizations.append(target)

    _write_json(document_path, document)
    return {"document_path": document_path, "work_dir": work_dir, "document": document,
            "timings": {"total_s": time.perf_counter() - started}, "visualizations": visualizations}


def _part(block):
    return {"block_id": block["block_id"], "page_number": block["page_number"], "bbox": list(block["bbox"]),
            "text": block["text"], "source": block["source"]}


def _attach(block, owner_id):
    block.update(flow=FLOW_ATTACHED, attached_to=owner_id)
    return block["block_id"]


def _caption_block(entry, block, by_region_id):
    """Attach the caption region an extraction phase paired with an object. Returns its block id."""
    caption = by_region_id.get((entry or {}).get("caption_region_id"))
    if caption is None or caption["flow"] != FLOW_MAIN:
        return None
    return _attach(caption, block["block_id"])


def _object_base(block, kind, position, entry, id_key):
    return {
        f"{kind}_id": f"page_{block['page_number']:03d}_{kind}_{position:03d}",
        "source_id": entry[id_key] if entry else None,
        "block_id": block["block_id"],
        "page_number": block["page_number"],
        "bbox": list(block["bbox"]),
    }


def _table_entry(block, position, tables_by_region, by_region_id):
    source = (tables_by_region or {}).get(block["source_region_id"])
    entry = _object_base(block, "table", position, source, "table_id")
    failed = source is not None and "error" in source
    entry.update(
        status=STATUS_NOT_EXTRACTED if source is None else STATUS_FAILED if failed else STATUS_OK,
        caption=source["caption"] if source else None,
        caption_block_id=_caption_block(source, block, by_region_id),
        n_rows=source["n_rows"] if source else 0,
        n_columns=source["n_columns"] if source else 0,
        header_rows=source["header_rows"] if source else None,
        headers=source["headers"] if source else None,
        rows=source["rows"] if source else [],
        extraction_source=source["source"] if source else None,
        error=source.get("error") if source else None,
    )
    return entry


def _figure_entry(block, position, figures_by_region, by_region_id):
    source = (figures_by_region or {}).get(block["source_region_id"])
    entry = _object_base(block, "figure", position, source, "figure_id")
    entry.update(
        status=STATUS_NOT_EXTRACTED if source is None else STATUS_FAILED if source["status"] != "ok" else STATUS_OK,
        figure_type=source["figure_type"] if source else "unknown",
        figure_number=source["figure_number"] if source else None,
        caption=source["caption"] if source else None,
        caption_block_id=_caption_block(source, block, by_region_id),
        image_path=source["image_path"] if source else None,
        labels=[item["text"] for item in source["ocr_text"]] if source else [],
        chart_data=source["chart_data"] if source else None,
        error=source.get("error") if source else None,
    )
    return entry


def _equation_entry(block, position, equations_by_region, page_blocks):
    source = (equations_by_region or {}).get(block["source_region_id"])
    entry = _object_base(block, "equation", position, source, "equation_id")
    number_block_id = None
    number_box = source and source.get("equation_number_pdf_bbox")
    if number_box:      # the "(4)" beside the equation is also a region of its own: it belongs to the equation
        for other in page_blocks:
            if other["type"] == "OTHER" and other["flow"] == FLOW_MAIN and bbox_iou(other["bbox"], number_box) > 0.2:
                number_block_id = _attach(other, block["block_id"])
                break
    entry.update(
        status=source["status"] if source else STATUS_NOT_EXTRACTED,
        latex=source["latex"] if source else None,
        normalized_latex=source["normalized_latex"] if source else None,
        equation_number=source["equation_number"] if source else None,
        equation_number_block_id=number_block_id,
        image_path=source["image_path"] if source else None,
        error=source.get("error") if source else None,
    )
    return entry


def _resolve_references(main_order, blocks, sections, tables, figures, equations):
    """Fill "references" of the text blocks: each mention with the block it points to, when exactly
    one object carries that number."""
    targets = {"figure": {}, "table": {}, "equation": {}, "section": {}}
    for figure in figures:
        targets["figure"].setdefault(_number_of(figure["figure_number"]), []).append(figure["block_id"])
    for table in tables:
        caption = table["caption"] or ""
        if re.match(r"\s*(?:Table|TABLE|Tab\.)", caption):
            targets["table"].setdefault(_number_of(caption), []).append(table["block_id"])
    for equation in equations:
        targets["equation"].setdefault(_number_of(equation["equation_number"]), []).append(equation["block_id"])
    for section in sections:
        if section["number"] and section["heading_block_id"]:
            targets["section"].setdefault(section["number"], []).append(section["heading_block_id"])
    for block in main_order:
        if block["type"] not in (TYPE_PARAGRAPH, "LIST", "OTHER") or not block["text"]:
            continue
        for mention in find_mentions(block["text"]):
            found = targets[mention["kind"]].get(mention["number"], [])
            block["references"].append({"kind": mention["kind"], "label": mention["label"],
                                        "target_block_id": found[0] if len(found) == 1 else None})


def summarize_document(document):
    """Counts of a document.json object."""
    blocks = [block for block in document["blocks"] if block["flow"] == FLOW_MAIN]
    layouts = [page["layout"] for page in document["pages"]]

    def count(kind):
        return sum(1 for block in blocks if block["type"] == kind)

    return {
        "pages": len(document["pages"]),
        "sections": sum(1 for s in document["sections"] if s["level"] == 1 and s["section_type"] != "front_matter"),
        "subsections": sum(1 for s in document["sections"] if s["level"] > 1),
        "blocks": len(document["blocks"]),
        "blocks_in_reading_order": len(blocks),
        "headings": count(TYPE_HEADING),
        "paragraphs": count(TYPE_PARAGRAPH),
        "lists": count("LIST"),
        "tables": count("TABLE"),
        "figures": count("FIGURE"),
        "charts": count("CHART"),
        "equations": count("EQUATION"),
        "captions_attached": sum(1 for b in document["blocks"] if b["flow"] == FLOW_ATTACHED and b["type"] == "CAPTION"),
        "captions_standalone": count("CAPTION"),
        "equation_numbers_attached": sum(1 for b in document["blocks"]
                                         if b["flow"] == FLOW_ATTACHED and b["type"] == "OTHER"),
        "paragraphs_merged": sum(1 for b in document["blocks"] if b["flow"] == FLOW_MERGED),
        "references_found": sum(len(b["references"]) for b in document["blocks"]),
        "references_resolved": sum(1 for b in document["blocks"] for r in b["references"] if r["target_block_id"]),
        "headers": len(document["metadata"]["headers"]),
        "footers": len(document["metadata"]["footers"]),
        "page_numbers": len(document["metadata"]["page_numbers"]),
        "layout": {layout: layouts.count(layout) for layout in LAYOUTS if layout in layouts},
    }


def validate_document(document):
    """Check a document.json object against AIEngine/schema/document.schema.json. Returns a list of
    problems, each with the place in the document it was found (empty = valid)."""
    try:
        import jsonschema
    except ImportError as exc:
        raise DocumentError("the jsonschema package is needed to validate document.json") from exc
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        schema = json.load(fh)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(map(str, error.absolute_path)))
    return ["/".join(map(str, error.absolute_path)) + ": " + error.message for error in errors]


def check_document_integrity(document):
    """What a schema cannot say about a document.json object. Returns a list of problems (empty = sound).

    Checked: block ids, object ids and section ids are unique; every block lies on an existing page
    and inside it; the main-flow blocks carry the orders 0..n-1 exactly once, and no other block has
    an order; orders rise with the page number; every section_id, parent_id, table_id, figure_id,
    equation_id, attached_to, merged_into and reference target exists; every main-flow block is
    listed in exactly one section and on its page; an attached caption is in no section list and
    belongs to exactly one object; every table, figure and equation has its block.
    """
    problems = []
    blocks, sections = document["blocks"], document["sections"]
    by_id = {block["block_id"]: block for block in blocks}
    section_ids = [section["section_id"] for section in sections]
    for name, ids in (("block_id", [b["block_id"] for b in blocks]), ("section_id", section_ids),
                      ("table_id", [t["table_id"] for t in document["tables"]]),
                      ("figure_id", [f["figure_id"] for f in document["figures"]]),
                      ("equation_id", [e["equation_id"] for e in document["equations"]])):
        if len(ids) != len(set(ids)):
            problems.append(f"{name} values are not unique")
    pages = {page["page_number"]: page for page in document["pages"]}
    if sorted(pages) != list(range(1, len(document["pages"]) + 1)):
        problems.append("page numbers are not 1..n")

    main = [block for block in blocks if block["flow"] == FLOW_MAIN]
    if sorted(block["order"] for block in main if block["order"] is not None) != list(range(len(main))):
        problems.append("the main-flow blocks do not carry the orders 0..n-1 exactly once")
    ordered = sorted(main, key=lambda block: block["order"] or 0)
    if any(a["page_number"] > b["page_number"] for a, b in zip(ordered, ordered[1:])):
        problems.append("the reading order goes back to an earlier page")
    listed = [block_id for section in sections for block_id in section["blocks"]]
    if sorted(listed) != sorted(block["block_id"] for block in main):
        problems.append("the section block lists are not exactly the main-flow blocks")
    on_pages = [block_id for page in document["pages"] for block_id in page["block_ids"]]
    if on_pages != [block["block_id"] for block in ordered]:
        problems.append("the page block lists are not the main-flow blocks in reading order")

    objects = {"table_id": {t["table_id"]: t for t in document["tables"]},
               "figure_id": {f["figure_id"]: f for f in document["figures"]},
               "equation_id": {e["equation_id"]: e for e in document["equations"]}}
    for block in blocks:
        where = block["block_id"]
        page = pages.get(block["page_number"])
        if page is None:
            problems.append(f"{where}: page {block['page_number']} does not exist")
        elif not (is_valid_bbox(block["bbox"]) and block["bbox"][0] >= -0.01 and block["bbox"][1] >= -0.01
                  and block["bbox"][2] <= page["width"] + 0.01 and block["bbox"][3] <= page["height"] + 0.01):
            problems.append(f"{where}: bbox is not a valid box on its page")
        if block["flow"] != FLOW_MAIN and block["order"] is not None:
            problems.append(f"{where}: a block outside the reading order has an order")
        if block["flow"] != FLOW_FURNITURE and block["section_id"] not in section_ids:
            problems.append(f"{where}: section {block['section_id']!r} does not exist")
        for key in ("attached_to", "merged_into"):
            if block.get(key) is not None and block[key] not in by_id:
                problems.append(f"{where}: {key} {block[key]!r} does not exist")
        if (block["flow"] == FLOW_ATTACHED) != (block.get("attached_to") is not None) or (
                block["flow"] == FLOW_MERGED) != (block.get("merged_into") is not None):
            problems.append(f"{where}: flow and attached_to / merged_into do not fit together")
        for key, known in objects.items():
            if key in block and (block[key] not in known or known[block[key]]["block_id"] != where):
                problems.append(f"{where}: {key} {block[key]!r} does not exist or names another block")
        for reference in block["references"]:
            if reference["target_block_id"] is not None and reference["target_block_id"] not in by_id:
                problems.append(f"{where}: reference target {reference['target_block_id']!r} does not exist")
    for section in sections:
        if section["parent_id"] is not None and section["parent_id"] not in section_ids:
            problems.append(f"{section['section_id']}: parent {section['parent_id']!r} does not exist")
        if section["heading_block_id"] is not None and section["heading_block_id"] not in by_id:
            problems.append(f"{section['section_id']}: heading block does not exist")
    owners = {}
    for kind, entries in (("table", document["tables"]), ("figure", document["figures"])):
        for entry in entries:
            caption_id = entry["caption_block_id"]
            if caption_id is None:
                continue
            caption = by_id.get(caption_id)
            if caption is None or caption["flow"] != FLOW_ATTACHED or caption.get("attached_to") != entry["block_id"]:
                problems.append(f"{entry[kind + '_id']}: its caption block is not attached to it")
            if caption_id in owners:
                problems.append(f"{caption_id}: caption attached to two objects")
            owners[caption_id] = entry["block_id"]
    for name, entries in (("table", document["tables"]), ("figure", document["figures"]),
                          ("equation", document["equations"])):
        for entry in entries:
            if by_id.get(entry["block_id"], {}).get(name + "_id") != entry[name + "_id"]:
                problems.append(f"{entry[name + '_id']}: its block does not point back to it")
    return problems


def check_information_loss(document, artifacts):
    """Compare a document.json object with the artifacts it was built from. Returns the counts on
    both sides and "problems" (empty when nothing was lost):

      - every layout region is exactly one block;
      - every text Phase 4 stored (native or OCR) is the text of its block, word for word;
      - every table, equation and figure of Phases 5-7 is an object of the document, including those
        whose extraction or recognition failed (with their image where the phase kept one);
      - every caption an extraction phase paired with an object is attached to it.
    """
    problems = []
    blocks = {block["source_region_id"]: block for block in document["blocks"]}
    regions = [region for page in artifacts["layout"]["pages"] for region in page["regions"]]
    if len(blocks) != len(document["blocks"]) or set(blocks) != {region["region_id"] for region in regions}:
        problems.append("the blocks are not exactly the layout regions")

    texts = {"native": 0, "ocr": 0}
    kept = {"native": 0, "ocr": 0}
    for page in artifacts["ocr"]["pages"]:
        for region in page["regions"]:
            if region["text_source"] not in texts:
                continue
            texts[region["text_source"]] += 1
            block = blocks.get(region["region_id"])
            own = block and (block["parts"][0]["text"] if "parts" in block else block["text"])
            if block and block["source"] == region["text_source"] and (own or "").split() == region["text"].split():
                kept[region["text_source"]] += 1
            else:
                problems.append(f"text of region {region['region_id']} is not the text of its block")

    counts = {"layout_regions": len(regions), "blocks": len(document["blocks"]),
              "native_text_regions": texts["native"], "native_text_blocks": kept["native"],
              "ocr_text_regions": texts["ocr"], "ocr_text_blocks": kept["ocr"]}
    for name, key, id_key in (("tables", "tables", "table_id"), ("equations", "equations", "equation_id"),
                              ("figures", "figures", "figure_id")):
        source = artifacts[name]
        entries = [entry for page in source["pages"] for entry in page[key]] if source else []
        present = {entry["source_id"] for entry in document[name]}
        counts[f"{name}_source"] = len(entries) if source else None
        counts[name] = len(document[name])
        missing = [entry[id_key] for entry in entries if entry[id_key] not in present]
        if missing:
            problems.append(f"{name} missing from the document: {', '.join(missing)}")
        paired = sum(1 for entry in entries if entry.get("caption_region_id"))
        attached = sum(1 for entry in document[name] if entry.get("caption_block_id"))
        if name != "equations":
            counts[f"{name}_captions_source"] = paired
            counts[f"{name}_captions_attached"] = attached
            if attached != paired:
                problems.append(f"{paired} {name} captions were paired but {attached} are attached")
    failed = [e for page in (artifacts["equations"] or {"pages": []})["pages"] for e in page["equations"]
              if e["status"] != "success"]
    kept_failed = [e for e in document["equations"] if e["status"] not in ("success", STATUS_NOT_EXTRACTED)]
    counts["equations_not_recognized_source"] = len(failed)
    counts["equations_not_recognized"] = len(kept_failed)
    if len(failed) != len(kept_failed) or any(e["image_path"] is None for e in kept_failed if any(
            f["equation_id"] == e["source_id"] and f["image_path"] for f in failed)):
        problems.append("an equation that was not recognized lost its entry or its image")
    counts["problems"] = problems
    return counts


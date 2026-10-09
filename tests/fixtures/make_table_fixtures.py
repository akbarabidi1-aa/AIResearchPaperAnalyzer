"""
Generates the SYNTHETIC fixtures of Phase 5 (tables). Needs PyMuPDF. Run from anywhere:

    python tests/fixtures/make_table_fixtures.py

Writes:
    tables/synthetic_tables.pdf           3 pages of native text, one table each:
                                            1  bordered table, caption above
                                            2  borderless numeric results table (7 rows), caption below
                                            3  bordered table with a merged header cell, caption above
    tables/synthetic_tables_scanned.pdf   the same pages as grayscale bitmaps (150 DPI), no text layer
    tables/synthetic_tables.json          the known content of every table (ground truth)
"""
import json
import os

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")

# A cell is a string, or (text, col_span) for a merged cell. "grid" in the ground truth repeats nothing:
# the positions covered by a merged cell hold "".
TABLES = [
    {
        "page_number": 1,
        "kind": "bordered",
        "heading": "1. Datasets",
        "paragraph": ("The experiments use three synthetic collections of labelled images. The table below lists "
                      "the number of classes and the size of the training and test split of each collection."),
        "caption": "Table 1. Statistics of the three synthetic datasets.",
        "caption_above": True,
        "bordered": True,
        "column_widths": [130, 80, 90, 90],
        "header_rows": [0],
        "cells": [
            ["Dataset", "Classes", "Train", "Test"],
            ["Alpha", "10", "50,000", "10,000"],
            ["Beta", "100", "73,257", "26,032"],
            ["Gamma", "200", "100,000", "5,000"],
        ],
    },
    {
        "page_number": 2,
        "kind": "borderless",
        "heading": "2. Results",
        "paragraph": ("Every method was trained three times. The table reports the mean accuracy, the mean F1 "
                      "score and the training time in minutes, without any ruling lines between the cells."),
        "caption": "Table 2. Comparison with baselines on the Alpha dataset.",
        "caption_above": False,
        "bordered": False,
        "column_widths": [130, 90, 80, 90],
        "header_rows": [0],
        "cells": [
            ["Method", "Accuracy", "F1", "Time"],
            ["Linear", "71.25", "70.90", "3.5"],
            ["MLP", "80.00", "79.42", "6.0"],
            ["ConvNet", "88.13", "87.60", "14.2"],
            ["ResNet", "92.10", "91.40", "31.0"],
            ["ViT", "94.70", "94.10", "58.4"],
            ["Ours", "95.32", "95.01", "40.7"],
        ],
    },
    {
        "page_number": 3,
        "kind": "merged_header",
        "heading": "3. Ablation",
        "paragraph": ("The last table groups two accuracy columns under one merged header cell. The size of "
                      "every model is given in millions of parameters in the final column."),
        "caption": "Table 3. Accuracy and size of the ablated models.",
        "caption_above": True,
        "bordered": True,
        "column_widths": [130, 80, 80, 100],
        "header_rows": [0, 1],
        "cells": [
            ["Model", ("Accuracy", 2), "Params"],
            ["", "Top-1", "Top-5", ""],
            ["Small", "81.20", "95.10", "5.3"],
            ["Base", "86.50", "97.42", "21.8"],
            ["Large", "88.10", "98.00", "86.0"],
        ],
    },
]

LEFT, TABLE_TOP, ROW_HEIGHT, FONT_SIZE = 90.0, 215.0, 22.0, 11


def draw_table(page, spec):
    widths = spec["column_widths"]
    edges = [LEFT]
    for width in widths:
        edges.append(edges[-1] + width)
    right = edges[-1]

    for r, row in enumerate(spec["cells"]):
        top = TABLE_TOP + r * ROW_HEIGHT
        column = 0
        for cell in row:
            text, span = cell if isinstance(cell, tuple) else (cell, 1)
            x0, x1 = edges[column], edges[column + span]
            if text:
                font = "hebo" if r in spec["header_rows"] else "helv"
                if span > 1:    # a merged cell is centred over its columns
                    x = (x0 + x1 - pymupdf.get_text_length(text, fontname=font, fontsize=FONT_SIZE)) / 2
                else:
                    x = x0 + 8
                page.insert_text((x, top + 15), text, fontsize=FONT_SIZE, fontname=font)
            if spec["bordered"]:
                page.draw_rect(pymupdf.Rect(x0, top, x1, top + ROW_HEIGHT), color=(0, 0, 0), width=0.7)
            column += span
    bottom = TABLE_TOP + len(spec["cells"]) * ROW_HEIGHT
    return [LEFT, TABLE_TOP, right, bottom]


def build_page(doc, spec):
    page = doc.new_page()
    page.insert_text((72, 90), spec["heading"], fontsize=14, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 105, 540, 160), spec["paragraph"], fontsize=11)
    rows = len(spec["cells"])
    caption_y = TABLE_TOP - 12 if spec["caption_above"] else TABLE_TOP + rows * ROW_HEIGHT + 20
    page.insert_text((LEFT, caption_y), spec["caption"], fontsize=10)
    bbox = draw_table(page, spec)
    page.insert_textbox(pymupdf.Rect(72, bbox[3] + 45, 540, bbox[3] + 100),
                        "The text continues below the table with an ordinary paragraph, so that the table is "
                        "surrounded by body text on both sides as it would be in a paper.", fontsize=11)
    return bbox


def ground_truth(spec, bbox):
    grid, spans = [], []
    for r, row in enumerate(spec["cells"]):
        out = []
        for cell in row:
            text, span = cell if isinstance(cell, tuple) else (cell, 1)
            if span > 1:
                spans.append({"row": r, "column": len(out), "row_span": 1, "col_span": span})
            out.extend([text] + [""] * (span - 1))
        grid.append(out)
    return {"page_number": spec["page_number"], "kind": spec["kind"], "bordered": spec["bordered"],
            "caption": spec["caption"], "pdf_bbox": bbox, "header_rows": spec["header_rows"],
            "n_rows": len(grid), "n_columns": len(grid[0]), "grid": grid, "merged_cells": spans}


def save(doc, name):
    os.makedirs(HERE, exist_ok=True)
    target = os.path.join(HERE, name)
    doc.save(target, garbage=4, deflate=True, no_new_id=True)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")
    return target


def main():
    with pymupdf.open() as doc:
        truth = [ground_truth(spec, build_page(doc, spec)) for spec in TABLES]
        digital = save(doc, "synthetic_tables.pdf")

    with pymupdf.open(digital) as source, pymupdf.open() as out:
        for page in source:
            pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
            out.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, pixmap=pix)
        save(out, "synthetic_tables_scanned.pdf")

    with open(os.path.join(HERE, "synthetic_tables.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tables": truth}, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()

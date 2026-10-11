"""
Generates the SYNTHETIC fixtures of Phase 6 (equations). Needs PyMuPDF and matplotlib (its mathtext
renderer draws the formulas; no TeX installation is used). Run from anywhere:

    python tests/fixtures/make_equation_fixtures.py

Writes:
    equations/synthetic_equations.pdf           4 pages of native text with three display equations each
                                                (vector drawings in three math fonts), six of them numbered
    equations/synthetic_equations_scanned.pdf   the same pages as grayscale bitmaps (150 DPI), no text layer
    equations/synthetic_equations.json          the known LaTeX, number and position of every equation
"""
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equations")

# "latex" is the ground truth. "draw" is what mathtext is given when it differs (mathtext sets no space
# after an operator, and has no matrix or alignment environment: those are composed from "rows").
# "font" is the mathtext font set: cm (Computer Modern), stix (Times-like), dejavusans (sans-serif).
EQUATIONS = [
    {"page_number": 1, "kind": "simple_algebra", "font": "cm", "number": None,
     "latex": r"y = mx + b"},
    {"page_number": 1, "kind": "summation", "font": "cm", "number": "(1)",
     "latex": r"L = -\sum_i y_i \log(p_i)", "draw": r"L = -\sum_i \, y_i \, \log(p_i)"},
    {"page_number": 1, "kind": "fraction", "font": "stix", "number": "(2)",
     "latex": r"z = \frac{x+y}{2}"},
    {"page_number": 2, "kind": "square_root", "font": "cm", "number": None,
     "latex": r"\sqrt{x^2 + y^2}"},
    {"page_number": 2, "kind": "greek", "font": "stix", "number": None,
     "latex": r"\alpha + \beta = \lambda"},
    {"page_number": 2, "kind": "subscript_superscript", "font": "dejavusans", "number": None,
     "latex": r"x_i^2"},
    {"page_number": 3, "kind": "integral", "font": "stix", "number": "(3)",
     "latex": r"\int_a^b f(x)\,dx"},
    {"page_number": 3, "kind": "numbered", "font": "cm", "number": "(4)",
     "latex": r"E = mc^2"},
    {"page_number": 3, "kind": "matrix", "font": "cm", "number": "(5)",
     "latex": r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}",
     "matrix": {"left": r"A =", "rows": [["a", "b"], ["c", "d"]]}},
    {"page_number": 4, "kind": "multiline_aligned", "font": "stix", "number": "(6)",
     "latex": r"\begin{aligned} f(x) &= (x+1)^2 \\ &= x^2 + 2x + 1 \end{aligned}",
     "aligned": [[r"f(x)", r"= (x+1)^2"], ["", r"= x^2 + 2x + 1"]]},
    {"page_number": 4, "kind": "symbol_fidelity_digits", "font": "cm", "number": None,
     "latex": r"x_0 \times l_1 = 10"},
    {"page_number": 4, "kind": "symbol_fidelity_greek", "font": "stix", "number": None,
     "latex": r"\lambda A = a - \alpha"},
]

HEADINGS = {1: "1. Model", 2: "2. Notation", 3: "3. Derivation", 4: "4. Further identities"}
PARAGRAPHS = [
    "The quantities used in this section are defined by the following display equation, which is set "
    "on a line of its own and centred between the margins of the page.",
    "Substituting the definition above into the objective gives the next expression. It is referred to "
    "in the text by its number, for example as Eq. (1), where it has one.",
    "A further relation follows directly. The remaining text of the page is an ordinary paragraph that "
    "comes after the last displayed equation.",
]
CLOSING = ("This paragraph closes the page. It contains running text only, so that every equation is "
           "surrounded by body text on both sides as it would be in a paper.")

LEFT, RIGHT, FONT_SIZE, MATH_SIZE = 72.0, 540.0, 11, 13
FIGURE_WIDTH, FIGURE_HEIGHT = 288.0, 144.0      # points; the drawing is cropped to its ink afterwards


def _figure():
    return plt.figure(figsize=(FIGURE_WIDTH / 72, FIGURE_HEIGHT / 72))


def _math(fig, x, y, source, font, **kwargs):
    """One mathtext string at (x, y), in points from the lower left corner of the figure."""
    return fig.text(x / FIGURE_WIDTH, y / FIGURE_HEIGHT, f"${source}$", fontsize=MATH_SIZE, math_fontfamily=font,
                    **kwargs)


def draw_equation(spec):
    """The equation as a one-page PDF, cropped to its ink (2 pt margin)."""
    fig, font = _figure(), spec["font"]
    if "matrix" in spec:
        rows, pitch_x, pitch_y, x0 = spec["matrix"]["rows"], 26.0, 20.0, 40.0
        height = pitch_y * len(rows)
        _math(fig, x0 - 8, 50, spec["matrix"]["left"], font, ha="right", va="center")
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                _math(fig, x0 + 12 + c * pitch_x, 50 + height / 2 - (r + 0.5) * pitch_y, cell, font,
                      ha="center", va="center")
        x1 = x0 + 24 + (len(rows[0]) - 1) * pitch_x
        for x, tick in ((x0, 4), (x1, -4)):         # square brackets
            top, bottom = (50 + height / 2) / FIGURE_HEIGHT, (50 - height / 2) / FIGURE_HEIGHT
            fig.add_artist(Line2D([(x + tick) / FIGURE_WIDTH, x / FIGURE_WIDTH, x / FIGURE_WIDTH,
                                   (x + tick) / FIGURE_WIDTH], [top, top, bottom, bottom],
                                  transform=fig.transFigure, color="black", linewidth=0.9))
    elif "aligned" in spec:
        for r, (left, right) in enumerate(spec["aligned"]):
            y = 70 - r * 22
            if left:
                _math(fig, 60, y, left, font, ha="right", va="baseline")
            _math(fig, 64, y, right, font, ha="left", va="baseline")
    else:
        _math(fig, 10, 50, spec.get("draw", spec["latex"]), font, va="baseline")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight", pad_inches=2 / 72,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return buffer.getvalue()


def build_page(doc, page_number, specs):
    """One page: heading, then paragraph / equation three times, then a closing paragraph.
    Returns the ground truth of its equations."""
    page = doc.new_page()
    page.insert_text((LEFT, 90), HEADINGS[page_number], fontsize=14, fontname="hebo")
    y, truth = 108.0, []
    for paragraph, spec in zip(PARAGRAPHS, specs):
        page.insert_textbox(pymupdf.Rect(LEFT, y, RIGHT, y + 46), paragraph, fontsize=FONT_SIZE)
        y += 58
        with pymupdf.open(stream=draw_equation(spec), filetype="pdf") as drawing:
            width, height = drawing[0].rect.width, drawing[0].rect.height
            x0 = (LEFT + RIGHT - width) / 2
            box = pymupdf.Rect(x0, y, x0 + width, y + height)
            page.show_pdf_page(box, drawing, 0)
        entry = {"page_number": page_number, "kind": spec["kind"], "font": spec["font"], "latex": spec["latex"],
                 "equation_number": spec["number"], "pdf_bbox": [round(v, 2) for v in box],
                 "number_pdf_bbox": None}
        if spec["number"]:
            number_width = pymupdf.get_text_length(spec["number"], fontname="helv", fontsize=FONT_SIZE)
            baseline = (box.y0 + box.y1) / 2 + 4
            page.insert_text((RIGHT - number_width, baseline), spec["number"], fontsize=FONT_SIZE)
            entry["number_pdf_bbox"] = [round(RIGHT - number_width, 2), round(baseline - 10, 2), RIGHT,
                                        round(baseline + 3, 2)]
        truth.append(entry)
        y += height + 22
    page.insert_textbox(pymupdf.Rect(LEFT, y, RIGHT, y + 46), CLOSING, fontsize=FONT_SIZE)
    return truth


def save(doc, name):
    os.makedirs(HERE, exist_ok=True)
    target = os.path.join(HERE, name)
    doc.save(target, garbage=4, deflate=True, no_new_id=True)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")
    return target


def main():
    with pymupdf.open() as doc:
        truth = []
        for page_number in sorted(HEADINGS):
            truth += build_page(doc, page_number, [s for s in EQUATIONS if s["page_number"] == page_number])
        digital = save(doc, "synthetic_equations.pdf")

    with pymupdf.open(digital) as source, pymupdf.open() as out:
        for page in source:
            pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
            out.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, pixmap=pix)
        save(out, "synthetic_equations_scanned.pdf")

    with open(os.path.join(HERE, "synthetic_equations.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"equations": truth}, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()

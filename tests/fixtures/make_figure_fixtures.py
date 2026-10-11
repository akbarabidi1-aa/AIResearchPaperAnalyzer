"""
Generates the SYNTHETIC fixtures of Phase 7 (figures and charts). Needs PyMuPDF, matplotlib, numpy and
Pillow. Run from anywhere:

    python tests/fixtures/make_figure_fixtures.py

Writes:
    figures/synthetic_figures.pdf           5 pages of native text with nine figures: three diagrams and
                                            five charts as vector drawings (their labels are native text)
                                            and one bitmap image
    figures/synthetic_figures_scanned.pdf   the same pages as colour bitmaps (150 DPI), no text layer
    figures/synthetic_figures.json          type, position, caption, number, labels and chart content of
                                            every figure

Page 1  architecture diagram, bitmap image          captions below
Page 2  bar chart with value labels (caption below), line chart without legend (caption ABOVE)
Page 3  scatter plot, two-series line chart with legend
Page 4  two columns: bar chart with its caption below (left), diagram with its caption above (right);
        the right caption is nearer to the left figure than that figure's own caption
Page 5  a diagram with a caption-like line at the same distance above and below it (ambiguous)
"""
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

plt.rcParams.update({
    "pdf.fonttype": 42,             # TrueType: the labels stay extractable text
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
})

LEFT, RIGHT, FONT_SIZE, CAPTION_SIZE = 72.0, 540.0, 10, 9
COLUMNS = ((72.0, 294.0), (318.0, 540.0))       # page 4
BODY = ("This paragraph is ordinary running text. It stands before and after the figures of the page so "
        "that every figure is surrounded by body text, as it would be in a paper.")
SHORT_BODY = ("Running text of a narrow column. It fills the lines above and below the figure of this "
              "column and has no other purpose.")

BAR_LABELLED = {
    "chart_type": "bar", "title": "Top-1 accuracy", "x_title": "Model", "y_title": "Accuracy (%)",
    "categories": ["ResNet", "ViT", "ConvNeXt", "Swin"], "values": [72.1, 84.7, 63.4, 91.2],
    "y_ticks": [0, 20, 40, 60, 80, 100], "value_labels": True,
}
LINE_SINGLE = {
    "chart_type": "line", "title": None, "x_title": "Epoch", "y_title": "Loss",
    "x": [1, 2, 3, 4, 5, 6], "series": [{"name": None, "y": [2.1, 1.3, 1.0, 0.6, 0.5, 0.3]}],
    "x_ticks": [1, 2, 3, 4, 5, 6], "y_ticks": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
}
SCATTER = {
    "chart_type": "scatter", "title": None, "x_title": "Parameters (M)", "y_title": "Accuracy (%)",
    "points": [[5, 62], [12, 68], [18, 66], [25, 74], [31, 71], [38, 79], [44, 77], [52, 83], [60, 81],
               [67, 88], [75, 86], [84, 92]],
    "x_ticks": [0, 20, 40, 60, 80, 100], "y_ticks": [50, 60, 70, 80, 90, 100],
}
LINE_MULTI = {
    "chart_type": "line", "title": "Validation accuracy", "x_title": "Epoch", "y_title": "Accuracy",
    "x": [1, 2, 3, 4, 5],
    "series": [{"name": "Method A", "y": [0.52, 0.66, 0.71, 0.80, 0.82]},
               {"name": "Method B", "y": [0.40, 0.47, 0.60, 0.64, 0.75]}],
    "x_ticks": [1, 2, 3, 4, 5], "y_ticks": [0.2, 0.4, 0.6, 0.8, 1.0],
}
BAR_PLAIN = {
    "chart_type": "bar", "title": None, "x_title": "Dataset", "y_title": "Error",
    "categories": ["Set A", "Set B", "Set C"], "values": [12.0, 7.5, 4.0],
    "y_ticks": [0, 5, 10, 15], "value_labels": False,
}


def _save(fig):
    """The figure as a one-page PDF, cropped to its ink (2 pt margin)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight", pad_inches=2 / 72,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return buffer.getvalue()


def _axes(width, height, spec, box=True):
    fig, ax = plt.subplots(figsize=(width / 72, height / 72))
    if spec["title"]:
        ax.set_title(spec["title"], fontsize=9)
    ax.set_xlabel(spec["x_title"])
    ax.set_ylabel(spec["y_title"])
    if not box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig, ax


def draw_bar(spec, width=300, height=190, color="#4477aa", box=True):
    fig, ax = _axes(width, height, spec, box)
    bars = ax.bar(spec["categories"], spec["values"], width=0.6, color=color)
    ax.set_yticks(spec["y_ticks"])
    ax.set_ylim(spec["y_ticks"][0], spec["y_ticks"][-1])
    if spec["value_labels"]:
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in spec["values"]], padding=2, fontsize=8)
    return _save(fig)


def draw_line(spec, width=300, height=190, box=True):
    fig, ax = _axes(width, height, spec, box)
    styles = [{"color": "#cc3311"}, {"color": "#228833", "marker": "o", "markersize": 3.5}]
    for series, style in zip(spec["series"], styles):
        ax.plot(spec["x"], series["y"], linewidth=1.4, label=series["name"], **style)
    ax.set_xticks(spec["x_ticks"])
    ax.set_yticks(spec["y_ticks"])
    ax.set_ylim(spec["y_ticks"][0], spec["y_ticks"][-1])
    if any(series["name"] for series in spec["series"]):
        ax.legend(loc="lower right", fontsize=8)
    return _save(fig)


def draw_scatter(spec, width=300, height=190):
    fig, ax = _axes(width, height, spec)
    xs, ys = zip(*spec["points"])
    ax.scatter(xs, ys, s=16, color="#aa3377")
    ax.set_xticks(spec["x_ticks"])
    ax.set_yticks(spec["y_ticks"])
    ax.set_xlim(spec["x_ticks"][0], spec["x_ticks"][-1])
    ax.set_ylim(spec["y_ticks"][0], spec["y_ticks"][-1])
    return _save(fig)


def draw_diagram(labels, width=330, height=60, fills=("#dde8f5", "#f5e6d0")):
    """Boxes in a row, joined by arrows, one label per box."""
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    count = len(labels)
    box_width = (width - 20 - 24 * (count - 1)) / count
    for index, label in enumerate(labels):
        x = 10 + index * (box_width + 24)
        ax.add_patch(Rectangle((x, 12), box_width, height - 24, facecolor=fills[index % len(fills)],
                               edgecolor="black", linewidth=0.9))
        ax.text(x + box_width / 2, height / 2, label, ha="center", va="center", fontsize=9)
        if index:
            ax.add_patch(FancyArrowPatch((x - 24, height / 2), (x, height / 2), arrowstyle="-|>",
                                         mutation_scale=9, linewidth=0.9, color="black", shrinkA=0, shrinkB=0))
    return _save(fig)


def draw_bitmap(width=480, height=320):
    """A continuous-tone picture (smooth colour fields with noise), as PNG bytes."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:height, 0:width].astype(float)
    channels = []
    for phase in (0.0, 1.3, 2.6):
        field = (np.sin(x / 61.0 + phase) + np.cos(y / 43.0 - phase) + np.sin((x + y) / 97.0 + 2 * phase)) / 3
        channels.append(field)
    pixels = (np.stack(channels, axis=-1) + 1) / 2 * 200 + 30
    for cx, cy, radius, colour in ((150, 120, 70, (235, 200, 90)), (340, 210, 90, (60, 110, 170))):
        weight = np.clip(1.2 - np.hypot(x - cx, y - cy) / radius, 0, 1)[..., None]
        pixels = pixels * (1 - weight) + np.array(colour) * weight
    pixels += rng.normal(0, 6, pixels.shape)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def chart_truth(spec):
    """The chart part of a ground-truth entry."""
    truth = {
        "chart_type": spec["chart_type"], "title": spec["title"],
        "x_axis": {"title": spec["x_title"],
                   "ticks": [str(v) for v in spec.get("x_ticks", spec.get("categories", []))]},
        "y_axis": {"title": spec["y_title"], "ticks": [str(v) for v in spec["y_ticks"]]},
        "legend": [s["name"] for s in spec.get("series", []) if s["name"]],
    }
    if spec["chart_type"] == "bar":
        truth["series"] = [{"name": None, "bars": [{"category": category, "value": value}
                                                   for category, value in zip(spec["categories"], spec["values"])]}]
        truth["value_labels"] = spec["value_labels"]
    elif spec["chart_type"] == "line":
        truth["series"] = [{"name": s["name"], "points": [{"x": x, "y": y} for x, y in zip(spec["x"], s["y"])]}
                           for s in spec["series"]]
    else:
        truth["series"] = [{"name": None, "points": [{"x": x, "y": y} for x, y in spec["points"]]}]
    return truth


class Page:
    """Places paragraphs, figures and captions from top to bottom of one column and records the truth."""

    def __init__(self, doc, truth, page=None, column=(LEFT, RIGHT), top=72.0):
        self.page = page if page is not None else doc.new_page()
        self.truth, self.left, self.right, self.y = truth, column[0], column[1], top
        self.page_number = self.page.number + 1

    def paragraph(self, text=BODY, height=40):
        self.page.insert_textbox(pymupdf.Rect(self.left, self.y, self.right, self.y + height), text,
                                 fontsize=FONT_SIZE)
        self.y += height + 14

    def caption(self, text):
        """One line of caption text at the current position. Returns its box."""
        baseline = self.y + CAPTION_SIZE * 1.075
        self.page.insert_text((self.left, baseline), text, fontsize=CAPTION_SIZE)
        width = pymupdf.get_text_length(text, fontname="helv", fontsize=CAPTION_SIZE)
        box = [self.left, round(self.y, 2), round(self.left + width, 2), round(self.y + CAPTION_SIZE * 1.374, 2)]
        self.y += CAPTION_SIZE * 1.374
        return box

    def figure(self, kind, figure_type, drawing, caption=None, above=None, gap=8.0, gap_above=8.0, **extra):
        """A figure with a caption below (`caption`) and/or above (`above`), each given as
        (figure number, caption text, separator). `drawing` is PDF bytes, or PNG bytes with a size."""
        captions = {}
        if above:
            captions["above"] = self.caption(above[0] + above[2] + above[1])
            self.y += gap_above
        if isinstance(drawing, tuple):
            data, width, height = drawing
            x0 = (self.left + self.right - width) / 2
            box = pymupdf.Rect(x0, self.y, x0 + width, self.y + height)
            self.page.insert_image(box, stream=data)
        else:
            with pymupdf.open(stream=drawing, filetype="pdf") as source:
                width, height = source[0].rect.width, source[0].rect.height
                x0 = (self.left + self.right - width) / 2
                box = pymupdf.Rect(x0, self.y, x0 + width, self.y + height)
                self.page.show_pdf_page(box, source, 0)
        self.y += height
        if caption:
            self.y += gap
            captions["below"] = self.caption(caption[0] + caption[2] + caption[1])
        entry = {"page_number": self.page_number, "kind": kind, "figure_type": figure_type,
                 "pdf_bbox": [round(v, 2) for v in box], "caption": None, "figure_number": None,
                 "caption_position": None, "caption_pdf_bbox": None,
                 "caption_candidates": [captions[key] for key in sorted(captions)], **extra}
        if len(captions) == 1:      # with two candidates at the same distance the truth is "no caption"
            position, used = next(iter(captions.items()))
            number, text, _ = caption or above
            entry.update(caption=text, figure_number=number, caption_position=position, caption_pdf_bbox=used)
        self.truth.append(entry)
        self.y += 16
        return entry


def build(doc):
    truth = []

    page = Page(doc, truth)
    page.page.insert_text((LEFT, 60), "1. Overview", fontsize=14, fontname="hebo")
    page.paragraph()
    labels = ["Input", "Backbone", "Transformer", "Classifier"]
    page.figure("architecture_diagram", "diagram", draw_diagram(labels),
                caption=("Figure 1", "Overview of the proposed architecture.", ": "), labels=labels)
    page.paragraph()
    page.figure("generic_image", "image", (draw_bitmap(), 240.0, 160.0),
                caption=("Fig. 2", "Qualitative example produced by the method.", ". "), labels=[])
    page.paragraph()

    page = Page(doc, truth)
    page.paragraph()
    page.figure("bar_chart_value_labels", "chart", draw_bar(BAR_LABELLED),
                caption=("Figure 3", "Accuracy of four models on the benchmark.", ": "),
                chart=chart_truth(BAR_LABELLED))
    page.paragraph()
    page.figure("line_chart_caption_above", "chart", draw_line(LINE_SINGLE, box=False),
                above=("Figure 4", "Training loss over six epochs.", ": "), chart=chart_truth(LINE_SINGLE))
    page.paragraph()

    page = Page(doc, truth)
    page.paragraph()
    page.figure("scatter_plot", "chart", draw_scatter(SCATTER),
                caption=("Figure 5", "Accuracy against model size.", ": "), chart=chart_truth(SCATTER))
    page.paragraph()
    page.figure("multi_series_line_chart_legend", "chart", draw_line(LINE_MULTI),
                caption=("Figure 6", "Validation accuracy of two methods.", ": "), chart=chart_truth(LINE_MULTI))
    page.paragraph()

    # Two columns. The caption of the right figure stands ABOVE it, at the height of the lower edge of
    # the left figure: it is the caption nearest to the left figure, and it is not that figure's caption.
    left = Page(doc, truth, column=COLUMNS[0])
    left.paragraph(SHORT_BODY, height=52)
    bar = left.figure("two_column_bar_chart", "chart", draw_bar(BAR_PLAIN, width=236, height=150, color="#ee7733"),
                      caption=("Figure 7", "Error on three datasets.", ": "), gap=30.0, chart=chart_truth(BAR_PLAIN))
    left.paragraph(SHORT_BODY, height=52)
    right = Page(doc, truth, page=left.page, column=COLUMNS[1])
    right.paragraph(SHORT_BODY, height=52)
    right.paragraph(SHORT_BODY, height=52)
    right.y = bar["pdf_bbox"][3] - 10.0
    labels = ["Encoder", "Decoder"]
    right.figure("two_column_diagram_caption_above", "diagram",
                 draw_diagram(labels, width=200, height=54, fills=("#e3f1df", "#efe0f2")),
                 above=("Fig. 8(a)", "Two-stage variant.", " "), labels=labels)
    right.paragraph(SHORT_BODY, height=52)

    page = Page(doc, truth)
    page.paragraph()
    labels = ["Stage 1", "Stage 2", "Stage 3"]
    page.figure("ambiguous_captions", "diagram", draw_diagram(labels, width=300),
                above=("Figure 9", "A line of caption text above the drawing.", ": "),
                caption=("Figure 10", "A line of caption text below the drawing.", ": "), labels=labels)
    page.paragraph()
    return truth


def save(doc, name):
    os.makedirs(HERE, exist_ok=True)
    target = os.path.join(HERE, name)
    doc.save(target, garbage=4, deflate=True, no_new_id=True)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")
    return target


def main():
    with pymupdf.open() as doc:
        truth = build(doc)
        digital = save(doc, "synthetic_figures.pdf")

    with pymupdf.open(digital) as source, pymupdf.open() as out:
        for page in source:
            pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csRGB)
            out.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, pixmap=pix)
        save(out, "synthetic_figures_scanned.pdf")

    with open(os.path.join(HERE, "synthetic_figures.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"figures": truth}, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()

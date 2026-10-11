"""Phase 7 chart reading: crop of a chart + its text -> axes, labels, legend, series -> chart_data.

    crop of the page image + ChartText items (native text or Phase 4 OCR, with their boxes)
        ->  find_plot_area     the two axis lines
        ->  text roles         title, axis titles, tick labels, legend, value labels (by position)
        ->  calibrate_axis     tick label value <-> pixel, linear axes only
        ->  marks              colour groups -> connected components -> bars / lines / points
        ->  chart_data         every number marked "label", "ocr" or "visual_estimate"

Rules and geometry only: no model is loaded and nothing is interpreted. A number read off the pixels
is an estimate and is always marked as one; what cannot be determined is null or an empty list.
Another reader (for example a chart-to-table model) can replace this one by subclassing ChartExtractor.
"""
import math
import re
from dataclasses import dataclass

from .common import PdfPipelineError
from .ocr import group_rows

CHART_BAR = "bar"
CHART_LINE = "line"
CHART_SCATTER = "scatter"
CHART_UNKNOWN = "unknown"
# Classes of the schema that these rules never assign: histogram, box, area, pie.
CHART_TYPES = (CHART_BAR, CHART_LINE, CHART_SCATTER, "histogram", "box", "area", "pie", CHART_UNKNOWN)

VALUE_LABEL = "label"               # a number printed in the chart, read from the PDF's own text
VALUE_OCR = "ocr"                   # a number printed in the chart, read by OCR
VALUE_VISUAL = "visual_estimate"    # measured on the pixels against the axis ticks
VALUE_SOURCES = (VALUE_LABEL, VALUE_OCR, VALUE_VISUAL)

STATUS_PARSED = "parsed"            # type known and at least one number recovered
STATUS_PARTIAL = "partial"          # axes found; type or numbers missing (see "notes")
STATUS_FAILED = "failed"            # no plot area, or the reader raised
CHART_STATUSES = (STATUS_PARSED, STATUS_PARTIAL, STATUS_FAILED)

ROLE_TITLE = "title"
ROLE_X_TITLE = "x_axis_title"
ROLE_Y_TITLE = "y_axis_title"
ROLE_X_TICK = "x_tick"
ROLE_Y_TICK = "y_tick"
ROLE_LEGEND = "legend"
ROLE_VALUE_LABEL = "value_label"
ROLE_ANNOTATION = "annotation"
TEXT_ROLES = (ROLE_TITLE, ROLE_X_TITLE, ROLE_Y_TITLE, ROLE_X_TICK, ROLE_Y_TICK, ROLE_LEGEND, ROLE_VALUE_LABEL,
              ROLE_ANNOTATION)

TEXT_NATIVE = "native"
TEXT_OCR = "ocr"

AXIS_DARK = 160             # grey level (0-255) below which an unsaturated pixel can be an axis line
MARK_DARK = 128             # ... a black or grey mark
SATURATED = 60              # HSV saturation (0-255) from which a pixel counts as coloured
AXIS_MIN_SPAN = 0.4         # an axis line runs over at least this share of the crop
AXIS_MAX_THICKNESS = 6      # pixels
HUE_WINDOW = 9              # OpenCV hue units (0-179) on each side of a colour group's centre
MAX_COLOUR_GROUPS = 8
MIN_MARK_AREA = 6           # pixels; smaller components are noise
BAR_MIN_FILL = 0.85         # share of its bounding box a bar fills
POINT_MIN_FILL = 0.4
LINE_MIN_SPAN = 0.2         # a line covers at least this share of the plot width
MIN_TYPE_SCORE = 0.6        # share of the mark pixels the chosen type must explain
MIN_SCATTER_POINTS = 3
LINE_TOLERANCE = 2.0        # pixels a line may deviate from the reported polyline
MAX_LINE_POINTS = 50
TICK_TOLERANCE = 4.0        # pixels a tick label may lie off the fitted axis scale
LEGEND_HANDLE_REACH = 3.5   # a legend handle lies within this many text heights left of its label
VERTICAL_RATIO = 1.5        # a text box this many times taller than wide is rotated text

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


class ChartError(PdfPipelineError):
    """Base class for chart reading failures."""


class ChartExtractionError(ChartError):
    """One chart could not be read. The figure keeps its crop and its text."""


@dataclass
class ChartText:
    """One piece of text of a figure. bbox [x0, y0, x1, y1] is in pixels of the PAGE image.
    `confidence` is the OCR confidence, None for native text. `vertical`: the text is rotated.
    `role` is filled in by the chart reader (one of TEXT_ROLES, or None)."""
    text: str
    bbox: tuple
    confidence: float = None
    source: str = TEXT_NATIVE
    vertical: bool = False
    role: str = None


def parse_number(text):
    """The number a label consists of, or None: "92.1" -> 92.1, "−0.5" -> -0.5, "1,200" -> 1200.0,
    "45%" -> 45.0. Anything else ("1e3", "10^3", "0,5", "ResNet", "Q1") is None."""
    cleaned = _THOUSANDS.sub("", "".join((text or "").replace("−", "-").split()))
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return float(cleaned) if _NUMBER.fullmatch(cleaned) else None


def calibrate_axis(ticks):
    """Linear scale of an axis from its numeric tick labels, given as (pixel position, value) pairs.

    Returns {"scale": a, "offset": b, "used": n, "ticks": total} with value = a * pixel + b, or None.
    The scale is the one most tick labels agree with (within TICK_TOLERANCE pixels), fitted to those by
    least squares; it needs two labels with different values, and more than half of all labels must
    agree with it. A logarithmic axis, or labels that were misread, give None rather than a wrong scale.
    """
    ticks = [(float(pixel), float(value)) for pixel, value in ticks]
    best = []
    for i, (p1, v1) in enumerate(ticks):
        for p2, v2 in ticks[i + 1:]:
            if v1 == v2 or abs(p1 - p2) < 2 * TICK_TOLERANCE:
                continue
            slope = (p2 - p1) / (v2 - v1)                      # pixels per unit
            inliers = [(p, v) for p, v in ticks if abs(p1 + (v - v1) * slope - p) <= TICK_TOLERANCE]
            if len(inliers) > len(best):
                best = inliers
    if len({v for _, v in best}) < 2 or 2 * len(best) <= len(ticks):
        return None
    mean_p = sum(p for p, _ in best) / len(best)
    mean_v = sum(v for _, v in best) / len(best)
    spread = sum((p - mean_p) ** 2 for p, _ in best)
    scale = sum((p - mean_p) * (v - mean_v) for p, v in best) / spread
    return {"scale": scale, "offset": mean_v - scale * mean_p, "used": len(best), "ticks": len(ticks)}


def to_value(axis, pixel):
    """Value at a pixel position on a calibrated axis, rounded to what one pixel can resolve."""
    value = axis["scale"] * pixel + axis["offset"]
    decimals = max(0, -math.floor(math.log10(abs(axis["scale"])))) if axis["scale"] else 0
    return round(value, decimals) + 0.0     # + 0.0: no "-0.0"


class ChartExtractor:
    """Interface of a chart reader. Subclasses set `name` and implement `_extract`; they may load a
    model in `_load_model` (loaded once, on first use)."""

    name = ""

    def __init__(self):
        self._model = None
        self.load_count = 0

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        if not self.loaded:
            try:
                self._model = self._load_model()
            except Exception as exc:
                raise ChartError(f"Cannot load chart reader {self.name}: {exc}") from exc
            self.load_count += 1
        return self

    def extract_chart(self, image, texts, origin=(0, 0)):
        """Read one chart. `image` is the crop of the page image (PIL), `origin` the page pixel of its
        top-left corner and `texts` the ChartText items of the crop (boxes in page pixels); their `role`
        is filled in. Returns a chart_data object whose pixel geometry is in page pixels.
        Raises ChartExtractionError."""
        self.load()
        for item in texts:
            item.role = None
        try:
            return self._extract(image, texts, origin)
        except ChartError:
            raise
        except Exception as exc:
            raise ChartExtractionError(f"{type(exc).__name__}: {exc}") from exc

    def describe(self):
        return {"name": self.name}

    def _load_model(self):
        return True

    def _extract(self, image, texts, origin):
        raise NotImplementedError


def empty_chart_data(reader_name, status=STATUS_FAILED, notes=()):
    """A chart_data object with nothing recovered."""
    return {
        "status": status,
        "extractor": reader_name,
        "chart_type": CHART_UNKNOWN,
        "chart_type_score": None,
        "plot_area_image_bbox": None,
        "title": None,
        "x_axis": {"title": None, "ticks": [], "numeric": False, "calibrated": False},
        "y_axis": {"title": None, "ticks": [], "numeric": False, "calibrated": False},
        "legend": [],
        "series": [],
        "annotations": [],
        "notes": list(notes),
    }


class RuleBasedChartExtractor(ChartExtractor):
    """Reads simple bar charts (vertical bars), line charts and scatter plots with OpenCV geometry.

    It needs two axis lines that meet at the lower left, linear axes with numeric tick labels for any
    number, and series that differ in colour. Everything else comes back as "unknown" or without
    numbers, with the reason in "notes"."""

    name = "rule-based-opencv"

    def describe(self):
        from .layout import _package_version
        return {"name": self.name, "opencv": _package_version("opencv-contrib-python"), "device": "cpu"}

    def find_axes(self, image, texts, origin=(0, 0)):
        """{"plot": (x0, y0, x1, y1) in crop pixels, "x": calibration or None, "y": ...}, or None when
        the crop has no pair of axis lines."""
        import numpy as np
        pixels = np.asarray(image.convert("RGB"))
        plot = find_plot_area(pixels)
        if plot is None:
            return None
        local = _localize(texts, origin)
        roles = _assign_outer_roles(local, plot, set())
        return {"plot": plot, "x": calibrate_axis(_tick_pairs(roles[ROLE_X_TICK], 0)),
                "y": calibrate_axis(_tick_pairs(roles[ROLE_Y_TICK], 1))}

    def looks_like_chart(self, image, texts, origin=(0, 0)):
        """True for a crop with two axis lines and an axis whose three or more numeric tick labels fit a
        linear scale. Used for figures the layout model did not label as a chart."""
        axes = self.find_axes(image, texts, origin)
        return bool(axes) and any(axes[key] and axes[key]["used"] >= 3 for key in ("x", "y"))

    def _extract(self, image, texts, origin):
        import cv2
        import numpy as np
        pixels = np.asarray(image.convert("RGB"))
        plot = find_plot_area(pixels)
        if plot is None:
            return empty_chart_data(self.name, notes=["no plot area: no pair of axis lines found"])
        px0, py0, px1, py1 = plot
        local = _localize(texts, origin)
        notes = []

        # Marks: coloured and dark pixels inside the plot area, text removed.
        hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
        grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        inside = np.zeros(grey.shape, bool)
        inside[py0 + 1:py1 - 2, px0 + 3:px1 - 2] = True
        coloured = (hsv[..., 1] >= SATURATED) & (hsv[..., 2] >= 40)
        dark = (hsv[..., 1] < SATURATED) & (grey < MARK_DARK)
        for item, box in local:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            dark[max(y0 - 2, 0):y1 + 2, max(x0 - 2, 0):x1 + 2] = False
        tolerance = max(4, int(0.02 * max(grey.shape)))
        horizontal, vertical, _ = _axis_lines(pixels)
        for row, start, end in horizontal:          # the frame of the plot area is not a mark
            if abs(row - py0) <= tolerance:
                dark[max(row - AXIS_MAX_THICKNESS, 0):row + AXIS_MAX_THICKNESS + 1, start:end] = False
        for column, start, end in vertical:
            if abs(column - px1) <= tolerance:
                dark[start:end, max(column - AXIS_MAX_THICKNESS, 0):column + AXIS_MAX_THICKNESS + 1] = False
        groups = _colour_groups(hsv, coloured, inside, pixels)
        if (dark & inside).sum() >= MIN_MARK_AREA:
            groups.append({"mask": dark, "colour": "#000000"})

        # Legend: a label with a patch of a series colour directly left of it.
        legend = []
        for item, box in sorted(local, key=lambda pair: (pair[1][1], pair[1][0])):
            group = _legend_group(box, groups, plot) if not item.vertical else None
            if group is not None:
                item.role = ROLE_LEGEND
                legend.append((item, box, group))
        for item, box, group in legend:
            height = box[3] - box[1]
            x0, x1 = int(box[0] - LEGEND_HANDLE_REACH * height), int(box[0])
            for other in groups:
                other["mask"][max(int(box[1] - 0.3 * height), 0):int(box[3] + 0.3 * height) + 1, max(x0, 0):x1] = False
            group.setdefault("name", item.text)

        roles = _assign_outer_roles(local, plot, {id(item) for item, _, _ in legend})
        x_axis = calibrate_axis(_tick_pairs(roles[ROLE_X_TICK], 0))
        y_axis = calibrate_axis(_tick_pairs(roles[ROLE_Y_TICK], 1))
        x_numeric = bool(roles[ROLE_X_TICK]) and all(parse_number(i.text) is not None for i, _ in roles[ROLE_X_TICK])
        y_numeric = bool(roles[ROLE_Y_TICK]) and all(parse_number(i.text) is not None for i, _ in roles[ROLE_Y_TICK])
        if roles[ROLE_Y_TICK] and y_axis is None:
            notes.append("y axis not calibrated: its tick labels do not give a linear numeric scale")
        elif y_axis is None:
            notes.append("y axis not calibrated: no tick labels found")
        if x_numeric and x_axis is None:
            notes.append("x axis not calibrated: its tick labels do not give a linear numeric scale")

        # Components of every colour group, each one a bar, a line, a point or something else.
        zero_y = None
        if y_axis and y_axis["scale"]:
            zero_y = -y_axis["offset"] / y_axis["scale"]
            zero_y = zero_y if py0 <= zero_y <= py1 else None
        totals = {CHART_BAR: 0, CHART_LINE: 0, CHART_SCATTER: 0, "other": 0}
        for group in groups:
            group["marks"] = _components(group["mask"] & inside, plot, zero_y)
            for mark in group["marks"]:
                totals[mark["kind"]] += mark["area"]
        chart_type, score = CHART_UNKNOWN, None
        if sum(totals.values()):
            best = max((CHART_BAR, CHART_LINE, CHART_SCATTER), key=lambda kind: totals[kind])
            score = totals[best] / sum(totals.values())
            count = sum(1 for group in groups for mark in group["marks"] if mark["kind"] == best)
            if score >= MIN_TYPE_SCORE and count >= (MIN_SCATTER_POINTS if best == CHART_SCATTER else 1):
                chart_type = best
            else:
                notes.append("chart type not determined: the marks are not clearly bars, lines or points")
        else:
            notes.append("chart type not determined: no marks found in the plot area")

        inner = [(item, box) for item, box in local if item.role is None
                 and px0 <= (box[0] + box[2]) / 2 <= px1 and py0 <= (box[1] + box[3]) / 2 <= py1]
        series = []
        for group in groups:
            marks = sorted((m for m in group["marks"] if m["kind"] == chart_type), key=lambda m: m["box"][0])
            if not marks:
                continue
            entry = {"name": group.get("name"), "colour": group["colour"]}
            if chart_type == CHART_BAR:
                entry["bars"] = [_bar(mark, roles[ROLE_X_TICK], inner, y_axis, zero_y, origin) for mark in marks]
            elif chart_type == CHART_LINE:
                entry["points"] = _line_points(group["mask"] & inside, marks, px1 - px0, x_axis, y_axis,
                                               roles[ROLE_X_TICK], origin, notes)
            else:
                entry["points"] = _scatter_points(marks, x_axis, y_axis, origin, notes)
            series.append(entry)
        for item, _ in inner:
            if item.role is None:
                item.role = ROLE_ANNOTATION

        has_numbers = any(bar["value"] is not None for s in series for bar in s.get("bars", ())) or any(
            point["y"] is not None for s in series for point in s.get("points", ()))
        if chart_type != CHART_UNKNOWN and not has_numbers:
            notes.append("no data values recovered")
        data = empty_chart_data(self.name, STATUS_PARSED if has_numbers else STATUS_PARTIAL, notes)
        data.update(
            chart_type=chart_type,
            chart_type_score=score,
            plot_area_image_bbox=[px0 + origin[0], py0 + origin[1], px1 + origin[0], py1 + origin[1]],
            title=_joined(roles[ROLE_TITLE]),
            x_axis={"title": _joined(roles[ROLE_X_TITLE]), "ticks": [i.text for i, _ in roles[ROLE_X_TICK]],
                    "numeric": x_numeric, "calibrated": x_axis is not None},
            y_axis={"title": _joined(roles[ROLE_Y_TITLE]),
                    "ticks": [i.text for i, _ in sorted(roles[ROLE_Y_TICK], key=lambda pair: -pair[1][1])],
                    "numeric": y_numeric, "calibrated": y_axis is not None},
            legend=[item.text for item, _, _ in legend],
            series=series,
            annotations=[item.text for item, _ in inner if item.role == ROLE_ANNOTATION],
        )
        return data


def find_plot_area(pixels):
    """The plot area (x0, y0, x1, y1) of a chart crop (RGB array), in crop pixels: from the y axis
    line to the right end of the x axis line, and from the top of the y axis line down to the x axis.

    An axis is a thin, dark, unsaturated line over at least AXIS_MIN_SPAN of the crop. The x axis is
    the lowest such horizontal line that a vertical one meets at its left end. None without that pair.
    """
    horizontal, vertical, (height, width) = _axis_lines(pixels)
    tolerance = max(4, int(0.02 * max(width, height)))
    for row, x_start, x_end in sorted(horizontal, reverse=True):
        for column, y_start, y_end in sorted(vertical):
            if x_start - tolerance <= column <= x_start + tolerance and y_end >= row - tolerance and y_start < row:
                if x_end - column >= 20 and row - y_start >= 20:
                    return (int(column), int(y_start), int(x_end), int(row))
    return None


def _axis_lines(pixels):
    """Thin dark lines of a crop that are long enough to be an axis: (horizontal [(row, x start,
    x end)], vertical [(column, y start, y end)], (height, width))."""
    import cv2
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    dark = (hsv[..., 1] < SATURATED) & (grey < AXIS_DARK)
    height, width = dark.shape
    return _thin_lines(dark, AXIS_MIN_SPAN * width), _thin_lines(dark.T, AXIS_MIN_SPAN * height), (height, width)


def _thin_lines(mask, min_length):
    """Lines along the rows of a boolean mask: [(row, start, end)] for every group of adjacent rows, at
    most AXIS_MAX_THICKNESS thick, that hold an unbroken run of at least `min_length` pixels."""
    import numpy as np
    rows, width = mask.shape
    padded = np.zeros((rows, width + 2), np.int8)
    padded[:, 1:-1] = mask
    steps = np.diff(padded, axis=1)
    found = []
    for row in np.flatnonzero(mask.sum(axis=1) >= min_length):
        starts, ends = np.flatnonzero(steps[row] == 1), np.flatnonzero(steps[row] == -1)
        longest = int(np.argmax(ends - starts))
        if ends[longest] - starts[longest] >= min_length:
            found.append((int(row), int(starts[longest]), int(ends[longest])))
    lines, group = [], []
    for entry in found + [None]:
        if entry is not None and group and entry[0] == group[-1][0] + 1:
            group.append(entry)
            continue
        if group and len(group) <= AXIS_MAX_THICKNESS:
            middle = max(group, key=lambda e: e[2] - e[1])
            lines.append((group[len(group) // 2][0], middle[1], middle[2]))
        group = [entry] if entry is not None else []
    return lines


def _localize(texts, origin):
    """[(item, box in crop pixels)] for the items that have text."""
    return [(item, (item.bbox[0] - origin[0], item.bbox[1] - origin[1], item.bbox[2] - origin[0],
                    item.bbox[3] - origin[1])) for item in texts if item.text and item.text.strip()]


def _assign_outer_roles(local, plot, taken):
    """Give the texts outside the plot area their roles, by position. Returns {role: [(item, box)]}
    for the title, the axis titles and the tick labels (x ticks left to right)."""
    px0, py0, px1, py1 = plot
    roles = {role: [] for role in (ROLE_TITLE, ROLE_X_TITLE, ROLE_Y_TITLE, ROLE_X_TICK, ROLE_Y_TICK)}
    free = [(item, box) for item, box in local if id(item) not in taken]

    below = [(box, item) for item, box in free if (box[1] + box[3]) / 2 > py1
             and px0 - (box[2] - box[0]) <= (box[0] + box[2]) / 2 <= px1 + (box[2] - box[0])]
    rows = group_rows(below)
    if rows:
        first = rows[0]
        if len(first) > 1 or parse_number(first[0][1].text) is not None:
            roles[ROLE_X_TICK] = [(item, box) for box, item in first]
            rows = rows[1:]
        if rows:
            roles[ROLE_X_TITLE] = [(item, box) for box, item in rows[0]]

    left = [(item, box) for item, box in free if (box[0] + box[2]) / 2 < px0
            and py0 - (box[3] - box[1]) <= (box[1] + box[3]) / 2 <= py1 + (box[3] - box[1])]
    roles[ROLE_Y_TITLE] = [(item, box) for item, box in left if item.vertical]
    ticks = [(item, box) for item, box in left if not item.vertical]
    words = [pair for pair in ticks if parse_number(pair[0].text) is None]
    if len(words) == 1 and len(ticks) >= 3 and not roles[ROLE_Y_TITLE]:      # a horizontal axis title
        roles[ROLE_Y_TITLE], ticks = words, [pair for pair in ticks if pair is not words[0]]
    roles[ROLE_Y_TICK] = ticks

    above = [(box, item) for item, box in free if (box[1] + box[3]) / 2 < py0
             and px0 - 0.2 * (px1 - px0) <= (box[0] + box[2]) / 2 <= px1 + 0.2 * (px1 - px0)]
    roles[ROLE_TITLE] = [(item, box) for row in group_rows(above) for box, item in row]

    for role, pairs in roles.items():
        for item, _ in pairs:
            item.role = role
    return roles


def _tick_pairs(ticks, axis):
    """(pixel centre, value) of the numeric tick labels, along x (axis 0) or y (axis 1)."""
    return [((box[axis] + box[axis + 2]) / 2, parse_number(item.text)) for item, box in ticks
            if parse_number(item.text) is not None]


def _joined(pairs):
    text = " ".join(item.text.strip() for item, _ in pairs)
    return text or None


def _colour_groups(hsv, coloured, inside, pixels):
    """Groups of coloured pixels by hue: [{"mask", "colour"}], largest first. The mask of a group
    covers the whole crop (so that a legend outside the plot area is found); its size is counted
    inside the plot area."""
    import numpy as np
    hue = hsv[..., 0].astype(int)
    remaining = coloured & inside
    minimum = max(30, int(0.0005 * inside.sum()))
    groups = []
    while len(groups) < MAX_COLOUR_GROUPS:
        counts = np.bincount(hue[remaining], minlength=180)
        window = sum(np.roll(counts, shift) for shift in range(-HUE_WINDOW, HUE_WINDOW + 1))
        centre = int(np.argmax(window))
        if window[centre] < minimum:
            break
        distance = np.abs(hue - centre)
        near = np.minimum(distance, 180 - distance) <= HUE_WINDOW
        mask = coloured & near
        red, green, blue = (int(np.median(pixels[..., c][mask & inside])) for c in range(3))
        groups.append({"mask": mask, "colour": f"#{red:02x}{green:02x}{blue:02x}"})
        remaining = remaining & ~near
    return groups


def _legend_group(box, groups, plot):
    """The colour group whose legend handle lies directly left of a text box, or None. A handle is a
    patch or a short line of the group's colour beside the text that does not go on above or below the
    text line (a bar or a curve passing by does)."""
    x0, y0, x1, y1 = box
    height = y1 - y0
    if height < 4 or (x0 + x1) / 2 < plot[0] or (y0 + y1) / 2 > plot[3]:
        return None
    left, right = int(x0 - LEGEND_HANDLE_REACH * height), int(x0 - 0.1 * height)
    if left < 0 or right <= left:
        return None
    best = None
    for group in groups:
        mask = group["mask"]
        beside = int(mask[max(int(y0), 0):int(y1) + 1, left:right].sum())
        if beside < max(6, 0.3 * height):
            continue
        around = int(mask[max(int(y0 - 1.5 * height), 0):int(y1 + 1.5 * height) + 1, left:right].sum())
        if around - beside <= 0.25 * beside and (best is None or beside > best[0]):
            best = (beside, group)
    return best[1] if best else None


def _components(mask, plot, zero_y):
    """Connected components of a mark mask, each classed as a bar, a line, a point or "other":
    [{"kind", "box": (x0, y0, x1, y1), "area", "thin", "centre": (x, y)}]."""
    import cv2
    import numpy as np
    px0, py0, px1, py1 = plot
    plot_width, plot_height = px1 - px0, py1 - py0
    tolerance = max(4, int(0.015 * plot_height))
    max_point = max(8, 0.08 * min(plot_width, plot_height))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    marks = []
    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area < MIN_MARK_AREA:
            continue
        fill = area / (width * height)
        on_baseline = abs(y + height - py1) <= tolerance + 2 or (
            zero_y is not None and min(abs(y + height - zero_y), abs(y - zero_y)) <= tolerance)
        thin = _stroke_thickness(labels[y:y + height, x:x + width] == index) <= max(8, 0.04 * plot_height)
        if width >= LINE_MIN_SPAN * plot_width and thin:
            kind = CHART_LINE
        elif fill >= BAR_MIN_FILL and width >= 4 and on_baseline:
            kind = CHART_BAR
        elif (width <= max_point and height <= max_point and fill >= POINT_MIN_FILL and area >= 9
              and 0.5 <= width / height <= 2.0):
            kind = CHART_SCATTER
        else:
            kind = "other"
        marks.append({"kind": kind, "box": (x, y, x + width, y + height), "area": area, "thin": thin,
                      "centre": (float(centres[index][0]) + 0.5, float(centres[index][1]) + 0.5)})
    # A line that another series crosses is drawn in pieces: thin pieces that are not bars or points
    # and together span the width of a line are one line.
    pieces = [mark for mark in marks if mark["thin"] and mark["kind"] in (CHART_LINE, "other")]
    covered = np.zeros(mask.shape[1], bool)
    for mark in pieces:
        covered[mark["box"][0]:mark["box"][2]] = True
    if covered.sum() >= LINE_MIN_SPAN * plot_width:
        for mark in pieces:
            mark["kind"] = CHART_LINE
    return marks


def _stroke_thickness(component):
    """Median number of pixels a component has per column."""
    import numpy as np
    per_column = component.sum(axis=0)
    return float(np.median(per_column[per_column > 0]))


def _nearest_tick(ticks, x):
    """Text of the x tick label nearest to pixel x, or None when none is within half a tick spacing."""
    if not ticks:
        return None
    centres = sorted((box[0] + box[2]) / 2 for _, box in ticks)
    spacing = min((b - a for a, b in zip(centres, centres[1:])), default=None)
    item, box = min(ticks, key=lambda pair: abs((pair[1][0] + pair[1][2]) / 2 - x))
    reach = spacing / 2 if spacing else (box[2] - box[0])
    return item.text if abs((box[0] + box[2]) / 2 - x) <= reach else None


def _bar(mark, x_ticks, inner, y_axis, zero_y, origin):
    """One bar: its category (the tick label under it), its height measured against the y axis, and
    the number printed above it when there is one."""
    x0, y0, x1, y1 = mark["box"]
    downward = zero_y is not None and abs(y0 - zero_y) < abs(y1 - zero_y)      # hangs below the zero line
    estimate = to_value(y_axis, y1 if downward else y0) if y_axis else None
    value, source = estimate, VALUE_VISUAL if estimate is not None else None
    for item, box in inner:
        number, height = parse_number(item.text), box[3] - box[1]
        beside_end = -0.5 * height <= (box[1] - y1 if downward else y0 - box[3]) <= 1.5 * height
        if number is not None and item.role is None and x0 <= (box[0] + box[2]) / 2 <= x1 and beside_end:
            item.role = ROLE_VALUE_LABEL
            value, source = number, VALUE_LABEL if item.source == TEXT_NATIVE else VALUE_OCR
            break
    return {"category": _nearest_tick(x_ticks, (x0 + x1) / 2), "value": value, "value_source": source,
            "visual_estimate": estimate,
            "image_bbox": [x0 + origin[0], y0 + origin[1], x1 + origin[0], y1 + origin[1]]}


def _point(x, y, x_axis, y_axis, x_ticks, origin):
    x_value = to_value(x_axis, x) if x_axis else _nearest_tick(x_ticks, x)
    y_value = to_value(y_axis, y) if y_axis else None
    return {"x": x_value, "y": y_value, "value_source": VALUE_VISUAL if y_value is not None else None,
            "image_point": [round(x + origin[0], 1), round(y + origin[1], 1)]}


def _line_points(mask, marks, plot_width, x_axis, y_axis, x_ticks, origin, notes):
    """The corner points of a line series: the centre of the stroke in every pixel column, reduced to
    the polyline that stays within LINE_TOLERANCE pixels of it. For a line drawn from point to point
    these are its data points; a smooth curve gives the corners of an approximation."""
    import cv2
    import numpy as np
    x0, x1 = min(m["box"][0] for m in marks), max(m["box"][2] for m in marks)
    y0, y1 = min(m["box"][1] for m in marks), max(m["box"][3] for m in marks)
    part = mask[y0:y1, x0:x1]
    columns = np.flatnonzero(part.any(axis=0))
    centres = np.array([[x0 + column + 0.5, y0 + float(np.median(np.flatnonzero(part[:, column]))) + 0.5]
                        for column in columns], np.float32)
    tolerance = LINE_TOLERANCE
    corners = cv2.approxPolyDP(centres.reshape(-1, 1, 2), tolerance, False).reshape(-1, 2)
    while len(corners) > MAX_LINE_POINTS:
        tolerance *= 1.5
        corners = cv2.approxPolyDP(centres.reshape(-1, 1, 2), tolerance, False).reshape(-1, 2)
    if tolerance > LINE_TOLERANCE:
        notes.append(f"a curve was reduced to {len(corners)} points (within {tolerance:.1f} px)")
    corners = merge_corners(corners.tolist(), max(6.0, 3 * _stroke_thickness(part)), max(12.0, 0.04 * plot_width))
    return [_point(x, y, x_axis, y_axis, x_ticks, origin) for x, y in corners]


def merge_corners(corners, reach, corner_reach):
    """Clean the corner points of a polyline ([(x, y)], in drawing order).

    Corners closer together than `reach` (the size of a marker) become one point, their mean. Two
    neighbouring corners closer than `corner_reach` that cut off one rounded or blurred corner are
    replaced by the point where the segments before and after them meet, when that point lies within
    3 * LINE_TOLERANCE of the cut.
    """
    groups = []
    for x, y in corners:
        if groups and math.hypot(x - groups[-1][-1][0], y - groups[-1][-1][1]) <= reach:
            groups[-1].append((x, y))
        else:
            groups.append([(x, y)])
    points = [(sum(x for x, _ in g) / len(g), sum(y for _, y in g) / len(g)) for g in groups]
    index = 1
    while index + 2 < len(points):
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = points[index - 1:index + 3]
        if math.hypot(cx - bx, cy - by) <= corner_reach:
            ux, uy, vx, vy = bx - ax, by - ay, cx - dx, cy - dy
            cross = ux * vy - uy * vx
            if abs(cross) > 1e-9:
                t = ((dx - ax) * vy - (dy - ay) * vx) / cross
                meet = (ax + t * ux, ay + t * uy)
                if math.hypot(meet[0] - (bx + cx) / 2, meet[1] - (by + cy) / 2) <= 3 * LINE_TOLERANCE:
                    points[index:index + 2] = [meet]
                    continue
        index += 1
    return points


def _scatter_points(marks, x_axis, y_axis, origin, notes):
    areas = sorted(mark["area"] for mark in marks)
    median = areas[len(areas) // 2]
    merged = sum(1 for mark in marks if mark["area"] > 1.8 * median)
    if merged:
        notes.append(f"{merged} mark(s) are larger than one point and may be several points that touch")
    return [_point(mark["centre"][0], mark["centre"][1], x_axis, y_axis, [], origin) for mark in marks]

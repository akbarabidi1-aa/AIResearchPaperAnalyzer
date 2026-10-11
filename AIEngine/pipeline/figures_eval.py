"""Small evaluation helpers for extracted figures: captions, figure numbers, figure types, and for
charts their type, labels, legend and numbers.

Engineering checks on fixtures with known content. Not a benchmark.
"""
import math

from .charts import VALUE_LABEL, VALUE_OCR
from .figures import TYPE_CHART
from .layout_eval import match_regions
from .ocr import clean_line

POINT_MATCH_LIMIT = 0.05        # a scatter point is found when one lies within this share of the axis ranges


def _same_text(expected, found):
    return clean_line(expected or "") == clean_line(found or "")


def _axis_range(ticks):
    """Span of an axis from its tick labels ("0" ... "100" -> 100.0), or None."""
    values = []
    for tick in ticks:
        try:
            values.append(float(tick))
        except ValueError:
            pass
    return max(values) - min(values) if len(values) >= 2 and max(values) > min(values) else None


def _errors(pairs):
    """{"count", "mae", "rmse"} of (expected, found) number pairs; mae and rmse are None without pairs."""
    differences = [found - expected for expected, found in pairs]
    if not differences:
        return {"count": 0, "mae": None, "rmse": None}
    return {"count": len(differences), "mae": sum(abs(d) for d in differences) / len(differences),
            "rmse": math.sqrt(sum(d * d for d in differences) / len(differences))}


def evaluate_chart(expected, chart):
    """Compare a chart_data object (or None) with the "chart" of a ground-truth figure.

    Returns
        chart_type_match
        titles_expected / titles_correct    chart title and the two axis titles that the truth has
        ticks_expected / ticks_correct      tick labels of both axes that were found on the right axis
        legend_expected / legend_correct    legend entries found; legend_exact: the same list, in order
        labelled_expected / labelled_exact  values printed in the chart (value labels) / read exactly,
                                            with value_source "label" or "ocr"
        points_expected / points_found      values that can only be measured (bar heights, line and
                                            scatter points) / those for which an estimate was matched
        y_errors, x_errors                  (expected, estimate) pairs of the matched estimates
        mae, rmse                           of the y estimates, in the units of the chart's y axis
        y_range                             span of the y axis, to compare errors between charts
    """
    chart = chart or {"chart_type": None, "title": None, "x_axis": {"title": None, "ticks": []},
                      "y_axis": {"title": None, "ticks": []}, "legend": [], "series": []}
    titles = [(expected["title"], chart["title"]), (expected["x_axis"]["title"], chart["x_axis"]["title"]),
              (expected["y_axis"]["title"], chart["y_axis"]["title"])]
    titles = [(want, got) for want, got in titles if want is not None]
    ticks_expected = ticks_correct = 0
    for axis in ("x_axis", "y_axis"):
        found = [clean_line(tick) for tick in chart[axis]["ticks"]]
        for tick in expected[axis]["ticks"]:
            ticks_expected += 1
            if clean_line(tick) in found:
                found.remove(clean_line(tick))
                ticks_correct += 1
    legend_found = [clean_line(entry) for entry in chart["legend"]]

    labelled_expected = labelled_exact = points_expected = 0
    y_errors, x_errors = [], []
    y_range, x_range = _axis_range(expected["y_axis"]["ticks"]), _axis_range(expected["x_axis"]["ticks"])
    for index, series in enumerate(expected["series"]):
        named = [s for s in chart["series"] if series["name"] and _same_text(series["name"], s.get("name"))]
        predicted = named[0] if named else (chart["series"][index] if not series["name"]
                                            and index < len(chart["series"]) else {})
        if "bars" in series:
            bars = {clean_line(bar["category"] or ""): bar for bar in predicted.get("bars", ())}
            for bar in series["bars"]:
                found = bars.get(clean_line(bar["category"]))
                points_expected += 1
                if found and found.get("visual_estimate") is not None:
                    y_errors.append((bar["value"], found["visual_estimate"]))
                if expected.get("value_labels"):
                    labelled_expected += 1
                    labelled_exact += bool(found and found["value_source"] in (VALUE_LABEL, VALUE_OCR)
                                           and found["value"] == bar["value"])
            continue
        candidates = [p for p in predicted.get("points", ()) if isinstance(p["x"], (int, float)) and p["y"] is not None]
        xs = sorted(point["x"] for point in series["points"])
        spacing = min((b - a for a, b in zip(xs, xs[1:]) if b > a), default=None)
        for point in series["points"]:
            points_expected += 1
            if not candidates:
                continue
            if expected["chart_type"] == "scatter" and x_range and y_range:
                nearest = min(candidates, key=lambda p: math.hypot((p["x"] - point["x"]) / x_range,
                                                                   (p["y"] - point["y"]) / y_range))
                close = math.hypot((nearest["x"] - point["x"]) / x_range,
                                   (nearest["y"] - point["y"]) / y_range) <= POINT_MATCH_LIMIT
            else:
                nearest = min(candidates, key=lambda p: abs(p["x"] - point["x"]))
                close = spacing is None or abs(nearest["x"] - point["x"]) <= spacing / 2
            if close:
                candidates.remove(nearest)
                y_errors.append((point["y"], nearest["y"]))
                x_errors.append((point["x"], nearest["x"]))
    errors = _errors(y_errors)
    return {
        "chart_type_match": chart["chart_type"] == expected["chart_type"],
        "titles_expected": len(titles),
        "titles_correct": sum(1 for want, got in titles if _same_text(want, got)),
        "ticks_expected": ticks_expected,
        "ticks_correct": ticks_correct,
        "legend_expected": len(expected["legend"]),
        "legend_correct": sum(1 for entry in expected["legend"] if clean_line(entry) in legend_found),
        "legend_exact": [clean_line(entry) for entry in expected["legend"]] == legend_found,
        "labelled_expected": labelled_expected,
        "labelled_exact": labelled_exact,
        "points_expected": points_expected,
        "points_found": len(y_errors),
        "y_errors": y_errors,
        "x_errors": x_errors,
        "mae": errors["mae"],
        "rmse": errors["rmse"],
        "y_range": y_range,
    }


def evaluate_figures(truth_figures, document, iou_threshold=0.3):
    """Compare a figures.json object with ground-truth figures ([{"page_number", "pdf_bbox",
    "figure_type", "caption", "figure_number", optionally "labels" and "chart"}]).

    Each ground-truth figure is paired with the extracted figure at the same place on the same page
    (boxes overlapping with IoU >= iou_threshold). Returns {"rows": [...], and the totals}:

        figures, found
        type_accuracy                  figure_type equal
        caption_accuracy               caption text equal; "no caption" counts when the truth has none
        number_accuracy                figure number equal; null counts when the truth has none
        label_accuracy                 share of the words printed in the generic figures that were read
        charts, chart_type_accuracy
        axis_title_accuracy, tick_label_accuracy, legend_accuracy
        exact_value_accuracy           values printed in a chart that were read exactly
        estimated_points_found         share of the measured values for which an estimate exists
        normalized_mae, normalized_rmse   errors of the estimates as a share of each chart's y-axis span

    All rates are over all ground-truth figures (or charts): one that was not found counts as wrong.
    A rate without anything to count is None.
    """
    predicted = [{**figure, "type": "FIGURE"} for page in document["pages"] for figure in page["figures"]]
    truth = [{**figure, "type": "FIGURE"} for figure in truth_figures]
    by_box = {(figure["page_number"], tuple(figure["pdf_bbox"])): figure for figure in predicted}

    rows = []
    for expected, match in zip(truth, match_regions(truth, predicted, iou_threshold)):
        figure = by_box.get((match["page_number"], tuple(match["predicted_bbox"] or ())))
        row = {
            "page_number": expected["page_number"], "kind": expected.get("kind"),
            "figure_id": figure and figure["figure_id"],
            "figure_type": figure and figure["figure_type"],
            "type_match": bool(figure) and figure["figure_type"] == expected["figure_type"],
            "caption_match": bool(figure) and _same_text(expected["caption"], figure["caption"]),
            "number_match": bool(figure) and _same_text(expected["figure_number"], figure["figure_number"]),
            "labels_expected": 0, "labels_found": 0, "chart": None,
        }
        if expected["figure_type"] == TYPE_CHART:
            row["chart"] = evaluate_chart(expected["chart"], figure and figure["chart_data"])
        else:
            words = [clean_line(item["text"]) for item in (figure or {}).get("ocr_text", ())]
            row["labels_expected"] = len(expected.get("labels", ()))
            row["labels_found"] = sum(1 for label in expected.get("labels", ()) if clean_line(label) in words)
        rows.append(row)

    def rate(correct, total):
        return correct / total if total else None

    def total(key, items):
        return sum(item[key] for item in items)

    charts = [row["chart"] for row in rows if row["chart"]]
    scaled = [(found - expected) / chart["y_range"] for chart in charts if chart["y_range"]
              for expected, found in chart["y_errors"]]
    return {
        "rows": rows,
        "figures": len(rows),
        "found": sum(1 for row in rows if row["figure_id"]),
        "type_accuracy": rate(sum(row["type_match"] for row in rows), len(rows)),
        "caption_accuracy": rate(sum(row["caption_match"] for row in rows), len(rows)),
        "number_accuracy": rate(sum(row["number_match"] for row in rows), len(rows)),
        "label_accuracy": rate(total("labels_found", rows), total("labels_expected", rows)),
        "charts": len(charts),
        "chart_type_accuracy": rate(sum(chart["chart_type_match"] for chart in charts), len(charts)),
        "axis_title_accuracy": rate(total("titles_correct", charts), total("titles_expected", charts)),
        "tick_label_accuracy": rate(total("ticks_correct", charts), total("ticks_expected", charts)),
        "legend_accuracy": rate(total("legend_correct", charts), total("legend_expected", charts)),
        "labelled_values": total("labelled_expected", charts),
        "exact_value_accuracy": rate(total("labelled_exact", charts), total("labelled_expected", charts)),
        "estimated_points": total("points_expected", charts),
        "estimated_points_found": rate(total("points_found", charts), total("points_expected", charts)),
        "normalized_mae": sum(abs(e) for e in scaled) / len(scaled) if scaled else None,
        "normalized_rmse": math.sqrt(sum(e * e for e in scaled) / len(scaled)) if scaled else None,
    }

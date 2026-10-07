"""Small evaluation helpers for layout detection: IoU, region matching, class accuracy.

Not a benchmark. Boxes are [x0, y0, x1, y1] with x0 < x1 and y0 < y1, in any one coordinate system.
"""
import math


def is_valid_bbox(bbox):
    """True for four finite numbers [x0, y0, x1, y1] with a positive area."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]


def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_iou(a, b):
    """Intersection over union of two boxes: 0.0 (disjoint or only touching) to 1.0 (identical)."""
    for bbox in (a, b):
        if not is_valid_bbox(bbox):
            raise ValueError(f"not a valid [x0, y0, x1, y1] box: {bbox!r}")

    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / (bbox_area(a) + bbox_area(b) - intersection)


def match_regions(ground_truth, predicted, iou_threshold=0.5, bbox_key="pdf_bbox"):
    """Pair every ground-truth region with the best overlapping prediction on the same page.

    Both arguments are lists of dicts with "page_number", "type" and `bbox_key`. A prediction is used
    at most once; pairs are assigned in order of decreasing IoU, whatever the classes are. Returns one
    row per ground-truth region, in input order:

        {"page_number", "truth_type", "truth_bbox", "predicted_type", "predicted_bbox", "iou", "correct_class"}

    A ground-truth region with no prediction at IoU >= iou_threshold has predicted_type None, iou 0.0
    and correct_class False.
    """
    candidates = []
    for t, truth in enumerate(ground_truth):
        for p, prediction in enumerate(predicted):
            if truth["page_number"] != prediction["page_number"]:
                continue
            iou = bbox_iou(truth[bbox_key], prediction[bbox_key])
            if iou >= iou_threshold:
                candidates.append((iou, t, p))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    matched, used = {}, set()
    for iou, t, p in candidates:
        if t not in matched and p not in used:
            matched[t] = (p, iou)
            used.add(p)

    rows = []
    for t, truth in enumerate(ground_truth):
        p, iou = matched.get(t, (None, 0.0))
        prediction = predicted[p] if p is not None else None
        rows.append({
            "page_number": truth["page_number"],
            "truth_type": truth["type"],
            "truth_bbox": list(truth[bbox_key]),
            "predicted_type": prediction["type"] if prediction else None,
            "predicted_bbox": list(prediction[bbox_key]) if prediction else None,
            "iou": iou,
            "correct_class": bool(prediction) and prediction["type"] == truth["type"],
        })
    return rows


def class_accuracy(rows):
    """Share of ground-truth regions matched to a prediction of the same class; None without rows."""
    if not rows:
        return None
    return sum(1 for row in rows if row["correct_class"]) / len(rows)

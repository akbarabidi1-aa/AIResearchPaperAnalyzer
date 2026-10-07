"""Compare detected layout regions with a hand-written ground-truth file and print the evaluation table.

    python AIEngine/tools/evaluate_layout.py tests/layout_eval/ground_truth/synthetic_digital.json
    python AIEngine/tools/evaluate_layout.py tests/layout_eval/ground_truth/*.json --layout-threshold 0.5

Ground-truth format and the meaning of the columns: tests/layout_eval/README.md.
A manual check on a handful of pages, not a benchmark. Exit code: 0 = ok, 2 = bad input.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import (  # noqa: E402
    DEFAULT_LAYOUT_THRESHOLD,
    PdfPipelineError,
    PPDocLayoutV3Detector,
    RegionType,
    class_accuracy,
    detect_document_layout,
    match_regions,
    normalize_detections,
    process_pdf,
)
from pipeline.common import pymupdf  # noqa: E402

BBOX_KEYS = {"pdf": "pdf_bbox", "image": "image_bbox"}


def load_ground_truth(path):
    """Returns (source_path, bbox_key, regions) with regions as match_regions expects them."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    bbox_key = BBOX_KEYS.get(data.get("bbox_space"))
    if bbox_key is None:
        raise ValueError(f"{path}: bbox_space must be \"pdf\" or \"image\"")
    types = {t.value for t in RegionType}
    regions = []
    for i, region in enumerate(data.get("regions") or []):
        if region.get("type") not in types:
            raise ValueError(f"{path}: regions[{i}].type is not one of {sorted(types)}")
        regions.append({"page_number": region.get("page_number", 1), "type": region["type"],
                        bbox_key: region["bbox"]})
    source = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), data["source"]))
    return source, bbox_key, regions


def predict(source, detector, threshold, work_dir):
    """Detected regions (dicts) of a PDF, or of a single page image (image coordinates only)."""
    if source.lower().endswith(".pdf"):
        process_pdf(source, work_dir)
        document = detect_document_layout(work_dir, detector, threshold=threshold)["document"]
        return [region for page in document["pages"] for region in page["regions"]]

    pix = pymupdf.Pixmap(source)
    # A bare image has no PDF page behind it: scale 1, so pdf_bbox equals image_bbox.
    page = {"page_number": 1, "image_width": pix.width, "image_height": pix.height,
            "pdf_width": float(pix.width), "pdf_height": float(pix.height), "scale_x": 1.0, "scale_y": 1.0}
    regions, _ = normalize_detections(detector.detect(source, threshold), page, threshold,
                                      detector.label_map, detector.name)
    return [region.to_dict() for region in regions]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate layout detection against ground-truth files.")
    parser.add_argument("ground_truth", nargs="+", help="ground-truth JSON file(s)")
    parser.add_argument("--layout-threshold", type=float, default=DEFAULT_LAYOUT_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="minimum IoU for a prediction to count as a match (default 0.5)")
    args = parser.parse_args(argv)

    paths = [p for pattern in args.ground_truth for p in (glob.glob(pattern) or [pattern])]
    detector = PPDocLayoutV3Detector()
    tmp = tempfile.mkdtemp(prefix="airpa-layout-eval-")
    all_rows = []
    try:
        print("| File | Page | Ground-truth class | Ground-truth bbox | Predicted class | IoU | Correct class |")
        print("|---|---|---|---|---|---|---|")
        for n, path in enumerate(paths):
            source, bbox_key, truth = load_ground_truth(path)
            if bbox_key == "pdf_bbox" and not source.lower().endswith(".pdf"):
                raise ValueError(f"{path}: an image source needs \"bbox_space\": \"image\"")
            predicted = predict(source, detector, args.layout_threshold, os.path.join(tmp, str(n)))
            rows = match_regions(truth, predicted, iou_threshold=args.iou_threshold, bbox_key=bbox_key)
            all_rows.extend(rows)
            for row in rows:
                print("| {} | {} | {} | {} | {} | {:.2f} | {} |".format(
                    os.path.basename(source), row["page_number"], row["truth_type"],
                    [round(v, 1) for v in row["truth_bbox"]], row["predicted_type"] or "(none)",
                    row["iou"], "yes" if row["correct_class"] else "no"))
    except (PdfPipelineError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    matched = [row for row in all_rows if row["predicted_type"]]
    print(f"Ground-truth regions: {len(all_rows)}")
    print(f"Matched (IoU >= {args.iou_threshold}): {len(matched)}")
    if matched:
        print(f"Mean IoU of matched regions: {sum(r['iou'] for r in matched) / len(matched):.3f}")
    accuracy = class_accuracy(all_rows)
    if accuracy is not None:
        print(f"Class accuracy: {accuracy:.3f} ({sum(r['correct_class'] for r in all_rows)}/{len(all_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

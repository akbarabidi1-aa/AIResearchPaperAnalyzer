"""Compare figures.json with ground-truth figures: captions, numbers, types, chart labels and data.

    python AIEngine/tools/evaluate_figures.py temp/figures --truth tests/fixtures/figures/synthetic_figures.json

The first argument is a working directory (or a figures.json file). The ground-truth file has
{"figures": [{"page_number", "pdf_bbox", "figure_type", "caption", "figure_number", "labels", "chart"}]}.
An engineering check on fixtures with known content, not a benchmark.
Exit code: 0 = ok, 2 = a file could not be read.
"""
import argparse
import json
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import evaluate_figures, validate_figures_document  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Accuracy of extracted figures against ground truth (Phase 7).")
    parser.add_argument("figures", help="working directory or figures.json")
    parser.add_argument("--truth", required=True, help="ground-truth JSON file")
    args = parser.parse_args(argv)

    try:
        path = args.figures
        if os.path.isdir(path):
            path = os.path.join(path, "metadata", "figures.json")
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
        problems = validate_figures_document(document)
        if problems:
            raise ValueError(f"Invalid {path}: " + "; ".join(problems[:5]))
        with open(args.truth, encoding="utf-8") as fh:
            truth = json.load(fh)["figures"]
        result = evaluate_figures(truth, document)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    def mark(value):
        return "yes" if value else "NO"

    def share(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"{'page':<5}{'figure':<10}{'kind':<34}{'type':<9}{'ok':>4}{'caption':>9}{'number':>8}")
    for row in result["rows"]:
        print(f"{row['page_number']:<5}{row['figure_id'] or '(none)':<10}{row['kind'] or '-':<34}"
              f"{row['figure_type'] or '-':<9}{mark(row['type_match']):>4}{mark(row['caption_match']):>9}"
              f"{mark(row['number_match']):>8}")
        chart = row["chart"]
        if chart:
            error = "n/a" if chart["mae"] is None else f"MAE {chart['mae']:.4g}, RMSE {chart['rmse']:.4g}"
            print(f"       chart type {mark(chart['chart_type_match'])}; titles {chart['titles_correct']}/"
                  f"{chart['titles_expected']}; ticks {chart['ticks_correct']}/{chart['ticks_expected']}; "
                  f"legend {chart['legend_correct']}/{chart['legend_expected']}; printed values "
                  f"{chart['labelled_exact']}/{chart['labelled_expected']}; estimates {chart['points_found']}/"
                  f"{chart['points_expected']} ({error})")
        elif row["labels_expected"]:
            print(f"       labels {row['labels_found']}/{row['labels_expected']}")
    print()
    print(f"Figures: {result['found']} of {result['figures']} found")
    print(f"Figure-type accuracy: {share(result['type_accuracy'])}")
    print(f"Caption match accuracy: {share(result['caption_accuracy'])}")
    print(f"Figure number accuracy: {share(result['number_accuracy'])}")
    print(f"Label accuracy (generic figures): {share(result['label_accuracy'])}")
    print(f"Charts: {result['charts']}")
    print(f"Chart type accuracy: {share(result['chart_type_accuracy'])}")
    print(f"Axis label accuracy: titles {share(result['axis_title_accuracy'])}, "
          f"tick labels {share(result['tick_label_accuracy'])}")
    print(f"Legend text accuracy: {share(result['legend_accuracy'])}")
    print(f"Exact value accuracy (values printed in the chart): {share(result['exact_value_accuracy'])} "
          f"({result['labelled_values']} values)")
    print(f"Visual estimates: {share(result['estimated_points_found'])} of {result['estimated_points']} values "
          f"matched; MAE {share(result['normalized_mae'])}, RMSE {share(result['normalized_rmse'])} "
          f"(as a share of the y-axis span)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

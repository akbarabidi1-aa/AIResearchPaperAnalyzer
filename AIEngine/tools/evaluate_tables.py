"""Compare tables.json with ground-truth tables: row/column counts, cell accuracy, structure.

    python AIEngine/tools/evaluate_tables.py temp/tables --truth tests/fixtures/tables/synthetic_tables.json

The first argument is a working directory (or a tables.json file). The ground-truth file has
{"tables": [{"page_number", "pdf_bbox", "grid", "merged_cells", "header_rows", "caption"}]}.
An engineering check on fixtures with known content, not a benchmark.
Exit code: 0 = ok, 2 = a file could not be read.
"""
import argparse
import json
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import evaluate_tables, validate_tables_document  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Accuracy of extracted tables against ground truth (Phase 5).")
    parser.add_argument("tables", help="working directory or tables.json")
    parser.add_argument("--truth", required=True, help="ground-truth JSON file")
    args = parser.parse_args(argv)

    try:
        path = os.path.join(args.tables, "metadata", "tables.json") if os.path.isdir(args.tables) else args.tables
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
        problems = validate_tables_document(document)
        if problems:
            raise ValueError(f"Invalid {path}: " + "; ".join(problems[:5]))
        with open(args.truth, encoding="utf-8") as fh:
            truth = json.load(fh)["tables"]
        result = evaluate_tables(truth, document)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    def mark(value):
        return "-" if value is None else ("yes" if value else "NO")

    print(f"{'page':<5}{'table':<11}{'source':<8}{'rows':>7}{'cols':>7}{'exact cells':>13}{'text':>7}"
          f"{'struct':>8}{'header':>8}{'caption':>9}")
    for row in result["rows"]:
        print(f"{row['page_number']:<5}{row['table_id'] or '(none)':<11}{row['source'] or '-':<8}"
              f"{row['predicted_rows']:>3}/{row['expected_rows']:<3}{row['predicted_columns']:>3}/{row['expected_columns']:<3}"
              f"{row['exact_cells']:>8}/{row['total_cells']:<4}{row['cell_text_accuracy']:>7.3f}"
              f"{row['structure_accuracy']:>8.3f}{mark(row['header_rows_match']):>8}{mark(row['caption_match']):>9}")
    print()
    print(f"Tables: {result['found']} of {result['tables']} found")
    print(f"Row count accuracy: {result['row_count_accuracy']:.3f}")
    print(f"Column count accuracy: {result['column_count_accuracy']:.3f}")
    print(f"Exact cell accuracy: {result['exact_cell_accuracy']:.4f} ({result['exact_cells']}/{result['total_cells']})")
    print(f"Cell text accuracy: {result['cell_text_accuracy']:.4f}")
    print(f"Structure accuracy: {result['structure_accuracy']:.3f}")
    for label, key in (("Header rows", "header_accuracy"), ("Captions", "caption_accuracy")):
        if result[key] is not None:
            print(f"{label} correct: {result[key]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

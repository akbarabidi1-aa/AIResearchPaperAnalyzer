"""Compare equations.json with ground-truth equations: exact match, character error rate, numbers.

    python AIEngine/tools/evaluate_equations.py temp/equations --truth tests/fixtures/equations/synthetic_equations.json

The first argument is a working directory (or an equations.json file). The ground-truth file has
{"equations": [{"page_number", "pdf_bbox", "latex", "equation_number"}]}. Formulas are compared in a
canonical spelling (see pipeline.math_eval.canonical_latex).
An engineering check on fixtures with known content, not a benchmark.
Exit code: 0 = ok, 2 = a file could not be read.
"""
import argparse
import json
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import evaluate_equations, validate_equations_document  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Accuracy of recognized equations against ground truth (Phase 6).")
    parser.add_argument("equations", help="working directory or equations.json")
    parser.add_argument("--truth", required=True, help="ground-truth JSON file")
    args = parser.parse_args(argv)

    try:
        path = args.equations
        if os.path.isdir(path):
            path = os.path.join(path, "metadata", "equations.json")
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
        problems = validate_equations_document(document)
        if problems:
            raise ValueError(f"Invalid {path}: " + "; ".join(problems[:5]))
        with open(args.truth, encoding="utf-8") as fh:
            truth = json.load(fh)["equations"]
        result = evaluate_equations(truth, document)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    def mark(value):
        return "-" if value is None else ("yes" if value else "NO")

    print(f"{'page':<5}{'equation':<10}{'kind':<24}{'status':<20}{'exact':>6}{'CER':>7}{'number':>8}")
    for row in result["rows"]:
        print(f"{row['page_number']:<5}{row['equation_id'] or '(none)':<10}{row['kind'] or '-':<24}"
              f"{row['status'] or 'not found':<20}{mark(row['exact_match']):>6}{row['cer']:>7.3f}"
              f"{mark(row['number_match']):>8}")
        if not row["exact_match"]:
            print(f"       expected  {row['expected']}")
            print(f"       predicted {row['predicted']}")
    print()
    print(f"Equations: {result['found']} of {result['equations']} found")
    print(f"Recognition success rate: {result['recognition_success_rate']:.3f} "
          f"({result['recognized']}/{result['equations']})")
    print(f"Exact LaTeX match: {result['exact_match_rate']:.3f} ({result['exact_matches']}/{result['equations']})")
    print(f"Strict match (whitespace removed only): {result['strict_matches']}/{result['equations']}")
    print(f"Character error rate: {result['cer']:.4f} ({result['edits']}/{result['characters']})")
    print(f"Normalized LaTeX accuracy: {result['normalized_latex_accuracy']:.4f}")
    if result["number_accuracy"] is not None:
        print(f"Equation numbers: {result['numbers_correct']}/{result['numbers_expected']} found, "
              f"accuracy {result['number_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
